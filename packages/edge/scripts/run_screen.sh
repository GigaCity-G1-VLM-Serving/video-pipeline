#!/usr/bin/env bash
# Run one A/B screen against a vLLM server: smoke test, selectors in parallel, then the
# event-oracle diagnostic. Raw outputs go to results/ (gitignored); export summaries
# with gigacity_edge.analysis.compare afterwards.
#
# Required env: POD_URL (vLLM base URL), VLLM_API_KEY (never logged), RUN_PREFIX.
# Optional env: MODEL, MODEL_ID, PROMPT_VARIANT (v1), PILOT (data/pilot80),
#   SELECTORS ("uniform motion clip-generic"), BUDGETS ("4 8 16"), PORT (8500),
#   CONCURRENCY (4), SERVER_CONFIG (packages/edge/scripts/qwen3-vl-30b-a3b-fp8.server.json),
#   TAR (data/tar/temporal_localization.json), VIDEOS (data/videos), SKIP_ORACLE (0).
#
# Example:
#   POD_URL=https://<pod>-8000.proxy.runpod.net VLLM_API_KEY=... RUN_PREFIX=q30lp \
#     PROMPT_VARIANT=v1 bash packages/edge/scripts/run_screen.sh
set -uo pipefail

: "${POD_URL:?set POD_URL}" "${VLLM_API_KEY:?set VLLM_API_KEY}" "${RUN_PREFIX:?set RUN_PREFIX}"
MODEL=${MODEL:-qwen3-vl-30b-a3b}
MODEL_ID=${MODEL_ID:-"qwen3-vl-30b-a3b-instruct-fp8@d9748a51 vllm-v0.11.0 ab-test-server"}
PROMPT_VARIANT=${PROMPT_VARIANT:-v1}
PILOT=${PILOT:-data/pilot80}
SELECTORS=${SELECTORS:-"uniform motion clip-generic"}
BUDGETS=${BUDGETS:-"4 8 16"}
PORT=${PORT:-8500}
CONCURRENCY=${CONCURRENCY:-4}
SERVER_CONFIG=${SERVER_CONFIG:-packages/edge/scripts/qwen3-vl-30b-a3b-fp8.server.json}
TAR=${TAR:-data/tar/temporal_localization.json}
VIDEOS=${VIDEOS:-data/videos}
SKIP_ORACLE=${SKIP_ORACLE:-0}
URL="http://127.0.0.1:${PORT}"

cd "$(git rev-parse --show-toplevel)"
mkdir -p results

VLLM_API_KEY="$VLLM_API_KEY" uv run --all-extras python -m gigacity_edge.standins.vllm_shim \
  --port "$PORT" --vllm-url "$POD_URL" --model "$MODEL" --model-id "$MODEL_ID" \
  --prompt-variant "$PROMPT_VARIANT" > "results/${RUN_PREFIX}-shim.log" 2>&1 &
SHIM=$!
trap 'kill "$SHIM" 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do curl -sf "$URL/health" >/dev/null && break; sleep 1; done

run() {
  uv run --all-extras python -m gigacity_edge.run --pilot "$PILOT/pilot_questions.jsonl" \
    --videos "$VIDEOS" --url "$URL" --server-config "$SERVER_CONFIG" "$@"
}

echo "== smoke"
run --selector uniform --budgets 4 --run-id "${RUN_PREFIX}-smoke" \
  --out "results/${RUN_PREFIX}-smoke" --limit 1 || exit 1
grep -q '"status":"completed"' "results/${RUN_PREFIX}-smoke/results.jsonl" \
  || { echo "smoke test failed"; exit 1; }

echo "== selectors: $SELECTORS"
started=$(date +%s)
pids=()
for selector in $SELECTORS; do
  # shellcheck disable=SC2086
  run --selector "$selector" --budgets $BUDGETS --run-id "${RUN_PREFIX}-${selector}" \
    --out "results/${RUN_PREFIX}-${selector}" --concurrency "$CONCURRENCY" &
  pids+=("$!")
done
# Wait only for the selector jobs: a bare `wait` would also wait for the shim forever.
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "selectors finished in $(( $(date +%s) - started ))s (status $status)"

if [ "$SKIP_ORACLE" != "1" ]; then
  echo "== event oracle (uses labels; diagnostic only)"
  uv run --all-extras python -m gigacity_edge.analysis.oracle \
    --pilot "$PILOT/pilot_questions.jsonl" --tar "$TAR" --videos "$VIDEOS" --url "$URL" \
    --run-id "${RUN_PREFIX}-oracle" --out "results/${RUN_PREFIX}-oracle" \
    --server-config "$SERVER_CONFIG" --concurrency "$CONCURRENCY" || status=1
fi
echo "DONE (status $status)"
exit "$status"
