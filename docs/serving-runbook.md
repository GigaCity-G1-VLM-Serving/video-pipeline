# C serving runbook (proposed)

**Status: proposed, not yet run on a GPU.** Settings here are pinned to match the
decision log and A/B's published runs so C's numbers are comparable with theirs.

## What this measures, and why it is not a repeat of A/B's work

A/B have published three result sets. Every one of them carries the line
"A/B test-server results, not C's serving measurements," and the reasons are
stated in those docs: latencies include the RunPod proxy round trip, and in the
November 11 screen three selectors ran in parallel, so "latencies are not load
measurements."

The only serving-load numbers that exist are for Qwen2.5-VL-7B on an A40
(November 10, section 4). A/B now propose Qwen3-VL-30B-A3B as the primary model.
**No load characterisation exists for the proposed model**, and that is C's
workstream:

- latency and time-to-first-token per frame budget, measured server-side
- throughput and queueing against concurrent requests
- deadline compliance at the 5 s deadline of decision 18
- cameras supported per GPU
- prompt tokens per budget, which is the real GPU cost the controller trades against

This also matters for whether C's contribution exists at all. Under
Qwen2.5-VL-7B, accuracy did not improve with more frames, so a frame budget had
nothing to optimise. Under Qwen3-VL-30B-A3B, uniform accuracy rises 60 → 63 → 68%
at 4 → 8 → 16 frames. The quality-versus-cost curve is real again, and the cost
half of it is unmeasured.

## 0. Before deploying

**GPU.** Qwen3-VL-30B-A3B-FP8 needs about 30 GB and, per A/B's note, an **Ada or
Hopper** card for FP8. The A100 in the original plan is Ampere and does not
qualify. Use an **L40S 48 GB** ($1.09/hr) — Ada, enough memory, and A/B already
ran this model on one, so the numbers cross-check.

**Cost.** Per-second billing, one-minute minimum. A/B's three runs cost about
$4.40 in total. Budget one to two hours.

**Deviation to record.** Decision 7 pins the container image
`vllm/vllm-openai:v0.11.0`. This runbook installs `vllm==0.11.0` into a PyTorch
template instead, which is the same vLLM version on a different base image.
Report that in results; switch to the image itself if any number looks off.

## 1. Deploy the pod

Pods → Deploy. No network volume — the pod's own volume disk at `/workspace`
persists while the pod exists, including while stopped.

- **GPU:** L40S 48 GB
- **Template:** RunPod PyTorch
- **Volume disk:** 80 GB (the model is ~30 GB)
- **Container disk:** 40 GB
- **Expose HTTP Ports:** empty
- **Region:** any

Do not set RunPod's `allowedCudaVersions` filter. Decision 7 records that pod
creation failed whenever it was set.

## 2. Install

```bash
export HF_HOME=/workspace/hf
echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc

apt-get update && apt-get install -y ffmpeg tmux git
pip install vllm==0.11.0

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

## 3. Start the server

```bash
tmux new -s vllm

vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":16,"video":0}' \
  --no-enable-prefix-caching \
  --seed 0
```

Detach with `Ctrl-b` then `d`. First launch downloads ~30 GB.

Prefix caching is off to match decision 7 and A/B's runs — leaving it on would
make repeated identical payloads look artificially fast, which is exactly the
wrong thing when measuring serving load.

No resolution flag is needed here: the client resizes to 448×252 before sending
(decision 10), so the server receives frames already at the pinned size.

## 4. Code and videos

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /workspace
git clone https://github.com/GigaCity-G1-VLM-Serving/video-pipeline
cd video-pipeline && git checkout feat/central-serving-smoke && uv sync --locked

cd /workspace
hf download Open-Space-Reasoning/AccidentBench --repo-type dataset \
  land_space/long/videos/000001.mp4 \
  land_space/long/videos/000025.mp4 \
  land_space/medium/videos/xWQgF70Imy4.mp4 \
  --local-dir /workspace/videos/Accident-Bench
```

Three videos is enough. This measures serving cost, not answer quality, so the
payloads only need to be realistic — 448×252 JPEGs at quality 85, which the
client produces.

## 5. Wait for the server

```bash
curl -s localhost:8000/health -o /dev/null -w '%{http_code}\n'
curl -s localhost:8000/v1/models | python -m json.tool
```

## 6. Latency and tokens per budget

```bash
cd /workspace/video-pipeline
V=/workspace/videos/Accident-Bench/land_space

uv run python -m gigacity_central.smoke \
  --video $V/long/videos/000001.mp4 --window-start-s 50 \
  --budgets 4,8,16 --repeats 10 --deadline-ms 5000 \
  --out results/budget-sweep.jsonl
```

`--repeats 10` gives enough samples for a median. `--deadline-ms 5000` applies
decision 18's deadline so `on_time` is recorded per request.

## 7. Concurrency sweep — the main result

```bash
for k in 1 2 4 8 16; do
  uv run python -m gigacity_central.smoke \
    --video $V/long/videos/000001.mp4 --window-start-s 50 \
    --budgets 8 --concurrency "$k" --repeats 5 --deadline-ms 5000 \
    --out "results/conc-$k.jsonl"
done
```

Then repeat at `--budgets 4` and `--budgets 16` if time allows. That gives
deadline compliance as a function of both load and budget, which is what the
load-aware controller needs in order to choose between them.

Cameras per GPU falls out of throughput: at one request per camera per 10 s
window, sustained requests/s × 10 is the camera count. A/B got about 35 cameras
for Qwen2.5-VL-7B on an A40; the 30B model will be lower, and by how much is the
number nobody has.

## 8. Stop the pod

Pods → your pod → **Stop**. Ends GPU billing, keeps `/workspace`.

```bash
cp -r results /workspace/results-$(date +%Y%m%d)
```

`results/` is gitignored, so raw JSONL stays out of the repo; publish a summary
under `docs/results/` the way A/B do, with the same status caveat about what the
numbers are and are not.

## 9. Open items this depends on

- **Model choice.** Decision 3 pins Qwen2.5-VL-7B; A/B propose Qwen3-VL-30B-A3B.
  This runbook assumes the proposal is adopted. If the team keeps Qwen2.5-VL-7B,
  the frames-versus-accuracy curve stays flat and the budget controller has
  little to optimise — that is a question for the team, not a detail.
- **GPU for the reported runs.** The plan says A100 40 GB, which cannot serve the
  FP8 model. Whichever card the team standardises on determines "cameras per GPU".
- **Deadline start.** Decision 18 sets 5 s after the window ends; `--deadline-ms`
  here measures from send time, which is close but not identical. Worth
  reconciling before results are published.
