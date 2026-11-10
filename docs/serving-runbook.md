# C serving runbook (proposed)

**Status: proposed, not yet validated on a GPU.** This records the intended
Milestone 1 deployment for workstream C so settings are pinned and reproducible
before baseline numbers are collected. Record the versions and prices you
actually encountered; the figures below were read from vendor pages, not from a
run of our own.

Goal of one pass through this runbook: the model loads, a 10-second window of
real traffic footage produces a logged Yes/No answer, and we have latency,
token counts, and byte counts for the 4/8/16 frame budgets.

---

## 0. Before you touch a GPU

**Accounts.** A RunPod account with credit. No Hugging Face token is needed:
neither Qwen2.5-VL-7B-Instruct nor the TAR annotations are gated. A Kaggle
account with `~/.kaggle/kaggle.json` is needed only if the video source you
pick is Kaggle-hosted.

**Cost.** RunPod bills per second with a one-minute minimum. An A100 80GB runs
roughly $1.19–1.79/hr depending on tier and variant; storage is billed
separately per GB-month. A first pass through this runbook is a couple of
hours. The expensive mistake is leaving a pod running overnight, so read
section 10 before section 1.

**Debug cheap, measure on target.** Qwen2.5-VL-7B in bfloat16 is about 16 GB of
weights, so it fits on much smaller cards. If the pipeline is still being
debugged, do that on a cheaper GPU and switch to the A100 only for the numbers
that get reported. Never mix the two in one results file.

---

## 1. Create the network volume first

A network volume is disk that outlives the pod. Without one, the 16 GB model
download repeats on every launch.

Storage → Network Volumes → create one of about 60 GB. Note its datacenter.

Two constraints that cause most first-time failures: a volume is pinned to one
datacenter and the pod must be launched in that same datacenter, and a volume
is attached at pod creation and cannot be attached to a pod that is already
running. Decide on the volume before deploying.

## 2. Deploy the pod

Pods → Deploy. Set:

- **GPU:** A100 80GB, in the volume's datacenter.
- **Template:** a RunPod PyTorch image (CUDA preinstalled).
- **Network volume:** the one from section 1; it mounts at `/workspace`.
- **Expose HTTP ports:** leave empty for now. See section 6 on why.

Deploy, then open the pod's web terminal, or connect over SSH.

## 3. Confirm the GPU and persist the model cache

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /workspace
```

Point the Hugging Face cache at the volume so weights survive a restart. Put
this in `~/.bashrc` as well, or it is lost on reconnect:

```bash
export HF_HOME=/workspace/hf
echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc
```

## 4. Install vLLM, ffmpeg, uv, and the repo

```bash
pip install -U vllm
apt-get update && apt-get install -y ffmpeg tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace
git clone https://github.com/GigaCity-G1-VLM-Serving/video-pipeline
cd video-pipeline && uv sync --locked

python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
```

Record the vLLM version, torch version, and driver in the run log. The serving
engine is not a research variable, so pin one version and one set of inference
settings for everything reported.

## 5. Start the server

Run it under `tmux` so it survives a dropped connection:

```bash
tmux new -s vllm

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":16,"video":0}' \
  --mm-processor-kwargs '{"min_pixels":50176,"max_pixels":1003520}' \
  --seed 0
```

Detach with `Ctrl-b` then `d`; reattach with `tmux attach -t vllm`. First launch
downloads about 16 GB, so expect several minutes; later launches read the volume
cache and start in well under a minute.

Two flags carry experimental weight and must not be left at defaults:

- `--limit-mm-per-prompt` caps images per request. The default is well below 16,
  so a 16-frame budget is rejected outright without it. Older vLLM builds take
  the `image=16` form instead of JSON; use whichever the installed version
  accepts and record which.
- `--mm-processor-kwargs` bounds per-image resolution. Qwen2.5-VL uses dynamic
  resolution, so without a cap a 1080p frame and a 480p frame produce different
  token counts. TAR videos come from eight source collections at mixed
  resolutions, so leaving this unset makes latency partly a function of source
  resolution rather than of frame budget, and the 4/8/16 comparison stops being
  controlled. `max_pixels` of 1003520 is about 320 visual tokens per image, so a
  16-frame window costs roughly 5k tokens and fits the context above.

Resolution is itself a confound worth measuring once: repeat one budget sweep at
a lower `max_pixels` to show whether collision answers are resolution-sensitive
before fixing a value for the reported runs.

## 6. Confirm it loaded

Wait for `Application startup complete` in the server log, then from a second
shell on the same pod:

```bash
curl -s localhost:8000/health -o /dev/null -w '%{http_code}\n'
curl -s localhost:8000/v1/models | python -m json.tool
curl -s localhost:8000/metrics | grep -E 'num_requests|gpu_cache_usage'
```

**Keep the port unexposed while measuring.** vLLM enforces no authentication by
default, and a port exposed on RunPod gets a public URL at
`https://<pod-id>-8000.proxy.runpod.net`, where anyone who has it can run
inference billed to our account. Nothing in this runbook needs the endpoint to
be public: the client runs on the same pod. Exposing it also puts internet
round-trip time into every measurement, which is exactly the thing we are
trying to measure cleanly.

When A/B eventually needs to reach it, expose port 8000 in the pod config,
restart the server with `--api-key <secret>`, and have callers set
`VLLM_API_KEY` to the same value. A 502 from the proxy almost always means the
server bound to localhost instead of `0.0.0.0`, or the exposed port does not
match `--port`.

## 7. Get the annotations and pick a few videos

The dataset ships annotations only; videos are not redistributed and are
fetched from their original sources, about 150 GB in total, mostly UCF-Crime.
Do not run the full download. Pull the annotations, pick a handful of videos,
then fetch only those from whichever single collection is easiest.

```bash
cd /workspace
hf download nvidia/PhysicalAI-Traffic-Anomaly-Reasoning --repo-type dataset \
  --include "train/*" "DOWNLOADING.md" "download_videos.py" --local-dir tar/
# older huggingface_hub installs: huggingface-cli download ...
```

Inspect the schema before writing any filtering against it:

```bash
python - <<'PY'
import json
from pathlib import Path
for name in ("bcq.json", "temporal_localization.json"):
    data = json.loads((Path("tar/train") / name).read_text())
    print(name, type(data).__name__, len(data))
    first = data[0] if isinstance(data, list) else next(iter(data.items()))
    print(json.dumps(first, indent=2)[:600], "\n")
PY
```

Window labels must come from `train/`, where answers are present. The `test/`
split has its answers redacted and is scored only by the challenge server, so
timestamps taken from there cannot be checked locally.

Pick roughly six videos with an annotated collision and six without, from one
collection, and read `DOWNLOADING.md` for the per-source flags before fetching.

## 8. Run the budget sweep

```bash
cd /workspace/video-pipeline
uv run python -m gigacity_central.smoke \
  --video /workspace/tar/videos/<collection>/<video>.mp4 \
  --base-url http://127.0.0.1:8000 \
  --budgets 4,8,16 --repeats 3 --window-s 10 \
  --window-start-s <a second or two before the annotated collision> \
  --out results/smoke-$(date +%Y%m%d).jsonl
```

Each record carries the window identity fields of the draft shared contract,
the requested budget, actual frame count, per-frame and total encoded bytes,
edge-side decode time, time to first token, total latency, prompt and completion
token counts, and the parsed Yes/No answer. Requested budget and observed bytes
stay separate fields so evaluation can compare what C asked for against what
was actually sent.

Then sweep concurrent requests, which is the load variable the team settled on
in place of camera count:

```bash
for k in 1 2 4 8 16; do
  uv run python -m gigacity_central.smoke --video <video> --budgets 8 \
    --concurrency "$k" --repeats 3 --out "results/conc-$k.jsonl"
done
```

Timings are measured server latencies from a client on the same host. There is
no network emulation here, so they are not end-to-end edge numbers and must not
be reported as such.

## 9. Check the degenerate-answer risk before trusting anything

The pilot found that on crash videos accuracy fell as frames rose and the model
answered No to about nine questions in ten. If that reproduces on the server,
correct-on-time rate is dominated by a constant No and the budget controller has
nothing to optimize, which would undercut the premise of the experiment rather
than be a result of it.

Run the sweep across the positive and negative videos from section 7 and look at
the answer spread the summary prints. If every answer is No, prompt and
resolution are the first suspects, and the finding goes to the team before any
baseline is collected; it changes what Milestone 2 can show.

## 10. Stop the pod

Stopping a pod ends GPU billing; the network volume keeps billing at its
per-GB-month rate, which is what makes the next launch fast. Terminating the pod
and deleting the volume ends all billing but means re-downloading the model.
Copy anything from the container's own disk to `/workspace` first, since only
the volume is persistent. `results/` is gitignored, so the JSONL stays out of
the repo; move what matters onto the volume before stopping.

```bash
cp -r results /workspace/results-$(date +%Y%m%d)
```

## 11. Still to settle

- Response deadline for correct-on-time, and whether it starts at window
  completion or earlier. Undefined at the time of writing.
- Fallback budget when no valid budget exists. C/D's plan proposes 4 frames;
  A/B's draft says the run's fixed default. These need to be the same number.
- Load thresholds separating low, moderate, and high load. To be tuned on the
  development split only, then frozen for reported runs.
- Uplink units: the meeting summary says 5 MB/s, A/B's plan says 5 Mbit/s. That
  is an 8x difference in every bandwidth conclusion.