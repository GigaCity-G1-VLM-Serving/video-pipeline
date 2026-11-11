# Decision log

The canonical record of project decisions. Proposals are discussed in the team
Google Docs; once both teams agree, the outcome is recorded here and in the affected
models and docs. If this log and a Google Doc disagree, this log wins.

Status: **agreed** (both teams or the mentors confirmed), **proposed** (one team's
proposal, awaiting the other), **open** (no proposal yet). Update the status and
date in the same change that implements the decision.

## Scope and workload

| # | Decision | Status | Source and notes |
|---|---|---|---|
| 1 | First workload is traffic collisions; recorded videos are replayed causally as simulated live cameras | agreed | October 27 meeting |
| 2 | Primary dataset is NVIDIA TAR; MetroSurv blocked (videos unreleased); DeepAccident is a backup | agreed | Dataset evaluation, October 2026 |
| 3 | Central model is Qwen2.5-VL-7B-Instruct with frozen weights | agreed | Pilot, proposal, and C/D plan |
| 4 | Milestone 1 is end-to-end whole-video QA; per-window monitoring follows | agreed | Mentor comment on the C/D plan, November 8 |
| 5 | Edge devices are simulated (Orin Nano Super class, ~5 Mbit/s uplink, 4 cameras per node, stressed to 8 and 12) | proposed | A/B answer to C/D open question 6 |
| 6 | Server load is reported as concurrent requests (1, 2, 4, 8, 16); camera count measures edge capacity only | proposed | Team comment on the C/D plan |

## Serving and media

| # | Decision | Status | Source and notes |
|---|---|---|---|
| 7 | Serving engine is vLLM, one pinned version | proposed | C/D plan; mentors asked to confirm. The October 6 project doc named SGLang. A/B test runs use `vllm/vllm-openai:v0.11.0` (CUDA 12.8) with prefix caching off. Pod creation failed whenever RunPod's `allowedCudaVersions` filter was set, although the A40 host it then assigned reports driver 595 (CUDA 13.2); an earlier attempt with a CUDA 12.9 build never served, cause unknown because no logs were captured. Newer releases are worth retrying with logs |
| 8 | Evidence is JPEG frames; clips are not sent | proposed | Matches milestone 1 and the C/D plan; revisit if bandwidth results favor clips |
| 9 | Frames reach the model as separate images with capture times in the prompt, not as "video" input | proposed | vLLM's Qwen2.5-VL video path uses one time interval per video and merges frames in pairs (checked in vLLM and Transformers source, November 10). Separate images keep irregular spacing correct |
| 10 | The edge resizes frames to fit 448×448, keeping aspect ratio, with sides a multiple of 28 (640×360 → 448×252), JPEG quality 85; C applies the same pixel cap so it never resizes again | proposed | Keeps edge byte counts equal to what C receives; about 144 visual tokens per frame |
| 11 | The B → C request is C/D's shared `InferenceRequest`/`SelectedFrame`; B sends it as HTTP JSON to C's service, which owns the prompt and calls vLLM | proposed by C/D, adopted by A/B; awaiting team sign-off | C/D `feature/shared-contracts`, adopted by A/B November 11; its own open questions still apply. Proposed v1 transport: `data_ref` is an inline base64 data URI |
| 11a | The request carries no frame budget; fixed budgets are recorded in `EdgeRecord.frame_budget` and the window ID, and C-issued budgets via `budget_id` | proposed | Gap noted when adopting the shared request; revisit with the budget protocol (#19) |
| 12 | Prompt: "Frame i at t s:" before each image, then the question and "Answer only Yes or No."; temperature 0, at most 16 output tokens | proposed | Must be identical across selectors and budgets |

## Records and evaluation

| # | Decision | Status | Source and notes |
|---|---|---|---|
| 13 | Records are JSONL, one file per component per run, joined on `run_id` and `window_id` | proposed | C/D plan assumes logs; spans can come later |
| 14 | Every window has an edge record and a result status; a window without a completed result counts as a miss | proposed | Draft `EdgeRecord` and `InferenceResult` |
| 15 | Pilot uses TAR train yes/no questions (test answers are hidden), split by source video, with saved IDs and seed | proposed | Temporary A/B pilot list until D's manifest exists |
| 16 | Labels never feed selection, budgeting, or inference; mock and stand-in results are labeled and kept apart from measured serving results | agreed | Project rules in `AGENTS.md` |

## Later milestones

| # | Decision | Status | Source and notes |
|---|---|---|---|
| 17 | Observation windows are 10 s with no overlap | proposed | Proposal default; both plans |
| 18 | Deadline is 5 s after the window ends; C still answers late work and marks it `late` | proposed | Proposal default |
| 19 | C pushes a frame budget (4, 8, or 16) for window N+1 when window N ends; the edge uses 4 if none arrives | proposed | C/D open question 3 |
| 20 | B rate-limits sends to emulate the uplink and records `simulated_network_ms` | proposed | Keeps arrivals at C realistic |
