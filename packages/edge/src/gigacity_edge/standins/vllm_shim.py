"""A/B stand-in for C's service, forwarding requests to a vLLM OpenAI-compatible server.

Frames are sent as separate images with capture times in the prompt (decisions #9,
#12). Results come from an A/B test server, not C's serving measurements.

Run: VLLM_API_KEY=... python -m gigacity_edge.standins.vllm_shim \
         --port 8200 --vllm-url https://<pod>-8000.proxy.runpod.net
"""

import argparse
import json
import math
import os
import string
import time
import urllib.request
from http.server import ThreadingHTTPServer

from gigacity_contracts import InferenceRequest, InferenceResult
from gigacity_edge.standins._http import make_server, unix_ms

DEFAULT_MODEL = "qwen2.5-vl-7b"
DEFAULT_MODEL_ID = "qwen2.5-vl-7b-instruct@cc594898 vllm-v0.11.0 ab-test-server"
ANSWER_SUFFIX = " Answer only Yes or No."
# Used when a request carries no query (live monitoring, C/D plan section 3.1).
DEFAULT_QUERY = "Did any traffic collision occur during the shown observation window?"

# Prompt variants. "v1" is the November 10-11 prompt and must stay unchanged.
PROMPT_VARIANTS: dict[str, dict[str, str | None]] = {
    "v1": {"preamble": None, "suffix": ANSWER_SUFFIX},
    "v2-context": {
        "preamble": "These are frames from a fixed traffic CCTV camera, in time order.",
        "suffix": ANSWER_SUFFIX,
    },
    "v3-careful": {
        "preamble": None,
        "suffix": " Look carefully at every frame for any collision or crash, even "
        "brief or partly visible. Answer only Yes or No.",
    },
}
TOP_LOGPROBS = 20


def build_chat_request(
    request: InferenceRequest,
    model: str,
    variant: str = "v1",
    logprobs: bool = False,
) -> dict:
    """OpenAI chat payload: interleaved frame captions and images, then the question.

    With the defaults the payload is identical to the November 10-11 runs.
    ``logprobs`` only asks vLLM to report token probabilities; with temperature 0 it
    does not change the generated answer.
    """
    prompt = PROMPT_VARIANTS[variant]
    content: list[dict] = []
    if prompt["preamble"]:
        content.append({"type": "text", "text": prompt["preamble"]})
    for index, frame in enumerate(request.frames, start=1):
        content.append(
            {
                "type": "text",
                "text": f"Frame {index} at {frame.capture_ms / 1000:.1f} s:",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": frame.data_ref},
            }
        )
    question = request.query or DEFAULT_QUERY
    content.append({"type": "text", "text": question + prompt["suffix"]})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 16,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = TOP_LOGPROBS
    return payload


def parse_answer(text: str | None) -> str:
    """Map raw model text to yes, no, or unparsed using its first word."""
    words = (text or "").strip().lower().split()
    if not words:
        return "unparsed"
    first = words[0].strip(string.punctuation)
    return first if first in ("yes", "no") else "unparsed"


def yes_probability(response: dict) -> float | None:
    """P(yes) / (P(yes) + P(no)) from the first generated token's top logprobs."""
    try:
        first = response["choices"][0]["logprobs"]["content"][0]
        candidates = first.get("top_logprobs") or [first]
    except (KeyError, IndexError, TypeError):
        return None
    totals = {"yes": 0.0, "no": 0.0}
    for candidate in candidates:
        word = str(candidate.get("token", "")).strip().strip(string.punctuation).lower()
        if word in totals and candidate.get("logprob") is not None:
            totals[word] += math.exp(candidate["logprob"])
    if totals["yes"] <= 0 or totals["no"] <= 0:
        return None
    return totals["yes"] / (totals["yes"] + totals["no"])


def call_vllm(url: str, api_key: str | None, payload: dict, timeout_s: float) -> dict:
    # RunPod's proxy sits behind Cloudflare, which rejects urllib's default
    # User-Agent with HTTP 403 (error 1010).
    headers = {"Content-Type": "application/json", "User-Agent": "gigacity-edge/0.1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    http_request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=timeout_s) as response:
        return json.loads(response.read())


def shim_result(
    request: InferenceRequest,
    received_at_unix_ms: int,
    vllm_url: str,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    model_id: str = DEFAULT_MODEL_ID,
    timeout_s: float = 120.0,
    variant: str = "v1",
    logprobs: bool = True,
) -> InferenceResult:
    payload = build_chat_request(request, model, variant, logprobs)
    started = time.perf_counter()
    common = {
        "run_id": request.run_id,
        "window_id": request.window.window_id,
        "model_id": model_id,
        "frames_used": len(request.frames),
        "queue_duration_ms": None,
        "received_at_unix_ms": received_at_unix_ms,
        "is_synthetic": False,
    }
    try:
        response = call_vllm(vllm_url, api_key, payload, timeout_s)
        text = response["choices"][0]["message"]["content"]
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        # OSError covers URLError/HTTPError, timeouts, and dropped connections.
        return InferenceResult(
            **common,
            status="failed",
            inference_duration_ms=round((time.perf_counter() - started) * 1000),
            completed_at_unix_ms=unix_ms(),
        )
    usage = response.get("usage") or {}
    return InferenceResult(
        **common,
        status="completed",
        answer_text=text,
        answer=parse_answer(text),
        yes_probability=yes_probability(response),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        inference_duration_ms=round((time.perf_counter() - started) * 1000),
        completed_at_unix_ms=unix_ms(),
    )


def build_server(
    vllm_url: str,
    api_key: str | None,
    host: str = "127.0.0.1",
    port: int = 8200,
    model: str = DEFAULT_MODEL,
    model_id: str = DEFAULT_MODEL_ID,
    variant: str = "v1",
    logprobs: bool = True,
) -> ThreadingHTTPServer:
    return make_server(
        host,
        port,
        lambda request, received: shim_result(
            request,
            received,
            vllm_url,
            api_key,
            model,
            model_id,
            variant=variant,
            logprobs=logprobs,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--prompt-variant", choices=sorted(PROMPT_VARIANTS), default="v1"
    )
    parser.add_argument(
        "--no-logprobs", action="store_true", help="do not request token logprobs"
    )
    args = parser.parse_args()
    model_id = args.model_id
    if args.prompt_variant != "v1":
        model_id = f"{model_id} prompt={args.prompt_variant}"
    server = build_server(
        args.vllm_url,
        os.environ.get("VLLM_API_KEY"),
        args.host,
        args.port,
        args.model,
        model_id,
        variant=args.prompt_variant,
        logprobs=not args.no_logprobs,
    )
    print(f"vllm-shim (A/B test server) on http://{args.host}:{server.server_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
