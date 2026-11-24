"""Adapters between central inference and model-serving APIs."""

import json
from collections.abc import Callable
from functools import partial
from urllib.request import Request, urlopen

from gigacity_contracts import InferenceRequest

Backend = Callable[[InferenceRequest], str | None]
DEFAULT_QUERY = "Did any traffic collision occur during the shown observation window?"


class BackendError(Exception):
    """Expected inference failure reported by an adapter."""


def _chat_payload(request: InferenceRequest, model: str) -> dict:
    # Preserve the A/B baseline prompt and the original capture time of each frame.
    content = []
    for index, frame in enumerate(request.frames, start=1):
        content.append(
            {
                "type": "text",
                "text": f"Frame {index} at {frame.capture_ms / 1000:.1f} s:",
            }
        )
        content.append({"type": "image_url", "image_url": {"url": frame.data_ref}})
    content.append(
        {
            "type": "text",
            "text": (request.query or DEFAULT_QUERY) + " Answer only Yes or No.",
        }
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 16,
    }


def _infer_vllm(
    request: InferenceRequest,
    *,
    endpoint: str,
    model: str,
    api_key: str | None,
    timeout: float,
) -> str | None:
    # RunPod's HTTP proxy rejects urllib's default User-Agent.
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "gigacity-central/0.1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    upstream = Request(
        endpoint,
        data=json.dumps(_chat_payload(request, model)).encode(),
        headers=headers,
    )
    try:
        with urlopen(upstream, timeout=timeout) as response:
            answer = json.load(response)["choices"][0]["message"]["content"]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        raise BackendError("vLLM chat completion failed") from error
    if answer is not None and not isinstance(answer, str):
        raise BackendError("vLLM returned non-text content")
    return answer


def vllm_backend(
    base_url: str, model: str, *, api_key: str | None, timeout: float
) -> Backend:
    """Connect to a vLLM server given its base URL, before the /v1 path."""
    return partial(
        _infer_vllm,
        endpoint=base_url.rstrip("/") + "/v1/chat/completions",
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
