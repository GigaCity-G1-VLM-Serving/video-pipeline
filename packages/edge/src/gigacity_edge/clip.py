"""CLIP ViT-B/32 frame and text embeddings via ONNX Runtime (optional ``clip`` extra).

Weights are downloaded on first use from a pinned Hugging Face revision into
``data/models/clip`` (override with ``GIGACITY_CLIP_DIR``) and never committed.

Preprocessing resizes the *full* frame to 224x224 without a center crop, so the
whole camera view (including crashes near the frame edges) is scored. This differs
from CLIP's training-time crop and slightly distorts the aspect ratio.
"""

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

REPO = "Xenova/clip-vit-base-patch32"
REVISION = "6b7c523703ae26e39be4b03774966f8571c7f523"
FILES = {
    "text": "onnx/text_model.onnx",
    "vision": "onnx/vision_model.onnx",
    "tokenizer": "tokenizer.json",
}
IMAGE_SIZE = 224
CONTEXT_LENGTH = 77
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def cache_dir() -> Path:
    return Path(os.environ.get("GIGACITY_CLIP_DIR", "data/models/clip"))


def ensure_files(directory: Path | None = None) -> dict[str, Path]:
    """Download the pinned model files if missing; return local paths."""
    directory = directory or cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, remote in FILES.items():
        local = directory / Path(remote).name
        if not local.exists() or local.stat().st_size == 0:
            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}"
            partial = local.with_suffix(local.suffix + ".part")
            urllib.request.urlretrieve(url, partial)
            partial.replace(local)
        paths[key] = local
    return paths


def preprocess(images: list[np.ndarray]) -> np.ndarray:
    """BGR frames -> normalized NCHW float32 batch at 224x224 (full frame, no crop)."""
    batch = np.empty((len(images), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for index, image in enumerate(images):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        scaled = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        batch[index] = scaled.transpose(2, 0, 1)
    return batch


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class ClipScorer:
    """Embeds frames and texts into CLIP's shared, L2-normalized space."""

    def __init__(self, directory: Path | None = None, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        paths = ensure_files(directory)
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        providers = ["CPUExecutionProvider"]
        self._text = ort.InferenceSession(
            str(paths["text"]), options, providers=providers
        )
        self._vision = ort.InferenceSession(
            str(paths["vision"]), options, providers=providers
        )
        self._tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
        self._text_inputs = {i.name for i in self._text.get_inputs()}
        if "input_ids" not in self._text_inputs:
            raise ValueError(f"unexpected text inputs: {sorted(self._text_inputs)}")
        self._vision_input = self._vision.get_inputs()[0].name
        self._text_cache: dict[str, np.ndarray] = {}

    def _token_ids(self, text: str) -> list[int]:
        ids = self._tokenizer.encode(text).ids
        if len(ids) > CONTEXT_LENGTH:
            ids = ids[: CONTEXT_LENGTH - 1] + [ids[-1]]  # keep the end token
        return ids

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """One embedding per text, cached. Texts are encoded one at a time, which
        avoids padding: the exported model takes no attention mask."""
        rows = []
        for text in texts:
            if text not in self._text_cache:
                ids = np.array([self._token_ids(text)], dtype=np.int64)
                feeds = {"input_ids": ids}
                if "attention_mask" in self._text_inputs:
                    feeds["attention_mask"] = np.ones_like(ids)
                output = self._text.run(None, feeds)[0]
                self._text_cache[text] = l2_normalize(output)[0]
            rows.append(self._text_cache[text])
        return np.stack(rows)

    def encode_images(
        self, images: list[np.ndarray], batch_size: int = 16
    ) -> np.ndarray:
        """L2-normalized embeddings for BGR frames."""
        outputs = []
        for start in range(0, len(images), batch_size):
            batch = preprocess(images[start : start + batch_size])
            outputs.append(self._vision.run(None, {self._vision_input: batch})[0])
        if not outputs:
            return np.empty((0, 512), dtype=np.float32)
        return l2_normalize(np.concatenate(outputs))


_SCORER: ClipScorer | None = None


def get_scorer() -> ClipScorer:
    """Lazily created process-wide scorer."""
    global _SCORER
    if _SCORER is None:
        _SCORER = ClipScorer()
    return _SCORER
