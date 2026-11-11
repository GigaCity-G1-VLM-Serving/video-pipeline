"""CLIP selector tests. Plumbing uses a fake scorer; no model download needed."""

import cv2
import numpy as np
import pytest

from gigacity_edge import selectors
from gigacity_edge.clip import IMAGE_SIZE, MEAN, STD, l2_normalize, preprocess
from gigacity_edge.run import prepare

FPS = 10
SECONDS = 4
BRIGHT_MS = (3000, 3300)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """640x360, 4 s at 10 fps; frames are bright only during BRIGHT_MS."""
    path = tmp_path_factory.mktemp("clip") / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (640, 360)
    )
    for index in range(FPS * SECONDS):
        t = index * 1000 // FPS
        level = 220 if BRIGHT_MS[0] <= t < BRIGHT_MS[1] else 20
        writer.write(np.full((360, 640, 3), level, dtype=np.uint8))
    writer.release()
    return path


class FakeScorer:
    """Image score = mean brightness; text embeddings are a fixed unit vector."""

    def __init__(self):
        self.texts: list[list[str]] = []

    def encode_images(self, images):
        brightness = np.array([[img.mean() / 255.0, 0.0] for img in images])
        return brightness.astype(np.float32)

    def encode_text(self, texts):
        self.texts.append(list(texts))
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


def test_preprocess_shape_and_normalization():
    white = np.full((360, 640, 3), 255, dtype=np.uint8)
    batch = preprocess([white, white])
    assert batch.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch.dtype == np.float32
    expected = (1.0 - MEAN) / STD
    np.testing.assert_allclose(batch[0, :, 10, 10], expected, rtol=1e-5)


def test_preprocess_converts_bgr_to_rgb():
    blue = np.zeros((50, 80, 3), dtype=np.uint8)
    blue[..., 0] = 255  # BGR blue
    batch = preprocess([blue])
    assert batch[0, 2].mean() > batch[0, 0].mean()  # RGB channel 2 is blue


def test_l2_normalize():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out[0], [0.6, 0.8])
    assert np.all(np.isfinite(out))


def test_clip_scores_peak_in_bright_window(video):
    scores = selectors.clip_scores(video, ["a crash"], scan_fps=10, scorer=FakeScorer())
    best_time, _ = max(scores, key=lambda item: item[1])
    assert BRIGHT_MS[0] <= best_time < BRIGHT_MS[1]


def test_clip_selector_picks_relevant_frame(video):
    scorer = FakeScorer()
    times, select_ms = selectors.clip(
        video, 2, 4000, query="Is there a crash?", scan_fps=10, scorer=scorer
    )
    assert len(times) == 2 and times == sorted(times)
    assert any(BRIGHT_MS[0] <= t < BRIGHT_MS[1] for t in times)
    assert scorer.texts == [["Is there a crash?"]]
    assert select_ms >= 0


def test_clip_without_query_uses_generic_prompts(video):
    scorer = FakeScorer()
    selectors.clip(video, 2, 4000, query=None, scan_fps=10, scorer=scorer)
    assert scorer.texts == [list(selectors.GENERIC_PROMPTS)]


def test_select_dispatches_clip_variants(video, monkeypatch):
    calls = []

    def fake_clip(path, n, duration_ms, query=None, scan_fps=2.0, scorer=None):
        calls.append(query)
        return [0], 0

    monkeypatch.setattr(selectors, "clip", fake_clip)
    selectors.select("clip", video, 4, 4000, "q?")
    selectors.select("clip-generic", video, 4, 4000, "q?")
    assert calls == ["q?", None]
    with pytest.raises(ValueError):
        selectors.select("nope", video, 4, 4000)


def test_prepare_passes_query_to_selector(video, monkeypatch):
    seen = []

    def fake_select(name, path, n, duration_ms, query=None):
        seen.append((name, query))
        return [0, 1000], 0

    monkeypatch.setattr("gigacity_edge.run.select", fake_select)
    prepared = prepare(video, "clip", 4, "Did a crash occur?")
    assert seen == [("clip", "Did a crash occur?")]
    assert prepared.times_ms == [0, 1000] and len(prepared.frames) == 2


def test_real_scorer_ranks_text_sensibly():
    """Runs only with the clip extra installed and the model cached locally."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from gigacity_edge.clip import ClipScorer, cache_dir

    if not (cache_dir() / "vision_model.onnx").exists():
        pytest.skip("CLIP weights not downloaded")
    scorer = ClipScorer()
    text = scorer.encode_text(["a red image", "a green image"])
    red = np.zeros((224, 224, 3), dtype=np.uint8)
    red[..., 2] = 255  # BGR red
    image = scorer.encode_images([red])
    np.testing.assert_allclose(np.linalg.norm(image, axis=1), 1.0, rtol=1e-5)
    similarity = image @ text.T
    assert similarity[0, 0] > similarity[0, 1]
