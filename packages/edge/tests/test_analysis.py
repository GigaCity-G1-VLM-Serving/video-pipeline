"""Comparison, threshold, oracle, and shim helpers, on tiny synthetic records."""

import json
import math
from pathlib import Path

import pytest

from gigacity_contracts import InferenceRequest
from gigacity_edge.analysis import compare, oracle, threshold
from gigacity_edge.run import load_server_config
from gigacity_edge.standins import vllm_shim

EXAMPLES = Path(__file__).resolve().parents[3] / "docs/examples"


def write_run(root: Path, run_id: str, answers: dict[str, tuple], budget: int = 8):
    """answers: question_id -> (answer or None for failed, yes_probability)."""
    root.mkdir(parents=True)
    edges, results = [], []
    for qid, (answer, probability) in answers.items():
        window_id = f"{run_id}:{qid}:b{budget}"
        edges.append(
            {
                "run_id": run_id,
                "window": {
                    "window_id": window_id,
                    "camera_id": "cam",
                    "source_video_id": f"v-{qid}.mp4",
                    "start_ms": 0,
                    "end_ms": 10000,
                },
                "status": "sent",
                "selector": "uniform",
                "frame_budget": budget,
                "selected_capture_ms": [1000],
                "total_bytes": 100,
                "decode_ms": 1,
                "select_ms": 1,
                "encode_ms": 1,
                "round_trip_ms": 50,
                "simulated_network_ms": None,
            }
        )
        results.append(
            {
                "run_id": run_id,
                "window_id": window_id,
                "status": "completed" if answer else "failed",
                "answer_text": answer,
                "answer": answer,
                "yes_probability": probability,
                "model_id": "m",
                "frames_used": 1,
                "received_at_unix_ms": 0,
                "is_synthetic": True,
            }
        )
    for name, rows in (("edge.jsonl", edges), ("results.jsonl", results)):
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def labels(tmp_path):
    rows = [
        {"question_id": f"q{i}", "answer": "Yes" if i < 4 else "No"} for i in range(8)
    ]
    return compare.load_labels(write_jsonl(tmp_path / "labels.jsonl", rows))


@pytest.fixture
def pilot(tmp_path):
    rows = [
        {"question_id": f"q{i}", "split": "calibration" if i % 2 == 0 else "evaluation"}
        for i in range(8)
    ]
    return compare.load_pilot(write_jsonl(tmp_path / "pilot.jsonl", rows))


def test_run_rows_marks_failures_wrong(tmp_path, labels, pilot):
    run = write_run(tmp_path / "a", "a", {"q0": ("yes", 0.9), "q1": (None, None)})
    rows = compare.run_rows(run, labels, pilot)
    assert [(r["question_id"], r["correct"], r["status"]) for r in rows] == [
        ("q0", True, "completed"),
        ("q1", False, "failed"),
    ]
    assert rows[0]["split"] == "calibration"


def test_csv_export_has_no_payloads(tmp_path, labels, pilot):
    run = write_run(tmp_path / "a", "a", {"q0": ("yes", 0.9)})
    out = tmp_path / "out.csv"
    compare.write_csv(compare.run_rows(run, labels, pilot), out)
    header, row = out.read_text().splitlines()
    assert header.split(",") == compare.COLUMNS
    assert "base64" not in row and row.startswith("a,q0,")


def test_sign_test_matches_exact_binomial():
    assert compare.sign_test(0, 0) == 1.0
    assert compare.sign_test(5, 0) == pytest.approx(2 / 32)
    assert compare.sign_test(15, 9) == pytest.approx(
        2 * sum(math.comb(24, k) for k in range(15, 25)) / 2**24
    )


def test_paired_counts_only_shared_questions(tmp_path, labels):
    a = compare.run_rows(
        write_run(tmp_path / "a", "a", {"q0": ("yes", None), "q1": ("no", None)}),
        labels,
        {},
    )
    b = compare.run_rows(
        write_run(tmp_path / "b", "b", {"q0": ("no", None), "q2": ("yes", None)}),
        labels,
        {},
    )
    result = compare.paired(a, b, 8)
    assert result["n"] == 1 and result["only_a"] == 1 and result["only_b"] == 0


def test_threshold_is_fit_on_calibration_only(tmp_path, labels, pilot):
    # Calibration (even ids): yes questions q0, q2 have p=0.3; no questions q4, q6 p=0.1.
    # Evaluation (odd ids) would prefer a different cut; it must not influence the fit.
    answers = {
        "q0": ("no", 0.3),
        "q2": ("no", 0.3),
        "q4": ("no", 0.1),
        "q6": ("no", 0.1),
        "q1": ("no", 0.05),
        "q3": ("no", 0.05),
        "q5": ("no", 0.9),
        "q7": ("no", 0.9),
    }
    run = write_run(tmp_path / "a", "a", answers)
    rows = compare.run_rows(run, labels, pilot)
    fitted = threshold.fit_threshold([r for r in rows if r["split"] == "calibration"])
    assert fitted == pytest.approx(0.3)
    table = threshold.table(rows)
    calibration_fit = next(
        r
        for r in table
        if r["split"] == "calibration" and r["method"] == "fitted threshold"
    )
    assert calibration_fit["balanced"] == 1.0


def test_threshold_handles_missing_probabilities(tmp_path, labels, pilot):
    run = write_run(tmp_path / "a", "a", {"q0": ("yes", None), "q4": ("no", None)})
    rows = compare.run_rows(run, labels, pilot)
    assert threshold.fit_threshold(rows) is None
    assert all(r["method"] == "parsed answer" for r in threshold.table(rows))


def test_oracle_times_stay_inside_widened_interval():
    times = oracle.oracle_times((10_000, 12_000), 60_000)
    assert len(times) == 8 and times == sorted(times)
    assert all(9_000 <= t <= 13_000 for t in times)
    assert oracle.oracle_times((0, 0), 1) == []


def response_with(top):
    return {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": top[0][0],
                            "logprob": top[0][1],
                            "top_logprobs": [
                                {"token": t, "logprob": lp} for t, lp in top
                            ],
                        }
                    ]
                }
            }
        ]
    }


def test_yes_probability_from_top_logprobs():
    response = response_with(
        [("No", math.log(0.6)), ("Yes", math.log(0.3)), (" yes", math.log(0.1))]
    )
    assert vllm_shim.yes_probability(response) == pytest.approx(0.4)
    assert vllm_shim.yes_probability({"choices": [{"message": {}}]}) is None
    assert vllm_shim.yes_probability(response_with([("No", 0.0)])) is None


def example_request() -> InferenceRequest:
    text = (EXAMPLES / "inference_request.json").read_text(encoding="utf-8")
    return InferenceRequest.model_validate_json(text)


def test_v1_payload_is_unchanged():
    payload = vllm_shim.build_chat_request(example_request(), "m")
    assert set(payload) == {"model", "messages", "temperature", "max_tokens"}
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Frame 1 at 1.2 s:"}
    assert content[-1]["text"].endswith("window? Answer only Yes or No.")
    with_logprobs = vllm_shim.build_chat_request(example_request(), "m", "v1", True)
    assert with_logprobs["messages"] == payload["messages"]
    assert with_logprobs["logprobs"] is True and with_logprobs["top_logprobs"] == 20


def test_prompt_variants_change_only_the_text():
    base = vllm_shim.build_chat_request(example_request(), "m")["messages"][0][
        "content"
    ]
    v2 = vllm_shim.build_chat_request(example_request(), "m", "v2-context")
    v2_content = v2["messages"][0]["content"]
    assert v2_content[0]["text"].startswith(
        "These are frames from a fixed traffic CCTV"
    )
    assert v2_content[1:] == base
    v3 = vllm_shim.build_chat_request(example_request(), "m", "v3-careful")
    v3_content = v3["messages"][0]["content"]
    assert v3_content[:-1] == base[:-1]
    assert "Look carefully at every frame" in v3_content[-1]["text"]


def test_server_config_rejects_secrets(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"gpu": "H100", "image": "vllm"}))
    assert load_server_config(good) == {"gpu": "H100", "image": "vllm"}
    assert load_server_config(None) is None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"api_key": "x"}))
    with pytest.raises(ValueError):
        load_server_config(bad)


def test_checked_in_server_config_is_clean():
    path = (
        Path(__file__).resolve().parents[1] / "scripts/qwen3-vl-30b-a3b-fp8.server.json"
    )
    assert load_server_config(path)["model_revision"].startswith("d9748a51")
