"""Pilot-list selection logic on synthetic annotation items (no network)."""

from gigacity_edge.pilot import (
    assign_splits,
    build_rows,
    eligible_videos,
    shuffled,
    strip_answer_suffix,
)

ITEMS = [
    {
        "video_id": "Accident-Bench/a.mp4",
        "question": "Q1?\nAnswer with Yes or No.",
        "answer": "Yes",
    },
    {
        "video_id": "Accident-Bench/a.mp4",
        "question": "Q2?\nAnswer with Yes or No.",
        "answer": "No",
    },
    {
        "video_id": "Accident-Bench/a.mp4",
        "question": "Q3?\nAnswer with Yes or No.",
        "answer": "Yes",
    },
    {
        "video_id": "Accident-Bench/b.mp4",
        "question": "Q4?\nAnswer with Yes or No.",
        "answer": "No",
    },
    {
        "video_id": "Accident-Bench/b.mp4",
        "question": "Q5?\nAnswer with Yes or No.",
        "answer": "Yes",
    },
    {
        "video_id": "Accident-Bench/c.mp4",
        "question": "Q6?\nAnswer with Yes or No.",
        "answer": "Yes",
    },
    {
        "video_id": "HTV/d.mp4",
        "question": "Q7?\nAnswer with Yes or No.",
        "answer": "Yes",
    },
    {
        "video_id": "HTV/d.mp4",
        "question": "Q8?\nAnswer with Yes or No.",
        "answer": "No",
    },
]


def test_suffix_is_stripped():
    assert strip_answer_suffix("Is it dry?\nAnswer with Yes or No.") == "Is it dry?"
    assert strip_answer_suffix("No suffix here?") == "No suffix here?"


def test_only_source_videos_with_both_answers_are_eligible():
    assert eligible_videos(ITEMS) == ["Accident-Bench/a.mp4", "Accident-Bench/b.mp4"]


def test_shuffle_is_deterministic_and_order_independent():
    ids = ["x", "y", "z", "w"]
    assert shuffled(ids, 0) == shuffled(list(reversed(ids)), 0)
    assert sorted(shuffled(ids, 0)) == sorted(ids)


def test_splits_are_by_video():
    splits = assign_splits(["a", "b", "c", "d", "e"], 0.4)
    assert [splits[v] for v in "abcde"] == ["calibration"] * 2 + ["evaluation"] * 3


def test_rows_pair_first_yes_and_no_and_keep_labels_separate():
    videos = ["Accident-Bench/a.mp4", "Accident-Bench/b.mp4"]
    questions, labels = build_rows(ITEMS, videos, 0.5)
    assert [q["question"] for q in questions] == ["Q1?", "Q2?", "Q5?", "Q4?"]
    assert [q["question_id"] for q in questions] == ["q0000", "q0001", "q0002", "q0003"]
    assert all("answer" not in q for q in questions)
    assert [label["answer"] for label in labels] == ["Yes", "No", "Yes", "No"]
    assert {q["video_id"]: q["split"] for q in questions} == {
        "Accident-Bench/a.mp4": "calibration",
        "Accident-Bench/b.mp4": "evaluation",
    }
