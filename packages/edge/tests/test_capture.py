"""Event-capture scoring helpers, with synthetic intervals only."""

import json

import pytest

from gigacity_edge.analysis import capture


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [("00:54", 54000), ("01:02.5", 62500), ("00:00:19.94", 19940), ("7", 7000)],
)
def test_parse_timestamp(stamp, expected):
    assert capture.parse_timestamp(stamp) == expected


def test_capture_stats_with_and_without_tolerance():
    interval = (10_000, 12_000)
    assert capture.capture_stats([9_500, 20_000], interval, 1000) == {
        "hit": True,
        "inside": 1,
        "frames": 2,
    }
    assert capture.capture_stats([9_500, 20_000], interval, 0)["hit"] is False


def test_load_intervals_keeps_first_valid_interval(tmp_path):
    path = tmp_path / "tl.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "video_id": "a.mp4",
                        "answer": '{"start": "00:05", "end": "00:07"}',
                    },
                    {
                        "video_id": "b.mp4",
                        "answer": '{"start": "00:09", "end": "00:09"}',
                    },
                ]
            }
        )
    )
    assert capture.load_intervals(path) == {"a.mp4": (5000, 7000)}


def test_summarize_rates():
    rows = [
        {
            "selector": "u",
            "budget": 4,
            "hit": True,
            "hit_strict": False,
            "inside": 1,
            "frames": 4,
        },
        {
            "selector": "u",
            "budget": 4,
            "hit": False,
            "hit_strict": False,
            "inside": 0,
            "frames": 4,
        },
    ]
    (row,) = capture.summarize(rows)
    assert row["hit_rate"] == 0.5 and row["frames_inside"] == 0.125
