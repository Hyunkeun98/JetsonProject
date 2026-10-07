import pytest

from jetson_app.segments import SegmentsError, load_segments
from jetson_app.timeparse import parse_dx1_timestamp


def _write(tmp_path, text):
    path = tmp_path / "segments.yaml"
    path.write_text(text, encoding="utf-8")
    return path


GOOD = """
train:
  - {start: "2026-10-02T11:15:00+09:00", end: "2026-10-02T11:32:50+09:00"}
normal_eval:
  - {start: "2026-10-02T12:11:00+09:00", end: "2026-10-02T12:40:00+09:00"}
anomaly:
  - {name: touch_1, start: "2026-10-02T13:00:00+09:00", end: "2026-10-02T13:01:00+09:00"}
  - {start: "2026-10-02T13:10:00+09:00", end: "2026-10-02T13:11:00+09:00"}
"""


def test_parses_all_kinds_with_timezone_applied(tmp_path):
    segments = load_segments(_write(tmp_path, GOOD))
    assert len(segments.train) == 1 and len(segments.normal_eval) == 1
    assert [s.name for s in segments.anomaly] == ["touch_1", "anomaly_2"]
    # 11:15 KST == 02:15 UTC
    assert segments.train[0].start_ns == parse_dx1_timestamp("2026-10-02T02:15:00Z")
    assert segments.eval_segments()[0].kind == "normal_eval"


def test_sections_are_optional(tmp_path):
    segments = load_segments(_write(tmp_path, "train:\n  - {start: '2026-10-02T11:15:00Z', end: '2026-10-02T11:16:00Z'}\n"))
    assert segments.normal_eval == () and segments.anomaly == ()


@pytest.mark.parametrize(
    "text, message",
    [
        ("train:\n  - {start: '2026-10-02T11:15:00', end: '2026-10-02T11:16:00'}\n", "시간대"),
        ("train:\n  - {start: '2026-10-02T11:16:00Z', end: '2026-10-02T11:15:00Z'}\n", "앞이어야"),
        ("train:\n  - {start: '2026-10-02T11:15:00Z'}\n", "end가 없습니다"),
        ("foo: []\n", "알 수 없는"),
        ("- 1\n", "매핑"),
        ("train: 3\n", "목록"),
        (
            "train:\n  - {start: '2026-10-02T11:15:00Z', end: '2026-10-02T11:20:00Z'}\n"
            "normal_eval:\n  - {start: '2026-10-02T11:19:00Z', end: '2026-10-02T11:30:00Z'}\n",
            "겹칩니다",
        ),
        (
            "anomaly:\n  - {name: a, start: '2026-10-02T11:15:00Z', end: '2026-10-02T11:16:00Z'}\n"
            "  - {name: a, start: '2026-10-02T11:17:00Z', end: '2026-10-02T11:18:00Z'}\n",
            "중복",
        ),
    ],
)
def test_invalid_files_are_rejected(tmp_path, text, message):
    with pytest.raises(SegmentsError, match=message):
        load_segments(_write(tmp_path, text))


def test_missing_file_is_a_segments_error(tmp_path):
    with pytest.raises(SegmentsError, match="읽을 수 없습니다"):
        load_segments(tmp_path / "nope.yaml")


def test_touching_segments_are_allowed(tmp_path):
    text = (
        "train:\n  - {start: '2026-10-02T11:15:00Z', end: '2026-10-02T11:20:00Z'}\n"
        "normal_eval:\n  - {start: '2026-10-02T11:20:00Z', end: '2026-10-02T11:30:00Z'}\n"
    )
    assert load_segments(_write(tmp_path, text)).normal_eval
