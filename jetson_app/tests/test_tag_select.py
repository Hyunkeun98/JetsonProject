import pytest

from jetson_app.buffer import Snapshot
from jetson_app.calibration import CalibrationSample
from jetson_app.config import ConfigError, load_equipment_config
from jetson_app.inference import InferenceEngine
from jetson_app.scores import ScoreRow, read_scores_csv, write_scores_csv
from jetson_app.segments import SegmentsError, load_segments
from jetson_app.tag_select import TagSelectError, format_selection, select_tags
from jetson_app.training import train_model

STEP = 100_000_000


def _rows(kind, name, z_by_tag, start=0, reset_first=True):
    """z_by_tag: 태그 -> 점수 목록(같은 길이)."""
    n = len(next(iter(z_by_tag.values())))
    return [
        ScoreRow(
            epoch_ns=start + i * STEP,
            kind=kind,
            segment=name,
            reset=reset_first and i == 0,
            groups={},
            tag_z={t: v[i] for t, v in z_by_tag.items()},
        )
        for i in range(n)
    ]


def _data():
    # pos는 정상에서도 가끔 크게 튄다(오탐 원인). vel은 이상에서만 오른다.
    normal = _rows("normal_eval", "n", {"pos": [0, 0, 20, 20, 20, 0, 0, 0], "vel": [0.1] * 8})
    a1 = _rows("anomaly", "a1", {"pos": [0, 0, 0, 0], "vel": [6, 6, 6, 6]}, start=10**12)
    a2 = _rows("anomaly", "a2", {"pos": [0, 0, 0, 0], "vel": [5, 5, 5, 5]}, start=2 * 10**12)
    return normal + a1 + a2


def test_best_combination_drops_the_tag_that_causes_normal_false_alarms():
    sel = select_tags(_data(), {"g": ("pos", "vel")}, {}, confirm=3)[0]
    assert sel.all_tags.normal_c == 20  # 이상 점수 6, 5는 정상 최대 20보다 낮다
    assert sel.best.tags == ("vel",)
    assert sel.best.detected == 2 and sel.best.gap == pytest.approx(5 - 0.1)


def test_all_tags_detect_nothing_when_one_tag_hides_the_anomaly():
    sel = select_tags(_data(), {"g": ("pos", "vel")}, {}, confirm=3)[0]
    assert sel.all_tags.detected == 0


def test_anomalies_are_assigned_to_their_group_only():
    data = _data()
    sel = select_tags(data, {"g": ("pos", "vel"), "h": ("pos", "vel")}, {"a1": ("g",), "a2": ("h",)}, 3)
    by = {s.group: s for s in sel}
    assert by["g"].anomalies == ("a1",) and by["h"].anomalies == ("a2",)


def test_group_without_assigned_anomaly_is_reported_without_a_choice():
    sel = select_tags(_data(), {"g": ("pos", "vel"), "h": ("pos", "vel")}, {"a1": ("g",), "a2": ("g",)}, 3)
    text = format_selection(sel, 3)
    assert "배정된 이상 구간이 없어" in text
    assert "score_exclude_tags" in text and "pos" in text


def test_single_tag_normal_critical_scores_work_without_anomalies():
    normal = _rows("normal_eval", "n", {"pos": [20] * 6, "vel": [0.1] * 6})
    sel = select_tags(normal, {"g": ("pos", "vel")}, {}, 3)[0]
    assert sel.single_normal_c == {"pos": 20, "vel": 0.1} and sel.anomalies == ()


def test_resets_break_the_consecutive_run():
    rows = _rows("normal_eval", "n", {"a": [9, 9, 0, 9, 9]})
    # reset이 중간에 있으면 두 번의 연속 2는 있어도 3은 없다
    rows[3] = ScoreRow(rows[3].epoch_ns, "normal_eval", "n", True, {}, rows[3].tag_z)
    sel = select_tags(rows, {"g": ("a",)}, {}, 3)[0]
    assert sel.all_tags.normal_c == 0


def test_errors_when_scores_have_no_tag_columns_or_no_normal_rows():
    plain = [ScoreRow(0, "normal_eval", "n", True, {"g": (1.0, "t")})]
    with pytest.raises(TagSelectError, match="태그별 점수"):
        select_tags(plain, {"g": ("a",)}, {}, 3)
    only_anomaly = _rows("anomaly", "a", {"a": [1, 1, 1]})
    with pytest.raises(TagSelectError, match="normal_eval"):
        select_tags(only_anomaly, {"g": ("a",)}, {}, 3)


def test_few_anomalies_warning():
    normal = _rows("normal_eval", "n", {"a": [0.1] * 5})
    a = _rows("anomaly", "x", {"a": [5] * 5}, start=10**12)
    text = format_selection(select_tags(normal + a, {"g": ("a",)}, {}, 3), 3)
    assert "과적합" in text


def test_scores_csv_round_trips_tag_scores(tmp_path):
    rows = _rows("normal_eval", "n", {"NX5:X_Pos": [1.5, 2.5], "NX5:X_Vel": [0.5, 0.25]})
    rows = [ScoreRow(r.epoch_ns, r.kind, r.segment, r.reset, {"g": (r.tag_z["NX5:X_Pos"], "NX5:X_Pos")}, r.tag_z) for r in rows]
    path = tmp_path / "s.csv"
    write_scores_csv(path, rows, ("g",))
    back, names = read_scores_csv(path)
    assert names == ("g",) and back[1].tag_z == {"NX5:X_Pos": 2.5, "NX5:X_Vel": 0.25}
    assert back[0].groups == {"g": (1.5, "NX5:X_Pos")}


def test_old_scores_csv_without_tag_columns_still_loads(tmp_path):
    rows = [ScoreRow(0, "normal_eval", "n", True, {"g": (1.0, "t")})]
    path = tmp_path / "old.csv"
    write_scores_csv(path, rows, ("g",))
    back, _ = read_scores_csv(path)
    assert back[0].tag_z == {}


def test_anomaly_segment_groups_in_segments_file(tmp_path):
    path = tmp_path / "seg.yaml"
    path.write_text(
        "anomaly:\n"
        '  - {name: x, start: "2026-10-08T10:00:00+09:00", end: "2026-10-08T10:00:10+09:00", groups: [axis_x]}\n',
        encoding="utf-8",
    )
    assert load_segments(path).anomaly[0].groups == ("axis_x",)
    path.write_text(
        "normal_eval:\n"
        '  - {name: n, start: "2026-10-08T10:00:00+09:00", end: "2026-10-08T10:00:10+09:00", groups: [axis_x]}\n',
        encoding="utf-8",
    )
    with pytest.raises(SegmentsError, match="anomaly에서만"):
        load_segments(path)


def _samples(n):
    return [CalibrationSample(timestamp=f"t{i}", values={"a": float(i % 2), "b": float(i)}) for i in range(n)]


def test_excluded_tag_is_left_out_of_the_group_score_but_kept_in_tag_scores():
    artifact = train_model(_samples(30), tags=("a", "b"), window_size=3, epochs=2, hidden_size=4, num_layers=1,
                           groups={"g": (None, ("a", "b"))})
    window = [Snapshot(values={"a": 1.0, "b": float(i)}) for i in range(3)]
    actual = Snapshot(values={"a": 0.0, "b": 50.0})
    full = InferenceEngine(artifact).score(window, actual)
    cut = InferenceEngine(artifact, exclude_tags=("b",)).score(window, actual)
    assert set(full.tag_z) == {"a", "b"} and set(cut.tag_z) == {"a", "b"}
    assert cut.group_results["g"].top_tag == "a"
    assert cut.group_results["g"].score == pytest.approx(full.tag_z["a"])
    assert full.group_results["g"].score == pytest.approx(max(full.tag_z.values()))


_BASE = """
equipment_id: e
mqtt: {subscribe_topics: [t], publish_topic: p, command_topic: c}
groups:
  g1: {tags: [a, b]}
  g2: {tags: [c]}
resample_interval_ms: 100
window_size: 5
calibration: {max_duration: 1d, min_samples: 10}
"""


def _cfg(tmp_path, extra):
    path = tmp_path / "c.yaml"
    path.write_text(_BASE + extra, encoding="utf-8")
    return load_equipment_config(path)


def test_score_exclude_tags_config(tmp_path):
    cfg = _cfg(tmp_path, "score_exclude_tags: [a]\n")
    assert cfg.score_exclude_tags == ("a",) and cfg.tags == ("a", "b", "c")
    assert cfg.group_specs() == _cfg(tmp_path, "").group_specs()  # 모델 호환성에는 영향이 없다


@pytest.mark.parametrize(
    "extra, message",
    [
        ("score_exclude_tags: [zz]\n", "not a scored tag"),
        ("score_exclude_tags: [a, b]\n", "no scored tag left"),
        ("score_exclude_tags: [c]\n", "no scored tag left"),
        ("score_exclude_tags: [a, a]\n", "more than once"),
        ("score_exclude_tags: []\n", "non-empty list"),
    ],
)
def test_score_exclude_tags_validation(tmp_path, extra, message):
    with pytest.raises(ConfigError, match=message):
        _cfg(tmp_path, extra)


def test_no_exclusion_is_proposed_when_no_combination_detects_anything():
    normal = _rows("normal_eval", "n", {"a": [9.0] * 6, "b": [9.0] * 6})
    anomaly = _rows("anomaly", "x", {"a": [1.0] * 5, "b": [1.0] * 5}, start=10**12)
    text = format_selection(select_tags(normal + anomaly, {"g": ("a", "b")}, {}, 3), 3)
    assert "제안 없음" in text and "score_exclude_tags" not in text
