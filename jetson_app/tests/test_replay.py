import json
import math
from datetime import timedelta

import pytest

from jetson_app.buffer import SlidingWindow
from jetson_app.calibration import CalibrationState, StateStore
from jetson_app.config import CalibrationConfig, EquipmentConfig, GroupConfig
from jetson_app.debounce import Debouncer
from jetson_app.inference import ActiveModelHolder, InferenceEngine
from jetson_app.replay import (
    ReplayError,
    load_matching_artifact,
    load_records,
    score_from_files,
    score_steps,
    steps_for_segment,
    train_from_files,
)
from jetson_app.segments import KIND_NORMAL_EVAL, Segment, SegmentSet
from jetson_app.snapshot_processor import SnapshotProcessor
from jetson_app.timeparse import format_epoch_ns, parse_dx1_timestamp
from jetson_app.training import state_marker_path

BASE_NS = parse_dx1_timestamp("2026-10-02T00:00:00Z")
STEP_NS = 100_000_000


def _ns(second):
    return BASE_NS + int(second * 1_000_000_000)


def _timestamp(second):
    return format_epoch_ns(_ns(second)).replace("+00:00", "+0000")


def _config(groups=False):
    kwargs = {}
    tags = ("a", "b")
    if groups:
        kwargs["groups"] = (GroupConfig(name="g", state_tag="b", tags=("a",)),)
    return EquipmentConfig(
        equipment_id="rep",
        subscribe_topics=("t/data",),
        publish_topic="p",
        command_topic="c",
        tags=tags,
        resample_interval_ms=100,
        window_size=3,
        calibration=CalibrationConfig(max_duration=timedelta(days=1), min_samples=10),
        max_lateness_ms=0,
        **kwargs,
    )


def _write_data(directory, seconds=60, splits=2):
    """0.1초 간격, a는 사인파, b는 0/1. 여러 파일과 메시지에 나눠 쓴다."""
    directory.mkdir(parents=True, exist_ok=True)
    handles = [(directory / f"DX_{i}.json").open("w", encoding="utf-8") for i in range(splits)]
    for n in range(seconds * 10):
        record = {
            "timestamp": _timestamp(n / 10),
            "a": math.sin(n / 5.0) * 10,
            "b": (n // 50) % 2,
            "ignored": 1,
        }
        handles[n % splits].write(json.dumps({"records": [record]}) + "\n")
    for handle in handles:
        handle.close()


def _segment(kind, name, start_s, end_s):
    return Segment(kind=kind, name=name, start_ns=_ns(start_s), end_ns=_ns(end_s))


def _segment_set(train=(), normal=(), anomaly=()):
    return SegmentSet(train=tuple(train), normal_eval=tuple(normal), anomaly=tuple(anomaly))


def test_load_records_filters_tags_segments_and_sorts(tmp_path):
    _write_data(tmp_path / "d", seconds=10)
    records = load_records(tmp_path / "d", ("a", "b"), (_segment("train", "t", 2, 4),))
    assert len(records) == 20
    assert [r.epoch_ns for r in records] == sorted(r.epoch_ns for r in records)
    assert all(_ns(2) <= r.epoch_ns < _ns(4) for r in records)
    assert set(records[0].values) == {"a", "b"}


def test_load_records_ignores_broken_lines_and_reports_missing_data(tmp_path):
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / "x.json").write_text(
        "not json\n\n" + json.dumps({"records": [{"timestamp": _timestamp(1), "a": 1.0}]}) + "\n",
        encoding="utf-8",
    )
    records = load_records(directory, ("a",), (_segment("train", "t", 0, 5),))
    assert len(records) == 1
    with pytest.raises(ReplayError, match="찾을 수 없습니다"):
        load_records(tmp_path / "nope", ("a",), ())
    (tmp_path / "empty").mkdir()
    with pytest.raises(ReplayError, match="파일이 없습니다"):
        load_records(tmp_path / "empty", ("a",), ())


def test_each_segment_gets_its_own_resampler(tmp_path):
    _write_data(tmp_path / "d", seconds=20)
    config = _config()
    first = _segment("train", "t1", 0, 5)
    second = _segment("train", "t2", 10, 15)
    records = load_records(tmp_path / "d", config.tags, (first, second))
    steps_first = steps_for_segment(config, records, first)
    steps_second = steps_for_segment(config, records, second)
    assert steps_first and steps_second
    assert max(s.epoch_ns for s in steps_first) < _ns(5)
    assert min(s.epoch_ns for s in steps_second) >= _ns(10)
    assert not steps_second[0].reset_window


@pytest.mark.parametrize("groups", [False, True])
def test_train_then_score_end_to_end(tmp_path, groups):
    _write_data(tmp_path / "d", seconds=60)
    config = _config(groups=groups)
    segments = _segment_set(
        train=[_segment("train", "t", 0, 40)],
        normal=[_segment(KIND_NORMAL_EVAL, "n", 40, 55)],
        anomaly=[_segment("anomaly", "x", 55, 60)],
    )
    model_path, trained_steps = train_from_files(config, tmp_path / "d", segments, tmp_path / "m")
    assert model_path.exists() and trained_steps > 300
    state = StateStore(state_marker_path(tmp_path / "m", "rep")).read()
    assert state == CalibrationState.MONITORING

    rows, group_names = score_from_files(config, tmp_path / "d", segments, tmp_path / "m")
    assert group_names == (("g",) if groups else ("all",))
    assert {r.kind for r in rows} == {"normal_eval", "anomaly"}
    assert rows[0].reset and sum(r.reset for r in rows if r.segment == "n") == 1
    first_anomaly = next(r for r in rows if r.segment == "x")
    assert first_anomaly.reset
    assert all(set(r.groups) <= set(group_names) for r in rows)


def test_train_requires_train_segments_and_data_in_range(tmp_path):
    _write_data(tmp_path / "d", seconds=10)
    config = _config()
    with pytest.raises(ReplayError, match="train 구간이 없습니다"):
        train_from_files(config, tmp_path / "d", _segment_set(), tmp_path / "m")
    far = _segment_set(train=[_segment("train", "t", 1000, 1010)])
    with pytest.raises(ReplayError, match="스텝이 없습니다"):
        train_from_files(config, tmp_path / "d", far, tmp_path / "m")


def test_score_needs_a_matching_model_and_eval_segments(tmp_path):
    _write_data(tmp_path / "d", seconds=60)
    config = _config()
    segments = _segment_set(
        train=[_segment("train", "t", 0, 40)], normal=[_segment(KIND_NORMAL_EVAL, "n", 40, 55)]
    )
    with pytest.raises(ReplayError, match="모델 파일이 없습니다"):
        score_from_files(config, tmp_path / "d", segments, tmp_path / "m")
    with pytest.raises(ReplayError, match="normal_eval 또는 anomaly"):
        score_from_files(config, tmp_path / "d", _segment_set(train=segments.train), tmp_path / "m")
    train_from_files(config, tmp_path / "d", segments, tmp_path / "m")
    from dataclasses import replace

    with pytest.raises(ReplayError, match="맞지 않습니다"):
        load_matching_artifact(replace(config, window_size=5), tmp_path / "m")


def test_score_fails_loudly_for_a_segment_without_scores(tmp_path):
    _write_data(tmp_path / "d", seconds=60)
    config = _config()
    segments = _segment_set(
        train=[_segment("train", "t", 0, 40)], normal=[_segment(KIND_NORMAL_EVAL, "n", 40, 40.2)]
    )
    train_from_files(config, tmp_path / "d", segments, tmp_path / "m")
    with pytest.raises(ReplayError, match="점수가 하나도"):
        score_from_files(config, tmp_path / "d", segments, tmp_path / "m")


class _RecordingPublisher:
    def __init__(self):
        self.published = []

    def publish(self, timestamp, score, alarm, top_tag, outputs):
        self.published.append((timestamp, score, {n: (o.score, o.top_tag) for n, o in outputs.items()}))


class _Calibration:
    state = CalibrationState.MONITORING

    def record_sample(self, snapshot, timestamp):
        pass


@pytest.mark.parametrize("groups", [False, True])
def test_replayed_scores_match_the_app_snapshot_processor(tmp_path, groups):
    """같은 스텝을 앱 처리기와 재생 루프에 넣으면 점수가 같아야 한다."""
    _write_data(tmp_path / "d", seconds=60)
    config = _config(groups=groups)
    segments = _segment_set(
        train=[_segment("train", "t", 0, 40)], normal=[_segment(KIND_NORMAL_EVAL, "n", 40, 55)]
    )
    train_from_files(config, tmp_path / "d", segments, tmp_path / "m")
    artifact = load_matching_artifact(config, tmp_path / "m")
    engine = InferenceEngine(artifact)
    segment = segments.normal_eval[0]
    records = load_records(tmp_path / "d", config.tags, (segment,))
    steps = steps_for_segment(config, records, segment)

    holder = ActiveModelHolder()
    holder.set(engine)
    names = tuple(config.group_specs())
    publisher = _RecordingPublisher()
    processor = SnapshotProcessor(
        sliding_window=SlidingWindow(config.window_size),
        calibration_manager=_Calibration(),
        inference_engine_holder=holder,
        debouncers={n: Debouncer(threshold=1e9, confirm_ticks=3) for n in names},
        result_publisher=publisher,
    )
    for step in steps:
        processor.process(step)

    rows = score_steps(engine, steps, config.window_size, segment)
    assert len(rows) == len(publisher.published) > 100
    for row, (_timestamp_text, _score, outputs) in zip(rows, publisher.published):
        assert row.groups == outputs


def test_cli_train_score_threshold_end_to_end(tmp_path, capsys):
    import json as _json

    from jetson_app.replay_cli import main as replay_main
    from jetson_app.threshold_cli import main as threshold_main

    _write_data(tmp_path / "d", seconds=60)
    (tmp_path / "cfg.yaml").write_text(
        "equipment_id: rep\n"
        "mqtt: {subscribe_topics: [t/data], publish_topic: p, command_topic: c}\n"
        "tags: [a, b]\n"
        "resample_interval_ms: 100\n"
        "max_lateness_ms: 0\n"
        "window_size: 3\n"
        "calibration: {max_duration: 1d, min_samples: 10}\n"
        "training: {epochs: 1}\n",
        encoding="utf-8",
    )
    (tmp_path / "seg.yaml").write_text(
        "train:\n  - {start: '2026-10-02T00:00:00Z', end: '2026-10-02T00:00:40Z'}\n"
        "normal_eval:\n  - {start: '2026-10-02T00:00:40Z', end: '2026-10-02T00:00:55Z'}\n"
        "anomaly:\n  - {name: x, start: '2026-10-02T00:00:55Z', end: '2026-10-02T00:01:00Z'}\n",
        encoding="utf-8",
    )
    common = ["--config", str(tmp_path / "cfg.yaml"), "--data-dir", str(tmp_path / "d"),
              "--segments", str(tmp_path / "seg.yaml"), "--model-dir", str(tmp_path / "m")]
    replay_main(["train"] + common)
    replay_main(["score"] + common + ["--out", str(tmp_path / "scores.csv")])
    threshold_main(["--scores", str(tmp_path / "scores.csv"), "--config", str(tmp_path / "cfg.yaml"),
                    "--report", str(tmp_path / "report.json")])
    out = capsys.readouterr().out
    assert "학습 완료" in out and "추천" in out
    report = _json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report[0]["confirm"] == 3 and report[0]["table"]


def test_cli_reports_errors_with_exit_code_2(tmp_path, capsys):
    from jetson_app.replay_cli import main as replay_main

    (tmp_path / "seg.yaml").write_text("train:\n  - {start: '2026-10-02T00:00:00', end: '2026-10-02T00:01:00'}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        replay_main(["train", "--config", "nope.yaml", "--data-dir", str(tmp_path), "--segments", str(tmp_path / "seg.yaml")])
    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err
