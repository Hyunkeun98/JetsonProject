import threading
from datetime import datetime, timedelta, timezone

import pytest

from jetson_app.config import CalibrationConfig, EquipmentConfig
from jetson_app.pipeline import build_pipeline
from jetson_app.publisher import GroupOutput
from jetson_app.score_history import HistoryRecordingPublisher, ScoreHistory


def _out(score, alarm=False, top="t"):
    return GroupOutput(score=score, alarm=alarm, top_tag=top)


def test_capacity_drops_the_oldest_rows():
    history = ScoreHistory(3, ("a",))
    for i in range(5):
        history.record(i * 1000, {"a": _out(float(i))})
    data = history.snapshot(seconds=3600, max_points=1000)
    assert [p["g"]["a"][0] for p in data["points"]] == [2.0, 3.0, 4.0]
    assert data["now_ms"] == 4000 and data["bucket_ms"] == 0


def test_empty_history_returns_no_points():
    assert ScoreHistory(10, ("a",)).snapshot(60, 100) == {"now_ms": None, "bucket_ms": 0, "points": []}


def test_snapshot_limits_to_the_requested_seconds():
    history = ScoreHistory(1000, ("a",))
    for i in range(100):
        history.record(i * 1000, {"a": _out(float(i))})
    data = history.snapshot(seconds=20, max_points=1000)
    assert data["points"][0]["t"] == 79000  # now(99000) - 20000 이후


def test_downsampling_keeps_the_spike_top_tag_and_alarm_flag():
    history = ScoreHistory(10000, ("a", "b"))
    for i in range(2000):
        score = 1.0
        alarm = False
        top = "normal_tag"
        if i == 1234:
            score, top = 50.0, "spike_tag"
        if 1500 <= i < 1503:
            alarm = True
        history.record(i * 100, {"a": _out(score, alarm, top), "b": _out(0.5)})
    data = history.snapshot(seconds=200, max_points=100)
    assert 50 <= len(data["points"]) <= 101 and data["bucket_ms"] > 0
    spike = [p for p in data["points"] if p["g"]["a"][0] == 50.0]
    assert len(spike) == 1 and spike[0]["g"]["a"][2] == "spike_tag"
    assert any(p["g"]["a"][1] for p in data["points"])
    assert not any(p["g"]["b"][1] for p in data["points"])


def test_concurrent_recording_is_safe():
    history = ScoreHistory(100000, ("a",))

    def writer(base):
        for i in range(2000):
            history.record(base + i, {"a": _out(1.0)})

    threads = [threading.Thread(target=writer, args=(k * 10**6,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(history.snapshot(10**9, 5000)["points"]) <= 5000
    assert len(history._rows) == 8000


class _Inner:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def publish(self, *args):
        self.calls.append(args)
        if self.fail:
            raise RuntimeError("publish failed")


def _iso(second):
    moment = datetime(2026, 10, 2, 3, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=second)
    return moment.isoformat()


def test_wrapper_forwards_exactly_and_records_groups():
    inner, history = _Inner(), ScoreHistory(10, ("g1", "g2"))
    wrapper = HistoryRecordingPublisher(inner, history)
    groups = {"g1": _out(2.0, True, "x"), "g2": _out(1.0)}
    wrapper.publish(_iso(0), 2.0, True, "x", groups)
    assert inner.calls == [(_iso(0), 2.0, True, "x", groups)]
    point = history.snapshot(60, 100)["points"][0]
    assert point["g"] == {"g1": [2.0, True, "x"], "g2": [1.0, False, "t"]}
    assert point["t"] == int(datetime(2026, 10, 2, 3, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


def test_wrapper_records_a_flat_config_result_as_group_all():
    history = ScoreHistory(10, ("all",))
    HistoryRecordingPublisher(_Inner(), history).publish(_iso(0), 4.0, False, "tag", None)
    assert history.snapshot(60, 100)["points"][0]["g"] == {"all": [4.0, False, "tag"]}


def test_history_failure_never_blocks_publishing(capsys):
    inner = _Inner()
    HistoryRecordingPublisher(inner, ScoreHistory(10, ("a",))).publish("not a time", 1.0, False, "t", None)
    assert len(inner.calls) == 1
    assert "이력 기록 실패" in capsys.readouterr().out


def test_inner_publish_error_propagates_before_recording():
    history = ScoreHistory(10, ("a",))
    with pytest.raises(RuntimeError):
        HistoryRecordingPublisher(_Inner(fail=True), history).publish(_iso(0), 1.0, False, "t", None)
    assert history.snapshot(60, 100)["points"] == []


def _config(tmp_path):
    return EquipmentConfig(
        equipment_id="h",
        subscribe_topics=("t",),
        publish_topic="p",
        command_topic="c",
        tags=("a",),
        resample_interval_ms=100,
        window_size=3,
        calibration=CalibrationConfig(max_duration=timedelta(days=1), min_samples=5),
    )


def test_pipeline_history_is_off_by_default_and_wired_when_requested(tmp_path):
    config = _config(tmp_path)
    off = build_pipeline(config, tmp_path / "c1", tmp_path / "m1", lambda samples: None)
    assert off.history is None
    on = build_pipeline(config, tmp_path / "c2", tmp_path / "m2", lambda samples: None, history_seconds=60)
    assert on.history is not None and on.history.capacity == 600
    assert isinstance(on.snapshotter._result_publisher, HistoryRecordingPublisher)
    assert not isinstance(off.snapshotter._result_publisher, HistoryRecordingPublisher)
