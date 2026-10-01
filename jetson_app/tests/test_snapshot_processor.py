from datetime import datetime, timedelta

from jetson_app.buffer import SlidingWindow, Snapshot
from jetson_app.calibration import (
    CalibrationBufferWriter,
    CalibrationManager,
    CalibrationState,
    StateStore,
)
from jetson_app.inference import AnomalyResult
from jetson_app.resampler import ResampledStep
from jetson_app.snapshot_processor import SnapshotProcessor
from jetson_app.timeparse import format_epoch_ns


def _step(i, value=None, reset_window=False):
    return ResampledStep(
        epoch_ns=i * 50_000_000,
        snapshot=Snapshot(values={"a": float(i) if value is None else value}),
        reset_window=reset_window,
    )


class _FakeCalibrationManager:
    def __init__(self, state):
        self.state = state
        self.recorded = []

    def record_sample(self, snapshot, timestamp):
        self.recorded.append((snapshot, timestamp))


class _FakeEngine:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def score(self, window, actual):
        self.calls.append((list(window), actual))
        return self._result


class _FakeHolder:
    def __init__(self, engine):
        self._engine = engine

    def get(self):
        return self._engine


class _FakeDebouncer:
    def __init__(self, alarm):
        self._alarm = alarm
        self.scores = []
        self.reset_calls = 0

    def update(self, score):
        self.scores.append(score)
        return self._alarm

    def reset(self):
        self.reset_calls += 1


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, timestamp, anomaly_score, alarm, top_deviant_tag, groups=None):
        self.published.append((timestamp, anomaly_score, alarm, top_deviant_tag, groups))


def _filled_window(size, count):
    window = SlidingWindow(size)
    for i in range(count):
        window.push(Snapshot(values={"a": float(i)}))
    return window


def _processor(
    window, manager, engine=None, debouncer=None, publisher=None, debouncers=None, **kwargs
):
    if debouncers is None and debouncer is not None:
        debouncers = {"all": debouncer}
    return SnapshotProcessor(
        sliding_window=window,
        calibration_manager=manager,
        inference_engine_holder=_FakeHolder(engine) if engine is not None else None,
        debouncers=debouncers,
        result_publisher=publisher,
        **kwargs,
    )


def test_process_does_not_score_when_calibrating():
    window = _filled_window(2, 2)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    publisher = _FakePublisher()
    processor = _processor(window, manager, engine, _FakeDebouncer(True), publisher)

    processor.process(_step(2, 99.0))

    assert engine.calls == []
    assert publisher.published == []
    assert manager.recorded  # CALIBRATING이어도 캘리브레이션 기록은 계속됨


def test_process_does_not_score_when_window_not_full():
    window = _filled_window(5, 2)
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    processor = _processor(window, manager, engine, _FakeDebouncer(False), _FakePublisher())

    processor.process(_step(2, 99.0))

    assert engine.calls == []


def test_process_scores_with_pre_push_window_and_publishes_with_event_time():
    window = _filled_window(2, 2)
    pre_push_window = window.to_list()
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    debouncer = _FakeDebouncer(alarm=True)
    publisher = _FakePublisher()
    processor = _processor(window, manager, engine, debouncer, publisher)

    step = _step(2, 99.0)
    processor.process(step)

    assert len(engine.calls) == 1
    scored_window, scored_actual = engine.calls[0]
    assert scored_window == pre_push_window  # push되기 *전* 윈도우로 채점됐는지 확인
    assert scored_actual.values == {"a": 99.0}
    assert debouncer.scores == [5.0]
    timestamp, anomaly_score, alarm, top_deviant_tag, _groups = publisher.published[0]
    assert timestamp == format_epoch_ns(step.epoch_ns)  # 현재 시각이 아니라 이벤트 시각
    assert (anomaly_score, alarm, top_deviant_tag) == (5.0, True, "a")
    assert manager.recorded[0][1] == format_epoch_ns(step.epoch_ns)


def test_process_skips_publish_when_engine_returns_none():
    window = _filled_window(2, 2)
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    publisher = _FakePublisher()
    processor = _processor(window, manager, _FakeEngine(None), _FakeDebouncer(False), publisher)

    processor.process(_step(2, 99.0))

    assert publisher.published == []


def test_process_does_not_score_when_debouncer_and_publisher_missing():
    # holder만 주입되고 debouncer/result_publisher가 None이면 채점을 조용히 건너뛰어야
    # 한다. 그냥 진행하면 AttributeError가 매 스텝 발생하고 포괄 예외 처리에 삼켜져
    # 로그만 무한히 쌓인다.
    window = _filled_window(2, 2)
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    processor = _processor(window, manager, engine)

    processor.process(_step(2, 99.0))  # 예외 없이 통과해야 한다

    assert engine.calls == []
    assert manager.recorded


def test_process_without_inference_collaborators_still_records_calibration():
    window = SlidingWindow(2)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.process(_step(1, 1.0))

    assert manager.recorded
    assert len(window.to_list()) == 1


def test_process_reset_window_clears_window_and_debouncer_before_pushing():
    window = _filled_window(3, 3)
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=1.0, top_deviant_tag="a"))
    debouncer = _FakeDebouncer(False)
    processor = _processor(window, manager, engine, debouncer, _FakePublisher())

    step = _step(10, 7.0, reset_window=True)
    processor.process(step)

    assert debouncer.reset_calls == 1
    assert window.to_list() == [step.snapshot]  # 비워진 뒤 이 스텝만 들어 있다
    assert engine.calls == []  # 윈도우가 비었으니 채점하지 않는다


def test_submit_then_process_pending_handles_steps_in_order():
    window = SlidingWindow(10)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.submit([_step(1), _step(2), _step(3)])
    processor.process_pending()

    assert [s.values["a"] for s in window.to_list()] == [1.0, 2.0, 3.0]
    assert len(manager.recorded) == 3


def test_submit_drops_oldest_when_queue_is_full():
    window = SlidingWindow(10)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager, max_queue_size=2)

    processor.submit([_step(1), _step(2), _step(3)])
    processor.process_pending()

    assert [s.values["a"] for s in window.to_list()] == [2.0, 3.0]
    assert processor.dropped == 1


def test_stop_flushes_steps_submitted_before_stop_and_stops_thread():
    window = SlidingWindow(10)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.start()
    processor.submit([_step(1), _step(2)])
    processor.stop()

    assert len(manager.recorded) == 2
    assert processor._thread.is_alive() is False


def test_stop_without_start_still_flushes_queue():
    window = SlidingWindow(10)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.submit([_step(1)])
    processor.stop()

    assert len(manager.recorded) == 1


def test_restart_after_stop_processes_again():
    window = SlidingWindow(10)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.start()
    processor.stop()
    processor.start()
    processor.submit([_step(1)])
    processor.stop()

    assert len(manager.recorded) == 1


def test_worker_survives_exception_in_processing(capsys):
    window = SlidingWindow(10)

    class _ExplodingManager(_FakeCalibrationManager):
        def record_sample(self, snapshot, timestamp):
            if snapshot.values["a"] == 1.0:
                raise RuntimeError("boom")
            super().record_sample(snapshot, timestamp)

    manager = _ExplodingManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.submit([_step(1), _step(2)])
    processor.process_pending()

    assert len(manager.recorded) == 1
    assert "boom" in capsys.readouterr().out


def test_heartbeat_line_printed_every_100_steps(capsys):
    window = SlidingWindow(10)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.submit([_step(i) for i in range(100)])
    processor.process_pending()

    assert "[snapshotter] 100번째 스냅샷 처리" in capsys.readouterr().out


def test_calibration_timestamps_are_parseable_by_prune(tmp_path):
    # 캘리브레이션 버퍼의 prune_older_than이 datetime.fromisoformat으로 읽는 값이므로,
    # 이벤트 시각 문자열이 Python 3.8에서 파싱 가능해야 한다.
    writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")
    manager = CalibrationManager(
        buffer_writer=writer,
        min_samples=1,
        max_duration=timedelta(days=7),
        train_fn=lambda samples: None,
        state_store=StateStore(tmp_path / "state"),
    )
    processor = _processor(SlidingWindow(3), manager)

    processor.process(_step(1, 5.0))

    sample = writer.read_all()[0]
    assert datetime.fromisoformat(sample.timestamp).tzinfo is not None


# ---- 그룹별 알람 ----

from jetson_app.debounce import Debouncer
from jetson_app.inference import GroupResult


def _grouped_result(a_score, b_score):
    return AnomalyResult(
        anomaly_score=max(a_score, b_score),
        top_deviant_tag="tag_a" if a_score >= b_score else "tag_b",
        group_results={
            "A": GroupResult(score=a_score, top_tag="tag_a"),
            "B": GroupResult(score=b_score, top_tag="tag_b"),
        },
    )


def test_group_alarms_are_independent_and_overall_alarm_is_their_or():
    window = _filled_window(2, 2)
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(_grouped_result(a_score=5.0, b_score=0.1))
    publisher = _FakePublisher()
    debouncers = {
        "A": Debouncer(threshold=3.0, confirm_ticks=2),
        "B": Debouncer(threshold=3.0, confirm_ticks=2),
    }
    processor = _processor(window, manager, engine, publisher=publisher, debouncers=debouncers)

    processor.process(_step(2))
    first_alarm = publisher.published[-1][2]
    processor.process(_step(3))
    timestamp, anomaly_score, alarm, top_deviant_tag, groups = publisher.published[-1]

    assert first_alarm is False  # 연속 2번째 스텝부터 확정
    assert alarm is True
    assert (anomaly_score, top_deviant_tag) == (5.0, "tag_a")
    assert groups["A"].alarm is True and groups["A"].score == 5.0 and groups["A"].top_tag == "tag_a"
    assert groups["B"].alarm is False and groups["B"].score == 0.1


def test_overall_alarm_is_false_when_no_group_alarm_is_confirmed():
    window = _filled_window(2, 2)
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(_grouped_result(a_score=1.0, b_score=0.5))
    publisher = _FakePublisher()
    debouncers = {"A": Debouncer(confirm_ticks=1), "B": Debouncer(confirm_ticks=1)}
    processor = _processor(window, manager, engine, publisher=publisher, debouncers=debouncers)

    processor.process(_step(2))

    assert publisher.published[-1][2] is False


def test_one_groups_over_threshold_streak_does_not_leak_into_another_group():
    window = _filled_window(2, 2)
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    results = iter(
        [
            _grouped_result(a_score=5.0, b_score=0.1),
            _grouped_result(a_score=0.1, b_score=5.0),
        ]
    )

    class _SequenceEngine:
        def score(self, window, actual):
            return next(results)

    publisher = _FakePublisher()
    debouncers = {
        "A": Debouncer(threshold=3.0, confirm_ticks=2),
        "B": Debouncer(threshold=3.0, confirm_ticks=2),
    }
    processor = SnapshotProcessor(
        sliding_window=window,
        calibration_manager=manager,
        inference_engine_holder=_FakeHolder(_SequenceEngine()),
        debouncers=debouncers,
        result_publisher=publisher,
    )

    processor.process(_step(2))
    processor.process(_step(3))

    groups = publisher.published[-1][4]
    assert groups["A"].alarm is False  # A: 초과 후 정상으로 돌아와 연속이 끊김
    assert groups["B"].alarm is False  # B: 마지막 스텝에서 처음 초과했을 뿐


def test_reset_window_resets_every_group_debouncer():
    window = _filled_window(3, 3)
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    first, second = _FakeDebouncer(False), _FakeDebouncer(False)
    processor = _processor(
        window,
        manager,
        _FakeEngine(_grouped_result(1.0, 1.0)),
        publisher=_FakePublisher(),
        debouncers={"A": first, "B": second},
    )

    processor.process(_step(10, 7.0, reset_window=True))

    assert (first.reset_calls, second.reset_calls) == (1, 1)
