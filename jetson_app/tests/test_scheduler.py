import time
from datetime import timedelta

from jetson_app.buffer import SlidingWindow, TagBuffer
from jetson_app.calibration import CalibrationBufferWriter, CalibrationManager, StateStore
from jetson_app.scheduler import PeriodicSnapshotter

from jetson_app.buffer import Snapshot
from jetson_app.calibration import CalibrationState
from jetson_app.inference import AnomalyResult


def _make_calibration_manager(tmp_path):
    writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")
    return CalibrationManager(
        buffer_writer=writer,
        min_samples=1,
        max_duration=timedelta(days=7),
        train_fn=lambda samples: None,
        state_store=StateStore(tmp_path / "state"),
    ), writer


def test_periodic_snapshotter_pushes_snapshots_to_window(tmp_path):
    tag_buffer = TagBuffer(tags=("a",))
    tag_buffer.update({"a": 1})
    window = SlidingWindow(window_size=100)
    calibration_manager, _ = _make_calibration_manager(tmp_path)

    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=window,
        calibration_manager=calibration_manager,
        interval_ms=10,
    )
    snapshotter.start()
    time.sleep(0.05)
    snapshotter.stop()

    assert len(window.to_list()) >= 2


def test_periodic_snapshotter_records_samples_into_calibration_buffer(tmp_path):
    tag_buffer = TagBuffer(tags=("a",))
    tag_buffer.update({"a": 1})
    window = SlidingWindow(window_size=100)
    calibration_manager, writer = _make_calibration_manager(tmp_path)

    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=window,
        calibration_manager=calibration_manager,
        interval_ms=10,
    )
    snapshotter.start()
    time.sleep(0.05)
    snapshotter.stop()

    assert writer.count() >= 2


def test_periodic_snapshotter_skips_completely_empty_snapshot(tmp_path):
    tag_buffer = TagBuffer(tags=("a",))  # update() 없이 -> 값이 전부 None
    window = SlidingWindow(window_size=100)
    calibration_manager, writer = _make_calibration_manager(tmp_path)

    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=window,
        calibration_manager=calibration_manager,
        interval_ms=10,
    )
    snapshotter.start()
    time.sleep(0.05)
    snapshotter.stop()

    assert window.to_list() == []
    assert writer.count() == 0


def test_periodic_snapshotter_stop_stops_the_background_thread(tmp_path):
    tag_buffer = TagBuffer(tags=("a",))
    tag_buffer.update({"a": 1})
    window = SlidingWindow(window_size=100)
    calibration_manager, _ = _make_calibration_manager(tmp_path)

    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=window,
        calibration_manager=calibration_manager,
        interval_ms=10,
    )
    snapshotter.start()
    time.sleep(0.02)
    snapshotter.stop()

    assert snapshotter._thread.is_alive() is False


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

    def update(self, score):
        self.scores.append(score)
        return self._alarm


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, timestamp, anomaly_score, alarm, top_deviant_tag):
        self.published.append((timestamp, anomaly_score, alarm, top_deviant_tag))


def _fill_window(tag_buffer, sliding_window, n):
    for i in range(n):
        tag_buffer.update({"a": float(i)})
        sliding_window.push(tag_buffer.snapshot())


def test_tick_does_not_score_when_calibrating():
    tag_buffer = TagBuffer(("a",))
    sliding_window = SlidingWindow(2)
    _fill_window(tag_buffer, sliding_window, 2)
    calibration_manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    holder = _FakeHolder(engine)
    publisher = _FakePublisher()
    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=50,
        inference_engine_holder=holder,
        debouncer=_FakeDebouncer(alarm=True),
        result_publisher=publisher,
    )
    tag_buffer.update({"a": 99.0})
    snapshotter._tick()
    assert engine.calls == []
    assert publisher.published == []
    assert calibration_manager.recorded  # CALIBRATING이어도 캘리브레이션 기록은 계속됨


def test_tick_does_not_score_when_window_not_full():
    tag_buffer = TagBuffer(("a",))
    sliding_window = SlidingWindow(5)
    _fill_window(tag_buffer, sliding_window, 2)  # 용량 5인데 2개뿐
    calibration_manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    holder = _FakeHolder(engine)
    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=50,
        inference_engine_holder=holder,
        debouncer=_FakeDebouncer(alarm=False),
        result_publisher=_FakePublisher(),
    )
    tag_buffer.update({"a": 99.0})
    snapshotter._tick()
    assert engine.calls == []


def test_tick_scores_with_pre_push_window_when_monitoring_and_full():
    tag_buffer = TagBuffer(("a",))
    sliding_window = SlidingWindow(2)
    _fill_window(tag_buffer, sliding_window, 2)
    pre_push_window = sliding_window.to_list()

    calibration_manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    holder = _FakeHolder(engine)
    debouncer = _FakeDebouncer(alarm=True)
    publisher = _FakePublisher()
    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=50,
        inference_engine_holder=holder,
        debouncer=debouncer,
        result_publisher=publisher,
    )
    tag_buffer.update({"a": 99.0})
    snapshotter._tick()

    assert len(engine.calls) == 1
    scored_window, scored_actual = engine.calls[0]
    assert scored_window == pre_push_window  # push되기 *전* 윈도우로 채점됐는지 확인
    assert scored_actual.values == {"a": 99.0}
    assert debouncer.scores == [5.0]
    assert len(publisher.published) == 1
    _, anomaly_score, alarm, top_deviant_tag = publisher.published[0]
    assert anomaly_score == 5.0
    assert alarm is True
    assert top_deviant_tag == "a"


def test_tick_skips_publish_when_engine_returns_none():
    tag_buffer = TagBuffer(("a",))
    sliding_window = SlidingWindow(2)
    _fill_window(tag_buffer, sliding_window, 2)
    calibration_manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(None)
    holder = _FakeHolder(engine)
    publisher = _FakePublisher()
    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=50,
        inference_engine_holder=holder,
        debouncer=_FakeDebouncer(alarm=False),
        result_publisher=publisher,
    )
    tag_buffer.update({"a": 99.0})
    snapshotter._tick()
    assert publisher.published == []


def test_tick_without_inference_collaborators_still_records_calibration():
    tag_buffer = TagBuffer(("a",))
    sliding_window = SlidingWindow(2)
    calibration_manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=50,
    )
    tag_buffer.update({"a": 1.0})
    snapshotter._tick()
    assert calibration_manager.recorded
