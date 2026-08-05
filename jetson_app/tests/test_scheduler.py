import time
from datetime import timedelta

from jetson_app.buffer import SlidingWindow, TagBuffer
from jetson_app.calibration import CalibrationBufferWriter, CalibrationManager, StateStore
from jetson_app.scheduler import PeriodicSnapshotter


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
