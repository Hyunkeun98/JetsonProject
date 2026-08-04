from datetime import datetime, timedelta, timezone

import pytest

from jetson_app.buffer import Snapshot
from jetson_app.calibration import (
    CalibrationBufferWriter,
    CalibrationError,
    CalibrationManager,
    CalibrationState,
)


def test_calibration_buffer_writer_append_and_read_all(tmp_path):
    writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")

    writer.append(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")
    writer.append(Snapshot(values={"a": 2}), "2026-08-04T00:00:01+00:00")

    samples = writer.read_all()
    assert len(samples) == 2
    assert samples[0].timestamp == "2026-08-04T00:00:00+00:00"
    assert samples[0].values == {"a": 1}
    assert samples[1].values == {"a": 2}


def test_calibration_buffer_writer_read_all_on_missing_file_returns_empty(tmp_path):
    writer = CalibrationBufferWriter(tmp_path / "does-not-exist.jsonl")

    assert writer.read_all() == []
    assert writer.count() == 0


def test_calibration_buffer_writer_clear_removes_data(tmp_path):
    writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")
    writer.append(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")

    writer.clear()

    assert writer.read_all() == []


def test_calibration_buffer_writer_prune_older_than_removes_old_samples(tmp_path):
    writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")
    writer.append(Snapshot(values={"a": 1}), "2026-08-01T00:00:00+00:00")
    writer.append(Snapshot(values={"a": 2}), "2026-08-04T00:00:00+00:00")

    writer.prune_older_than(datetime(2026, 8, 3, tzinfo=timezone.utc))

    samples = writer.read_all()
    assert len(samples) == 1
    assert samples[0].values == {"a": 2}


def _make_manager(tmp_path, min_samples=2, max_duration=timedelta(days=7), train_calls=None):
    writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")
    calls = train_calls if train_calls is not None else []

    def train_fn(samples):
        calls.append(samples)

    manager = CalibrationManager(
        buffer_writer=writer,
        min_samples=min_samples,
        max_duration=max_duration,
        train_fn=train_fn,
    )
    return manager, writer, calls


def test_calibration_manager_starts_in_calibrating_state(tmp_path):
    manager, _, _ = _make_manager(tmp_path)

    assert manager.state == CalibrationState.CALIBRATING


def test_calibration_manager_record_sample_writes_while_calibrating(tmp_path):
    manager, writer, _ = _make_manager(tmp_path)

    manager.record_sample(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")

    assert writer.count() == 1


def test_calibration_manager_train_command_below_min_samples_raises(tmp_path):
    manager, writer, calls = _make_manager(tmp_path, min_samples=5)
    manager.record_sample(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")

    with pytest.raises(CalibrationError):
        manager.handle_train_command()

    assert manager.state == CalibrationState.CALIBRATING
    assert calls == []


def test_calibration_manager_train_command_transitions_and_clears_buffer(tmp_path):
    manager, writer, calls = _make_manager(tmp_path, min_samples=2)
    manager.record_sample(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")
    manager.record_sample(Snapshot(values={"a": 2}), "2026-08-04T00:00:01+00:00")

    manager.handle_train_command()

    assert manager.state == CalibrationState.MONITORING
    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert writer.count() == 0


def test_calibration_manager_train_command_while_monitoring_raises(tmp_path):
    manager, _, _ = _make_manager(tmp_path, min_samples=1)
    manager.record_sample(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")
    manager.handle_train_command()

    with pytest.raises(CalibrationError):
        manager.handle_train_command()


def test_calibration_manager_recalibrate_clears_buffer_and_resets_state(tmp_path):
    manager, writer, _ = _make_manager(tmp_path, min_samples=1)
    manager.record_sample(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")
    manager.handle_train_command()
    assert manager.state == CalibrationState.MONITORING

    manager.handle_recalibrate_command()

    assert manager.state == CalibrationState.CALIBRATING
    assert writer.count() == 0


def test_calibration_manager_record_sample_ignored_while_monitoring(tmp_path):
    manager, writer, _ = _make_manager(tmp_path, min_samples=1)
    manager.record_sample(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")
    manager.handle_train_command()

    manager.record_sample(Snapshot(values={"a": 2}), "2026-08-04T00:00:02+00:00")

    assert writer.count() == 0
