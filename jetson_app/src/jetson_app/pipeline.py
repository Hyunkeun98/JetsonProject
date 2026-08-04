from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .buffer import SlidingWindow, TagBuffer
from .calibration import CalibrationBufferWriter, CalibrationManager, TrainFn
from .command_subscriber import CommandSubscriber
from .config import EquipmentConfig
from .mqtt_subscriber import MqttRecordSubscriber, Record
from .scheduler import PeriodicSnapshotter


@dataclass(frozen=True)
class Pipeline:
    config: EquipmentConfig
    tag_buffer: TagBuffer
    sliding_window: SlidingWindow
    calibration_manager: CalibrationManager
    snapshotter: PeriodicSnapshotter
    mqtt_subscriber: MqttRecordSubscriber
    command_subscriber: CommandSubscriber


def build_pipeline(
    config: EquipmentConfig,
    calibration_dir: str | Path,
    train_fn: TrainFn,
) -> Pipeline:
    tag_buffer = TagBuffer(config.tags)
    sliding_window = SlidingWindow(config.window_size)

    buffer_path = Path(calibration_dir) / f"{config.equipment_id}.jsonl"
    buffer_writer = CalibrationBufferWriter(buffer_path)
    calibration_manager = CalibrationManager(
        buffer_writer=buffer_writer,
        min_samples=config.calibration.min_samples,
        max_duration=config.calibration.max_duration,
        train_fn=train_fn,
    )

    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=config.resample_interval_ms,
    )

    def on_record(record: Record) -> None:
        tag_buffer.update(record.values)

    mqtt_subscriber = MqttRecordSubscriber(config, on_record=on_record)
    command_subscriber = CommandSubscriber(config.command_topic, calibration_manager)
    command_subscriber.attach(mqtt_subscriber.client)

    return Pipeline(
        config=config,
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        snapshotter=snapshotter,
        mqtt_subscriber=mqtt_subscriber,
        command_subscriber=command_subscriber,
    )
