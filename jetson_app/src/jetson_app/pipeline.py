from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .buffer import SlidingWindow, TagBuffer
from .calibration import CalibrationBufferWriter, CalibrationManager, StateStore, TrainFn
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
    # 잘못된 --calibration-dir이 백그라운드 스레드가 아니라 기동 시점에 드러나도록
    # 디렉터리를 미리 만든다 (CLI의 OSError 처리에 걸린다).
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    buffer_writer = CalibrationBufferWriter(buffer_path)
    # TODO(Task 8): state 파일을 model_dir 기준으로 재배치할 예정. 지금은 임시로
    # calibration_dir 아래에 둔다.
    state_store = StateStore(Path(calibration_dir) / f"{config.equipment_id}.state")
    calibration_manager = CalibrationManager(
        buffer_writer=buffer_writer,
        min_samples=config.calibration.min_samples,
        max_duration=config.calibration.max_duration,
        train_fn=train_fn,
        state_store=state_store,
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
