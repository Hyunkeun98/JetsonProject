from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .buffer import SlidingWindow, TagBuffer
from .calibration import (
    CalibrationBufferWriter,
    CalibrationManager,
    CalibrationSample,
    CalibrationState,
    StateStore,
    TrainFn,
)
from .command_subscriber import CommandSubscriber
from .config import EquipmentConfig
from .debounce import Debouncer
from .inference import ActiveModelHolder, InferenceEngine
from .mqtt_subscriber import MqttRecordSubscriber, Record
from .publisher import ResultPublisher
from .scheduler import PeriodicSnapshotter
from .training import ModelArtifact, load_artifact, model_artifact_path, state_marker_path


@dataclass(frozen=True)
class Pipeline:
    config: EquipmentConfig
    tag_buffer: TagBuffer
    sliding_window: SlidingWindow
    calibration_manager: CalibrationManager
    inference_engine_holder: ActiveModelHolder
    snapshotter: PeriodicSnapshotter
    mqtt_subscriber: MqttRecordSubscriber
    command_subscriber: CommandSubscriber


def build_pipeline(
    config: EquipmentConfig,
    calibration_dir: str | Path,
    model_dir: str | Path,
    train_fn: TrainFn,
) -> Pipeline:
    tag_buffer = TagBuffer(config.tags)
    sliding_window = SlidingWindow(config.window_size)

    def on_record(record: Record) -> None:
        tag_buffer.update(record.values)

    mqtt_subscriber = MqttRecordSubscriber(config, on_record=on_record)
    result_publisher = ResultPublisher(
        client=mqtt_subscriber.client, publish_topic=config.publish_topic
    )

    buffer_path = Path(calibration_dir) / f"{config.equipment_id}.jsonl"
    # 잘못된 --calibration-dir이 백그라운드 스레드가 아니라 기동 시점에 드러나도록
    # 디렉터리를 미리 만든다 (CLI의 OSError 처리에 걸린다).
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    buffer_writer = CalibrationBufferWriter(buffer_path)

    model_path = model_artifact_path(model_dir, config.equipment_id)
    state_path = state_marker_path(model_dir, config.equipment_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_store = StateStore(state_path)

    inference_engine_holder = ActiveModelHolder()
    debouncer = Debouncer()

    def _check_artifact_matches_config(artifact: ModelArtifact) -> None:
        """설정 YAML의 window_size/tags가 학습 이후 바뀌면 모델은 정상적으로 로드되지만
        InferenceEngine.score()가 매 틱 None을 반환해 아무 것도 발행하지 않는다
        (겉보기 상태는 정상 MONITORING). 손상된 모델 파일과 동일하게 취급한다."""
        if artifact.window_size != config.window_size or set(artifact.tags) != set(config.tags):
            raise ValueError(
                f"model artifact incompatible with current config: "
                f"window_size {artifact.window_size} vs {config.window_size}, "
                f"tags {artifact.tags} vs {config.tags}"
            )

    def wrapped_train_fn(samples: list[CalibrationSample]) -> None:
        train_fn(samples)
        # 학습이 방금 성공적으로 저장한 모델을 즉시 메모리에 올려, 다음 틱부터
        # 바로 채점을 시작할 수 있게 한다 (재시작을 기다릴 필요 없음).
        artifact = load_artifact(model_path)
        _check_artifact_matches_config(artifact)
        inference_engine_holder.set(InferenceEngine(artifact))
        # 새 모델은 오차 통계가 완전히 다르므로, 이전 모델 점수로 쌓인 연속 초과
        # 카운터를 물려받아 첫 틱부터 알람이 확정되는 일이 없도록 리셋한다.
        debouncer.reset()

    calibration_manager = CalibrationManager(
        buffer_writer=buffer_writer,
        min_samples=config.calibration.min_samples,
        max_duration=config.calibration.max_duration,
        train_fn=wrapped_train_fn,
        state_store=state_store,
    )

    if calibration_manager.state == CalibrationState.MONITORING:
        try:
            artifact = load_artifact(model_path)
            _check_artifact_matches_config(artifact)
            inference_engine_holder.set(InferenceEngine(artifact))
            print(f"[build_pipeline] 저장된 모델을 불러와 MONITORING으로 재개: {model_path}")
        except Exception as e:
            # 모델 파일 손상/누락, 또는 config와 불일치 — MONITORING 진입을 막고
            # CALIBRATING으로 폴백해 사람이 재학습을 판단하도록 한다
            # (상위 문서 5절 에러 처리 원칙).
            print(f"[build_pipeline] 모델 로드 실패, CALIBRATING으로 폴백: {e}")
            calibration_manager.handle_recalibrate_command()

    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=config.resample_interval_ms,
        inference_engine_holder=inference_engine_holder,
        debouncer=debouncer,
        result_publisher=result_publisher,
    )

    command_subscriber = CommandSubscriber(config.command_topic, calibration_manager)
    command_subscriber.attach(mqtt_subscriber.client)

    return Pipeline(
        config=config,
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        inference_engine_holder=inference_engine_holder,
        snapshotter=snapshotter,
        mqtt_subscriber=mqtt_subscriber,
        command_subscriber=command_subscriber,
    )
