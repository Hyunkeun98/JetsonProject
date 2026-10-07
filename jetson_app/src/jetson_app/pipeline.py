from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .buffer import SlidingWindow
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
from .resampler import EventTimeResampler
from .score_history import HistoryRecordingPublisher, ScoreHistory
from .snapshot_processor import SnapshotProcessor
from .training import ModelArtifact, load_artifact, model_artifact_path, state_marker_path


@dataclass(frozen=True)
class Pipeline:
    config: EquipmentConfig
    resampler: EventTimeResampler
    sliding_window: SlidingWindow
    calibration_manager: CalibrationManager
    inference_engine_holder: ActiveModelHolder
    snapshotter: SnapshotProcessor
    mqtt_subscriber: MqttRecordSubscriber
    command_subscriber: CommandSubscriber
    # 웹 화면용 점수 이력. build_pipeline(history_seconds=0)이면 None이다.
    history: ScoreHistory | None = None


def build_pipeline(
    config: EquipmentConfig,
    calibration_dir: str | Path,
    model_dir: str | Path,
    train_fn: TrainFn,
    history_seconds: int = 0,
) -> Pipeline:
    resampler = EventTimeResampler(
        tags=config.tags,
        interval_ms=config.resample_interval_ms,
        max_lateness_ms=config.max_lateness_ms,
        window_size=config.window_size,
    )
    sliding_window = SlidingWindow(config.window_size)

    def on_record(record: Record) -> None:
        # 리샘플러가 이 record로 새로 확정한 격자 칸들을 처리 스레드의 큐로 넘긴다.
        # MQTT 스레드는 큐에 넣기만 하고, 점수 계산은 처리 스레드가 한다.
        snapshotter.submit(resampler.add(record))

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
    group_specs = config.group_specs()
    # 그룹마다 독립된 디바운서: 한 그룹의 연속 초과가 다른 그룹의 알람 확정에 영향을 주지 않는다.
    debouncers = {
        name: Debouncer(
            threshold=config.alarm.threshold, confirm_ticks=config.alarm.confirm_steps
        )
        for name in group_specs
    }

    def _check_artifact_matches_config(artifact: ModelArtifact) -> None:
        """설정 YAML의 window_size/tags/resample_interval_ms/그룹 구성이 학습 이후 바뀌면 모델은
        정상적으로 로드되지만 InferenceEngine.score()가 매 스텝 None을 반환하거나
        (window_size/tags), 학습 때와 다른 시간 간격의 윈도우로 엉뚱한 점수를 내거나
        (resample_interval_ms), 그룹 점수/상태별 기준이 현재 설정과 어긋난다(groups).
        겉보기 상태는 정상 MONITORING이므로, 손상된 모델 파일과 동일하게 취급한다.
        격자 정보나 그룹 정보가 없는 기존 artifact도 여기서 걸러져 재학습을 유도한다."""
        if (
            artifact.window_size != config.window_size
            or set(artifact.tags) != set(config.tags)
            or artifact.resample_interval_ms != config.resample_interval_ms
            or artifact.groups != group_specs
        ):
            raise ValueError(
                f"model artifact incompatible with current config: "
                f"window_size {artifact.window_size} vs {config.window_size}, "
                f"tags {artifact.tags} vs {config.tags}, "
                f"resample_interval_ms {artifact.resample_interval_ms} vs "
                f"{config.resample_interval_ms}, "
                f"groups {artifact.groups} vs {group_specs}"
            )

    def wrapped_train_fn(samples: list[CalibrationSample]) -> None:
        train_fn(samples)
        # 학습이 방금 성공적으로 저장한 모델을 즉시 메모리에 올려, 다음 스텝부터
        # 바로 채점을 시작할 수 있게 한다 (재시작을 기다릴 필요 없음).
        artifact = load_artifact(model_path)
        _check_artifact_matches_config(artifact)
        inference_engine_holder.set(InferenceEngine(artifact))
        # 새 모델은 오차 통계가 완전히 다르므로, 이전 모델 점수로 쌓인 연속 초과
        # 카운터를 물려받아 첫 스텝부터 알람이 확정되는 일이 없도록 모든 그룹을 리셋한다.
        for debouncer in debouncers.values():
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

    history = None
    processor_publisher = result_publisher
    if history_seconds > 0:
        capacity = max(1, history_seconds * 1000 // config.resample_interval_ms)
        history = ScoreHistory(capacity, tuple(group_specs))
        processor_publisher = HistoryRecordingPublisher(result_publisher, history)

    snapshotter = SnapshotProcessor(
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        inference_engine_holder=inference_engine_holder,
        debouncers=debouncers,
        result_publisher=processor_publisher,
    )

    command_subscriber = CommandSubscriber(config.command_topic, calibration_manager)
    command_subscriber.attach(mqtt_subscriber.client)

    return Pipeline(
        config=config,
        resampler=resampler,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        inference_engine_holder=inference_engine_holder,
        snapshotter=snapshotter,
        mqtt_subscriber=mqtt_subscriber,
        command_subscriber=command_subscriber,
        history=history,
    )
