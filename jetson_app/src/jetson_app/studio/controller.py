"""화면이 쓰는 앱 제어 계층: 단계(설정→수집→학습→감시)를 한곳에서 관리한다.

새로 계산하는 로직은 없다. 파이프라인 생성은 build_pipeline, 학습은 CalibrationManager와
make_deviation_train_fn, 태그 추천은 deviation_suggest를 그대로 부른다.
"""
from __future__ import annotations

import threading
import time
from enum import Enum
from pathlib import Path

from ..calibration import CalibrationState
from ..config import ConfigError, load_equipment_config
from ..deviation import deviation_artifact_path, load_deviation_artifact, make_deviation_train_fn
from ..pipeline import build_pipeline
from ..score_history import DEFAULT_HISTORY_SECONDS
from . import wizard
from .errors import StudioError

NO_DATA_WARN_SECONDS = 30.0
_EMPTY_SCORES = {"now_ms": None, "bucket_ms": 0, "points": []}


class Stage(str, Enum):
    SETUP = "SETUP"            # 저장된 설정이 없다
    READY = "READY"            # 설정은 있고 아직 브로커에 연결하지 않았다
    COLLECTING = "COLLECTING"  # 연결됨, 정상 데이터를 모으는 중
    TRAINING = "TRAINING"      # 학습 스레드가 도는 중
    MONITORING = "MONITORING"  # 학습 결과로 감시 중


class MqttIO:
    """파이프라인의 MQTT 클라이언트를 백그라운드 스레드로 돌리고 멈춘다(테스트에서는 가짜로 바꾼다)."""

    def connect(self, pipeline, host: str, port: int) -> None:
        pipeline.mqtt_subscriber.connect(host, port)
        pipeline.mqtt_subscriber.client.loop_start()

    def disconnect(self, pipeline) -> None:
        client = pipeline.mqtt_subscriber.client
        client.loop_stop()
        client.disconnect()

    def is_connected(self, pipeline) -> bool:
        return bool(pipeline.mqtt_subscriber.client.is_connected())


class StudioController:
    def __init__(
        self,
        workspace,
        io=None,
        pipeline_factory=build_pipeline,
        clock=time.monotonic,
        run_worker: bool = True,
    ) -> None:
        self._workspace = Path(workspace)
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._calibration_dir = self._workspace / "calibration"
        self._model_dir = self._workspace / "models"
        self._io = io or MqttIO()
        self._factory = pipeline_factory
        self._clock = clock
        self._run_worker = run_worker
        self._lock = threading.RLock()
        self._pipeline = None
        self._config = None
        self._training = False
        self._train_thread = None
        self._error = None
        self._last_count = 0
        self._last_rx = None
        self._started_at = 0.0
        self._load_saved_config()

    # --- 설정 ---------------------------------------------------------------------------

    def _load_saved_config(self) -> None:
        settings = wizard.load_settings(self._workspace)
        if settings is None:
            return
        try:
            self._config = load_equipment_config(self._workspace / "configs" / settings["config"])
        except ConfigError as e:
            self._error = f"저장된 설정을 읽지 못했습니다: {e}"

    def analyze(self, data_dir: str) -> dict:
        return wizard.analyze(data_dir)

    def form(self) -> dict:
        return wizard.saved_form(self._workspace)

    def save_config(self, form: dict) -> None:
        with self._lock:
            if self._pipeline is not None:
                raise StudioError("실행 중에는 설정을 바꿀 수 없습니다. 먼저 중지하세요.")
            self._config = wizard.save_config(self._workspace, form)
            self._error = None

    # --- 실행 ---------------------------------------------------------------------------

    @property
    def pipeline(self):
        return self._pipeline

    def start(self) -> None:
        with self._lock:
            if self._config is None:
                raise StudioError("먼저 설정을 저장하세요.")
            if self._pipeline is not None:
                raise StudioError("이미 실행 중입니다.")
            config = self._config
            if config.tags or config.deviation is None:
                raise StudioError("이 화면은 위치 신호의 기준값 이탈(deviation) 설정만 지원합니다.")
            settings = wizard.load_settings(self._workspace)
            host, port = settings["host"], settings["port"]
            train_fn = make_deviation_train_fn(
                config, deviation_artifact_path(self._model_dir, config.equipment_id)
            )
            pipeline = self._factory(
                config=config,
                calibration_dir=self._calibration_dir,
                model_dir=self._model_dir,
                train_fn=train_fn,
                history_seconds=DEFAULT_HISTORY_SECONDS,
            )
            try:
                self._io.connect(pipeline, host, port)
            except OSError as e:
                raise StudioError(
                    f"브로커({host}:{port})에 연결할 수 없습니다. Mosquitto가 실행 중인지 확인하세요. ({e})"
                ) from e
            if self._run_worker:
                pipeline.snapshotter.start()
            self._pipeline = pipeline
            self._error = None
            self._last_count = pipeline.calibration_manager.sample_count
            self._last_rx = None
            self._started_at = self._clock()

    def stop(self) -> None:
        with self._lock:
            if self._pipeline is None:
                return
            if self._training:
                raise StudioError("학습이 끝난 뒤에 중지할 수 있습니다.")
            pipeline = self._pipeline
            self._pipeline = None
        if self._run_worker:
            pipeline.snapshotter.stop()
        self._io.disconnect(pipeline)

    def feed_record(self, record) -> None:
        """MQTT 스레드가 하는 일(리샘플 후 처리 큐에 넣기)을 그대로 한다. 시험과 재생용."""
        pipeline = self._pipeline
        if pipeline is None:
            raise StudioError("실행 중이 아닙니다.")
        pipeline.snapshotter.submit(pipeline.resampler.add(record))

    def pump(self) -> None:
        """처리 큐를 동기로 비운다(run_worker=False일 때 시험이 쓴다)."""
        if self._pipeline is not None:
            self._pipeline.snapshotter.process_pending()

    # --- 학습 ---------------------------------------------------------------------------

    def train(self) -> None:
        with self._lock:
            pipeline = self._pipeline
            if pipeline is None:
                raise StudioError("먼저 시작해서 정상 데이터를 모으세요.")
            if self._training:
                raise StudioError("이미 학습 중입니다.")
            manager = pipeline.calibration_manager
            if manager.state != CalibrationState.CALIBRATING:
                raise StudioError("이미 학습이 끝났습니다. 다시 학습하려면 '처음부터 다시 수집'을 누르세요.")
            if manager.sample_count < manager.min_samples:
                raise StudioError(
                    f"정상 데이터가 부족합니다: {manager.sample_count} / {manager.min_samples}"
                )
            self._training = True
            self._error = None

        def _run() -> None:
            try:
                manager.handle_train_command()
            except Exception as e:
                with self._lock:
                    self._error = f"학습에 실패했습니다: {e}"
            finally:
                with self._lock:
                    self._training = False

        self._train_thread = threading.Thread(target=_run, daemon=True)
        self._train_thread.start()

    def join_training(self, timeout=None) -> None:
        thread = self._train_thread
        if thread is not None:
            thread.join(timeout)

    def recalibrate(self) -> None:
        with self._lock:
            if self._pipeline is None:
                raise StudioError("실행 중이 아닙니다.")
            if self._training:
                raise StudioError("학습이 끝난 뒤에 다시 수집할 수 있습니다.")
            self._pipeline.calibration_manager.handle_recalibrate_command()
            self._error = None
            self._last_count = 0
            self._last_rx = None
            self._started_at = self._clock()

    # --- 화면용 조회 ----------------------------------------------------------------------

    def _stage_locked(self) -> Stage:
        if self._config is None:
            return Stage.SETUP
        if self._pipeline is None:
            return Stage.READY
        if self._training:
            return Stage.TRAINING
        if self._pipeline.calibration_manager.state == CalibrationState.MONITORING:
            return Stage.MONITORING
        return Stage.COLLECTING

    def _signals(self, config) -> list:
        path = deviation_artifact_path(self._model_dir, config.equipment_id)
        if not path.exists():
            return []
        try:
            artifact = load_deviation_artifact(path)
        except Exception:
            return []
        return [
            {"name": n, "tag": s.tag, "sigma": s.sigma, "limit": s.limit, "move_range": s.move_range}
            for n, s in artifact.signals.items()
        ]

    def state(self) -> dict:
        with self._lock:
            config, pipeline = self._config, self._pipeline
            stage = self._stage_locked()
            info = {
                "stage": stage.value,
                "equipment_id": config.equipment_id if config else None,
                "error": self._error,
                "connected": False,
                "collected": 0,
                "required": config.calibration.min_samples if config else 0,
                "no_data": False,
                "signals": [],
                "threshold": config.alarm.threshold if config else None,
            }
            if pipeline is not None:
                info["connected"] = bool(self._io.is_connected(pipeline))
                count = pipeline.calibration_manager.sample_count
                now = self._clock()
                if count != self._last_count:
                    self._last_count = count
                    self._last_rx = now
                info["collected"] = count
                if stage == Stage.COLLECTING:
                    since = self._last_rx if self._last_rx is not None else self._started_at
                    info["no_data"] = (now - since) > NO_DATA_WARN_SECONDS
            if stage == Stage.MONITORING and config is not None:
                info["signals"] = self._signals(config)
            return info

    def meta(self) -> dict:
        """기존 점수 그래프 화면(dashboard.html)이 읽는 /api/meta와 같은 모양."""
        with self._lock:
            config, pipeline = self._config, self._pipeline
            if config is None or pipeline is None or pipeline.history is None:
                return {
                    "equipment_id": config.equipment_id if config else "",
                    "groups": [],
                    "threshold": config.alarm.threshold if config else 0,
                    "confirm_steps": config.alarm.confirm_steps if config else 1,
                    "interval_ms": config.resample_interval_ms if config else 100,
                    "history_seconds": DEFAULT_HISTORY_SECONDS,
                    "state": self._stage_locked().value,
                }
            return {
                "equipment_id": config.equipment_id,
                "groups": list(pipeline.history.group_names),
                "threshold": config.alarm.threshold,
                "confirm_steps": config.alarm.confirm_steps,
                "interval_ms": config.resample_interval_ms,
                "history_seconds": DEFAULT_HISTORY_SECONDS,
                "state": pipeline.calibration_manager.state.value,
            }

    def scores(self, seconds: float, max_points: int) -> dict:
        pipeline = self._pipeline
        if pipeline is None or pipeline.history is None:
            return dict(_EMPTY_SCORES)
        return pipeline.history.snapshot(seconds, max_points)
