"""화면이 쓰는 앱 제어 계층: 6단계(연결 → 살펴보기 → 전체 수집 → 태그 추천 → 학습 → 감시)를 관리한다.

새로 계산하는 로직은 없다. 사이클 세기·수집·시험은 각 모듈, 학습은 trainer, 감시 파이프라인은
build_pipeline, 태그 추천은 deviation_suggest를 그대로 부른다.
"""
from __future__ import annotations

import threading
import time
from enum import Enum
from pathlib import Path

from ..artifact_paths import state_marker_path
from ..calibration import CalibrationState, StateStore
from ..deviation import deviation_artifact_path, load_deviation_artifact, make_deviation_train_fn
from ..pipeline import build_pipeline
from ..score_history import DEFAULT_HISTORY_SECONDS
from . import wizard
from .collector import DEFAULT_MAX_BYTES, Collector
from .errors import StudioError
from .probe import ProbeSession
from .raw_source import RawSource
from .trainer import train_from_collection

_EMPTY_SCORES = {"now_ms": None, "bucket_ms": 0, "points": []}


class Stage(str, Enum):
    CONNECT = "CONNECT"    # 1: 연결 정보가 없다
    PROBE = "PROBE"        # 2: 연결 정보는 있고 수집한 데이터는 없다(살펴보기)
    COLLECT = "COLLECT"    # 3: 전체 수집 중
    TAGS = "TAGS"          # 4: 수집본이 있고 학습한 모델은 없다(태그 추천)
    TRAIN = "TRAIN"        # 5: 학습 중
    MONITOR = "MONITOR"    # 6: 학습한 모델이 있다(감시 중이거나 감시를 시작할 수 있다)


STEP = {Stage.CONNECT: 1, Stage.PROBE: 2, Stage.COLLECT: 3, Stage.TAGS: 4, Stage.TRAIN: 5, Stage.MONITOR: 6}


class MqttIO:
    """감시 파이프라인의 MQTT 클라이언트를 백그라운드 스레드로 돌리고 멈춘다(테스트에서는 가짜로 바꾼다)."""

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
        source_factory=RawSource,
        pipeline_factory=build_pipeline,
        clock=time.monotonic,
        run_worker: bool = True,
        max_collect_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._workspace = Path(workspace)
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._calibration_dir = self._workspace / "calibration"
        self._model_dir = self._workspace / "models"
        self._io = io or MqttIO()
        self._source_factory = source_factory
        self._factory = pipeline_factory
        self._clock = clock
        self._run_worker = run_worker
        self._lock = threading.RLock()
        self._collector = Collector(self._workspace, source_factory, clock=clock, max_bytes=max_collect_bytes)
        self._probe = None
        self._pipeline = None
        self._settings = None
        self._config = None
        self._training = False
        self._analyzing = 0              # 분석(오래 걸릴 수 있음)이 도는 횟수
        self._train_thread = None
        self._error = None
        self._load_saved()

    # --- 저장된 상태 ----------------------------------------------------------------------

    def _load_saved(self) -> None:
        self._settings = wizard.load_settings(self._workspace)
        if self._settings is None:
            return
        self._collector.set_equipment(self._settings["equipment_id"])
        try:
            self._config = wizard.load_config(self._workspace)
        except StudioError as e:
            self._error = str(e)

    def _need_settings(self) -> dict:
        if self._settings is None:
            raise StudioError(wizard.NO_CONNECTION)
        return self._settings

    def _require_idle(self, action: str) -> None:
        """감시·학습·수집 중에는 새 작업을 시작할 수 없다."""
        if self._pipeline is not None:
            raise StudioError(f"감시 중에는 {action}할 수 없습니다. 먼저 감시를 중지하세요.")
        if self._training:
            raise StudioError(f"학습이 끝난 뒤에 {action}할 수 있습니다.")
        if self._analyzing:
            raise StudioError(f"분석이 끝난 뒤에 {action}할 수 있습니다.")
        if self._collector.state()["active"]:
            raise StudioError(f"수집 중에는 {action}할 수 없습니다. 먼저 수집을 종료하세요.")

    def _model_ready_locked(self) -> bool:
        if self._settings is None:
            return False
        eid = self._settings["equipment_id"]
        if not deviation_artifact_path(self._model_dir, eid).exists():
            return False
        return StateStore(state_marker_path(self._model_dir, eid)).read() == CalibrationState.MONITORING

    def _reset_model_locked(self) -> None:
        """학습 결과를 감시에 쓰지 않도록 표시한다(파일은 지우지 않고 다음 학습이 덮어쓴다)."""
        if self._settings is not None:
            StateStore(state_marker_path(self._model_dir, self._settings["equipment_id"])).write(
                CalibrationState.CALIBRATING
            )

    # --- 1단계: 연결 ---------------------------------------------------------------------

    def connection(self) -> dict:
        return wizard.saved_connection(self._workspace)

    def save_connection(self, form: dict) -> None:
        with self._lock:
            if self._pipeline is not None or self._training or self._analyzing or self._collector.state()["active"]:
                raise StudioError("진행 중인 작업(감시, 학습, 분석, 수집)이 있습니다. 끝나거나 중지한 뒤에 저장하세요.")
            if self._probe is not None and self._probe.snapshot()["active"]:
                raise StudioError("살펴보기가 진행 중입니다. 먼저 시험을 종료하세요.")
            wizard.save_connection(self._workspace, form)
            self._settings = wizard.load_settings(self._workspace)
            self._collector.set_equipment(self._settings["equipment_id"])
            self._config = None
            self._error = None
            try:
                self._config = wizard.load_config(self._workspace)
            except StudioError as e:
                self._error = str(e)

    # --- 2단계: 살펴보기 -----------------------------------------------------------------

    def probe_start(self) -> None:
        with self._lock:
            settings = self._need_settings()
            self._require_idle("시험을 시작")
            if self._probe is not None and self._probe.snapshot()["active"]:
                return
            probe = ProbeSession(
                self._source_factory(), settings["host"], settings["port"], settings["topics"], clock=self._clock
            )
            try:
                probe.start()
            except OSError as e:
                raise StudioError(
                    f"브로커({settings['host']}:{settings['port']})에 연결할 수 없습니다. "
                    f"브로커가 실행 중인지 확인하세요. ({e})"
                ) from e
            self._probe = probe
            self._error = None

    def probe_stop(self) -> None:
        with self._lock:
            self._stop_probe_locked()

    def _stop_probe_locked(self) -> None:
        if self._probe is not None:
            self._probe.stop()

    # --- 3단계: 전체 수집 ----------------------------------------------------------------

    def collect_start(self) -> None:
        with self._lock:
            settings = self._need_settings()
            self._require_idle("수집을 시작")
            self._stop_probe_locked()
            self._collector.start(settings["host"], settings["port"], settings["topics"])
            self._reset_model_locked()                      # 새 데이터로 다시 학습해야 한다
            self._error = None

    def collect_stop(self) -> None:
        with self._lock:
            self._collector.stop()

    def discard(self) -> None:
        with self._lock:
            self._need_settings()
            self._require_idle("지우기")
            self._collector.discard()

    def import_folder(self, data_dir: str) -> None:
        with self._lock:
            self._need_settings()
            self._require_idle("불러오기")
            self._stop_probe_locked()
            self._collector.import_folder(data_dir)
            self._reset_model_locked()
            self._error = None

    # --- 4단계: 태그 추천 ----------------------------------------------------------------

    def analyze(self, cycle_tag=None, with_tags: bool = True) -> dict:
        """수집본으로 태그를 추천하고 사이클 수를 센다. 오래 걸릴 수 있어 컨트롤러 락을 잡지 않는다."""
        with self._lock:
            self._need_settings()
            if self._training or self._collector.state()["active"]:
                raise StudioError("수집이나 학습이 끝난 뒤에 분석할 수 있습니다.")
            if not self._collector.state()["has_data"]:
                raise StudioError("수집한 데이터가 없습니다. 먼저 3단계에서 수집하세요.")
            directory = self._collector.current_dir
            self._analyzing += 1                    # 분석하는 동안 수집·지우기·불러오기가 같은 파일을 건드리지 못하게 한다
        try:
            result = {}
            if with_tags:
                result.update(wizard.analyze(str(directory)))
            result["cycles"] = self._collector.cycle_report(cycle_tag).to_dict()
            return result
        finally:
            with self._lock:
                self._analyzing -= 1

    # --- 5단계: 학습 ---------------------------------------------------------------------

    def train(self, tags) -> None:
        with self._lock:
            self._need_settings()
            if self._training:
                raise StudioError("이미 학습 중입니다.")
            self._require_idle("학습")
            if not self._collector.state()["has_data"]:
                raise StudioError("수집한 데이터가 없습니다. 먼저 3단계에서 수집하세요.")
            config = wizard.save_config(self._workspace, tags)
            self._config = config
            self._reset_model_locked()                       # 실패하면 4단계에 남도록 먼저 표시한다
            self._training = True
            self._error = None
            directory = self._collector.current_dir

        def _run() -> None:
            try:
                train_from_collection(config, directory, self._model_dir)
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

    # --- 6단계: 감시 ---------------------------------------------------------------------

    @property
    def pipeline(self):
        return self._pipeline

    def start(self) -> None:
        with self._lock:
            settings = self._need_settings()
            if self._pipeline is not None:
                raise StudioError("이미 감시 중입니다.")
            self._require_idle("감시를 시작")
            if not self._model_ready_locked():
                raise StudioError("학습이 끝난 모델이 없습니다. 먼저 4~5단계에서 학습하세요.")
            # 연결 정보가 바뀌었을 수 있으니 같은 태그로 설정 파일을 다시 만든다.
            config = wizard.save_config(self._workspace, wizard.saved_tags(self._workspace))
            self._config = config
            train_fn = make_deviation_train_fn(config, deviation_artifact_path(self._model_dir, config.equipment_id))
            pipeline = self._factory(
                config=config,
                calibration_dir=self._calibration_dir,
                model_dir=self._model_dir,
                train_fn=train_fn,
                history_seconds=DEFAULT_HISTORY_SECONDS,
            )
            if pipeline.calibration_manager.state != CalibrationState.MONITORING:
                raise StudioError("저장된 학습 결과를 불러오지 못했습니다. 4단계에서 다시 학습하세요(콘솔 로그에 이유가 있습니다).")
            try:
                self._io.connect(pipeline, settings["host"], settings["port"])
            except OSError as e:
                raise StudioError(
                    f"브로커({settings['host']}:{settings['port']})에 연결할 수 없습니다. "
                    f"브로커가 실행 중인지 확인하세요. ({e})"
                ) from e
            if self._run_worker:
                pipeline.snapshotter.start()
            self._pipeline = pipeline
            self._error = None

    def stop(self) -> None:
        with self._lock:
            if self._pipeline is None:
                return
            pipeline = self._pipeline
            self._pipeline = None
        if self._run_worker:
            pipeline.snapshotter.stop()
        self._io.disconnect(pipeline)

    def retrain(self) -> None:
        """감시를 멈추고 4단계로 돌아간다(같은 수집본으로 태그를 다시 골라 학습한다)."""
        self.stop()
        with self._lock:
            if self._training:
                raise StudioError("학습이 끝난 뒤에 다시 학습할 수 있습니다.")
            self._reset_model_locked()
            self._error = None

    def shutdown(self) -> None:
        """앱을 끝낼 때: 시험·수집·감시를 모두 닫는다. 이미 닫혀 있어도 오류를 내지 않는다."""
        for action in (self.probe_stop, self.collect_stop, self.stop):
            try:
                action()
            except Exception as e:
                print(f"[studio] 종료 정리 중 오류(무시): {e!r}")

    def feed_record(self, record) -> None:
        """MQTT 스레드가 하는 일(리샘플 후 처리 큐에 넣기)을 그대로 한다. 시험과 재생용."""
        pipeline = self._pipeline
        if pipeline is None:
            raise StudioError("감시 중이 아닙니다.")
        pipeline.snapshotter.submit(pipeline.resampler.add(record))

    def pump(self) -> None:
        """처리 큐를 동기로 비운다(run_worker=False일 때 시험이 쓴다)."""
        if self._pipeline is not None:
            self._pipeline.snapshotter.process_pending()

    # --- 화면용 조회 ----------------------------------------------------------------------

    def _stage_locked(self, collect: dict) -> Stage:
        if self._settings is None:
            return Stage.CONNECT
        if self._pipeline is not None:
            return Stage.MONITOR
        if self._training:
            return Stage.TRAIN
        if collect["active"]:
            return Stage.COLLECT
        if self._model_ready_locked():
            return Stage.MONITOR
        if collect["has_data"]:
            return Stage.TAGS
        return Stage.PROBE

    def _can_locked(self, collect: dict, probe, ready: bool) -> dict:
        settings = self._settings is not None
        running = self._pipeline is not None
        probing = bool(probe and probe["active"])
        collecting = collect["active"]
        analyzing = self._analyzing > 0
        idle = settings and not (running or self._training or collecting or analyzing)
        return {
            "save_connection": not (running or self._training or collecting or probing or analyzing),
            "probe_start": idle and not probing,
            "probe_stop": probing,
            "collect_start": idle,
            "collect_stop": collecting,
            "discard": idle and collect["has_data"],
            "import": idle,
            "analyze": idle and collect["has_data"],
            "train": idle and collect["has_data"],
            "start": idle and ready,
            "stop": running,
            "retrain": settings and not (self._training or collecting) and (running or ready),
        }

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
            collect = self._collector.state()
            probe = self._probe.snapshot() if self._probe is not None else None
            stage = self._stage_locked(collect)
            config = self._config
            ready = self._model_ready_locked()
            return {
                "stage": stage.value,
                "step": STEP[stage],
                "equipment_id": self._settings["equipment_id"] if self._settings else None,
                "error": self._error,
                "probe": probe,
                "collect": collect,
                "training": self._training,
                "analyzing": self._analyzing > 0,
                "model_ready": ready,
                "running": self._pipeline is not None,
                "connected": bool(self._pipeline is not None and self._io.is_connected(self._pipeline)),
                "signals": self._signals(config) if (ready and config is not None) else [],
                "threshold": config.alarm.threshold if config else None,
                "can": self._can_locked(collect, probe, ready),
            }

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
                    "state": "IDLE",
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
