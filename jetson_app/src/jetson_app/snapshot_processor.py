from __future__ import annotations

import queue
import threading

from .buffer import SlidingWindow
from .calibration import CalibrationManager, CalibrationState
from .debounce import Debouncer
from .droplog import DropCounter
from .inference import ActiveModelHolder, GroupResult
from .publisher import GroupOutput, ResultPublisher
from .resampler import ResampledStep
from .timeparse import format_epoch_ns

_HEARTBEAT_STEP_INTERVAL = 100  # 처리한 스텝 100개마다 한 줄 (50ms 격자면 약 5초)
_DEFAULT_MAX_QUEUE_SIZE = 10_000


class SnapshotProcessor:
    """리샘플러가 확정한 스텝(`ResampledStep`)을 큐로 받아 처리하는 워커.

    MQTT 스레드는 `submit()`으로 큐에 넣기만 하고, 점수 계산(GRU 추론)은 별도 스레드가
    한다. 스텝마다 하는 일: 새 스냅샷을 윈도우에 넣기 *전에* 그 시점까지의 윈도우로 다음
    값을 예측해 이상 점수를 계산·발행하고(MONITORING이고 모델이 있을 때만), 슬라이딩
    윈도우와 캘리브레이션 버퍼에 스냅샷을 쌓는다. 스냅샷을 먼저 넣으면 "미래"를 보고
    예측하는 꼴이 되어 스코어링이 무의미해진다.
    """

    def __init__(
        self,
        sliding_window: SlidingWindow,
        calibration_manager: CalibrationManager,
        inference_engine_holder: ActiveModelHolder | None = None,
        debouncers: dict[str, Debouncer] | None = None,
        result_publisher: ResultPublisher | None = None,
        max_queue_size: int = _DEFAULT_MAX_QUEUE_SIZE,
    ) -> None:
        self._sliding_window = sliding_window
        self._calibration_manager = calibration_manager
        self._inference_engine_holder = inference_engine_holder
        self._debouncers = debouncers
        self._result_publisher = result_publisher
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._dropped = DropCounter("snapshot_processor: 처리가 밀려 폐기한 스냅샷")
        self._stop_event = threading.Event()
        self._thread = None
        self._step_count = 0

    @property
    def dropped(self) -> int:
        return self._dropped.total

    def submit(self, steps: list[ResampledStep]) -> None:
        """MQTT 스레드에서 호출. 큐가 가득 차면 가장 오래된 스텝을 버리고 로그를 남긴다."""
        for step in steps:
            while True:
                try:
                    self._queue.put_nowait(step)
                    break
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._dropped.add()

    def process_pending(self) -> None:
        """큐에 쌓인 스텝을 호출한 스레드에서 모두 처리한다(stop()과 테스트에서 사용)."""
        while True:
            try:
                step = self._queue.get_nowait()
            except queue.Empty:
                return
            self._process_safely(step)

    def process(self, step: ResampledStep) -> None:
        if step.reset_window:
            # 통신 단절 등으로 윈도우 길이를 넘는 공백이 있었다: 공백 앞뒤를 이어 붙인
            # 윈도우로 학습/추론하지 않도록 비우고, 연속 초과 카운터도 되돌린다.
            self._sliding_window.clear()
            for debouncer in (self._debouncers or {}).values():
                debouncer.reset()
        snapshot = step.snapshot
        timestamp = format_epoch_ns(step.epoch_ns)

        self._score_and_publish(timestamp, snapshot)

        self._sliding_window.push(snapshot)
        self._calibration_manager.record_sample(snapshot, timestamp)
        self._step_count += 1
        if self._step_count % _HEARTBEAT_STEP_INTERVAL == 0:
            print(
                f"[snapshotter] {self._step_count}번째 스냅샷 처리, "
                f"윈도우 {len(self._sliding_window.to_list())}/{self._sliding_window.window_size}, "
                f"캘리브레이션 상태={self._calibration_manager.state.value}"
            )

    def _score_and_publish(self, timestamp: str, snapshot) -> None:
        # 세 협력자는 함께 있어야만 채점이 성립한다. 하나라도 없는 상태로 진행하면
        # 뒤쪽 update()/publish()에서 AttributeError가 나고, _process_safely의 포괄 예외
        # 처리에 삼켜져 매 스텝 로그만 쏟아진다.
        if (
            self._inference_engine_holder is None
            or self._debouncers is None
            or self._result_publisher is None
        ):
            return
        if self._calibration_manager.state != CalibrationState.MONITORING:
            return
        if not self._sliding_window.is_full():
            return
        engine = self._inference_engine_holder.get()
        if engine is None:
            return
        result = engine.score(self._sliding_window.to_list(), snapshot)
        if result is None:
            return
        # 그룹 구성이 없는 결과(평면 설정)는 전체 점수를 단일 그룹 "all"로 본다.
        group_results = result.group_results or {
            "all": GroupResult(score=result.anomaly_score, top_tag=result.top_deviant_tag)
        }
        outputs = {
            name: GroupOutput(
                score=group.score,
                alarm=self._debouncers[name].update(group.score),
                top_tag=group.top_tag,
            )
            for name, group in group_results.items()
        }
        alarm = any(output.alarm for output in outputs.values())
        self._result_publisher.publish(
            timestamp, result.anomaly_score, alarm, result.top_deviant_tag, outputs
        )

    def _process_safely(self, step: ResampledStep) -> None:
        try:
            self.process(step)
        except Exception as e:
            # 백그라운드 스레드가 죽으면 데이터 수집이 조용히 멈추므로
            # 어떤 예외도 로그만 남기고 계속 진행한다.
            print(f"[SnapshotProcessor] 스텝 처리 중 오류 발생, 계속 진행: {e}")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                step = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._process_safely(step)

    def start(self) -> None:
        # stop()이 set()한 이벤트를 지우지 않으면, stop() 이후 재시작된 스레드는
        # 루프 조건이 이미 참(정지)인 채로 시작해 아무것도 처리하지 않는다.
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        # 정지 직전까지 제출된 스텝이 큐에 남지 않도록 마저 처리한다.
        self.process_pending()
