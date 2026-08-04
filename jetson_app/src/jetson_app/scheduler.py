from __future__ import annotations

import threading
from datetime import datetime, timezone

from .buffer import SlidingWindow, TagBuffer
from .calibration import CalibrationManager

_HEARTBEAT_TICK_INTERVAL = 100  # 기본 50ms 간격 기준 약 5초마다


class PeriodicSnapshotter:
    """`interval_ms`마다 TagBuffer 스냅샷을 SlidingWindow에 밀어넣고,
    CalibrationManager에도 전달한다 (CALIBRATING 상태일 때만 실제로 저장됨).
    모든 태그가 None인(한 번도 값을 못 받은) 스냅샷은 건너뛴다.
    """

    def __init__(
        self,
        tag_buffer: TagBuffer,
        sliding_window: SlidingWindow,
        calibration_manager: CalibrationManager,
        interval_ms: int,
    ) -> None:
        self._tag_buffer = tag_buffer
        self._sliding_window = sliding_window
        self._calibration_manager = calibration_manager
        self._interval_seconds = interval_ms / 1000
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_count = 0

    def _tick(self) -> None:
        snapshot = self._tag_buffer.snapshot()
        if all(value is None for value in snapshot.values.values()):
            return
        self._sliding_window.push(snapshot)
        timestamp = datetime.now(timezone.utc).isoformat()
        self._calibration_manager.record_sample(snapshot, timestamp)
        self._tick_count += 1
        if self._tick_count % _HEARTBEAT_TICK_INTERVAL == 0:
            print(
                f"[snapshotter] {self._tick_count}번째 스냅샷 처리, "
                f"윈도우 {len(self._sliding_window.to_list())}/{self._sliding_window.window_size}, "
                f"캘리브레이션 상태={self._calibration_manager.state.value}"
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                # 백그라운드 스레드가 죽으면 데이터 수집이 조용히 멈추므로
                # 어떤 예외도 로그만 남기고 계속 진행한다.
                print(f"[PeriodicSnapshotter] tick 처리 중 오류 발생, 계속 진행: {e}")
            self._stop_event.wait(self._interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
