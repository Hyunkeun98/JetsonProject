"""torch를 쓰지 않는 공용 타입. inference.py(torch 필요)와 분리해 두면 deviation 전용 구성은 torch 없이 돈다."""
from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class GroupResult:
    score: float
    top_tag: str


class ActiveModelHolder:
    """SnapshotProcessor(매 스텝, 읽기)와 학습 완료 콜백(train 명령 시, 쓰기)이 서로
    다른 스레드에서 접근하는 현재 엔진을 스레드세이프하게 공유한다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine: object | None = None

    def get(self):
        with self._lock:
            return self._engine

    def set(self, engine) -> None:
        with self._lock:
            self._engine = engine
