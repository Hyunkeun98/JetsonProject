from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Snapshot:
    values: dict[str, float | int | None]


class TagBuffer:
    """설정된 태그들의 최신값을 스레드세이프하게 유지하는 캐시.

    한 번 값이 들어오면, 다음 update()가 오기 전까지 snapshot()은 계속
    같은 값을 반환한다 (짧은 결측에 대한 ffill은 이 특성으로 자연히 만족된다).
    """

    def __init__(self, tags: tuple[str, ...]) -> None:
        self._tags = tags
        self._lock = threading.Lock()
        self._latest: dict[str, float | int] = {}

    def update(self, values: dict[str, float | int]) -> None:
        with self._lock:
            for tag, value in values.items():
                if tag in self._tags:
                    self._latest[tag] = value

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(values={tag: self._latest.get(tag) for tag in self._tags})


class SlidingWindow:
    """고정 크기 롤링 윈도우. 용량을 넘으면 가장 오래된 항목을 버린다."""

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._window_size = window_size
        self._items: deque[Snapshot] = deque(maxlen=window_size)

    def push(self, snapshot: Snapshot) -> None:
        self._items.append(snapshot)

    def is_full(self) -> bool:
        return len(self._items) == self._window_size

    def to_list(self) -> list[Snapshot]:
        return list(self._items)
