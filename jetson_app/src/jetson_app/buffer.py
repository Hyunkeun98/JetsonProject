from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Snapshot:
    values: dict[str, float | int | None]


class SlidingWindow:
    """고정 크기 롤링 윈도우. 용량을 넘으면 가장 오래된 항목을 버린다."""

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._window_size = window_size
        self._items: deque[Snapshot] = deque(maxlen=window_size)

    @property
    def window_size(self) -> int:
        return self._window_size

    def push(self, snapshot: Snapshot) -> None:
        self._items.append(snapshot)

    def clear(self) -> None:
        self._items.clear()

    def is_full(self) -> bool:
        return len(self._items) == self._window_size

    def to_list(self) -> list[Snapshot]:
        return list(self._items)
