from __future__ import annotations

DEFAULT_THRESHOLD = 3.0
DEFAULT_CONFIRM_TICKS = 3


class Debouncer:
    """이상 점수가 threshold를 confirm_ticks번 연속으로 넘어야 알람을 확정한다.
    한 틱이라도 threshold 밑으로 내려가면 카운터를 리셋한다 — 순간적 노이즈로 인한
    오탐을 줄이기 위함(설계 스펙 2026-08-04 문서 6절 5번)."""

    def __init__(
        self, threshold: float = DEFAULT_THRESHOLD, confirm_ticks: int = DEFAULT_CONFIRM_TICKS
    ) -> None:
        if confirm_ticks <= 0:
            raise ValueError("confirm_ticks must be positive")
        self._threshold = threshold
        self._confirm_ticks = confirm_ticks
        self._consecutive_over = 0

    def update(self, score: float) -> bool:
        """새 점수를 반영하고, 현재 알람이 확정 상태인지 반환한다."""
        if score >= self._threshold:
            self._consecutive_over += 1
        else:
            self._consecutive_over = 0
        return self._consecutive_over >= self._confirm_ticks
