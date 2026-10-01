from __future__ import annotations

import threading
from dataclasses import dataclass

from .buffer import Snapshot
from .droplog import DropCounter
from .mqtt_subscriber import Record

# 확정된 칸보다 "크게" 뒤처진 record가 이만큼 연속으로 오면 DX1 시계가 뒤로 보정된 것으로 본다.
_RESYNC_AFTER_FAR_BEHIND = 10


@dataclass(frozen=True)
class ResampledStep:
    """격자 한 칸을 확정한 결과. epoch_ns는 칸의 시작 시각.
    reset_window가 True면 직전 스텝과 윈도우 길이를 넘는 공백이 있었다는 뜻이므로,
    소비자는 슬라이딩 윈도우를 비우고 이 스텝부터 다시 쌓아야 한다."""

    epoch_ns: int
    snapshot: Snapshot
    reset_window: bool = False


class EventTimeResampler:
    """record의 이벤트 시각(timestamp)을 기준으로 값을 `interval_ms` 격자에 배치한다.

    칸 k는 [k·Δ, (k+1)·Δ). 칸 안에 같은 태그의 값이 여럿이면 이벤트 시각이 가장 늦은
    값을, 없으면 직전 값을 유지(ffill)한다. 지금까지 본 가장 늦은 이벤트 시각에서
    `max_lateness_ms`를 뺀 시각 이전의 칸은 확정(방출)한다 — 여러 토픽이 서로 다른
    시점에 도착해도 같은 시각의 값이 한 스냅샷에 모이도록 기다리는 시간이다.
    Jetson 시계는 전혀 쓰지 않는다.
    """

    def __init__(
        self,
        tags: tuple[str, ...],
        interval_ms: int,
        max_lateness_ms: int,
        window_size: int,
    ) -> None:
        self._tags = tags
        self._tag_set = set(tags)
        self._interval_ns = interval_ms * 1_000_000
        self._lateness_ns = max_lateness_ms * 1_000_000
        self._window_size = window_size
        self._lock = threading.Lock()
        # 칸 인덱스 -> {태그: (이벤트 시각 ns, 값)}
        self._pending: dict[int, dict[str, tuple[int, float | int]]] = {}
        self._last_values: dict[str, float | int] = {}
        self._max_ns: int | None = None
        self._next_bucket: int | None = None  # 다음에 방출할 칸. 첫 방출 전에는 None
        self._reset_pending = False
        # 지연 허용 + 윈도우 길이보다 더 뒤처진 record는 단순 지연이 아니라 시계 문제로 본다.
        self._resync_buckets = self._lateness_ns // self._interval_ns + window_size
        self._behind_streak = 0
        self._late = DropCounter("resampler: 확정된 칸보다 늦게 도착해 폐기한 record")

    @property
    def late_dropped(self) -> int:
        return self._late.total

    def add(self, record: Record) -> list[ResampledStep]:
        """record를 반영하고, 이 record 때문에 새로 확정된 스텝들을 시간순으로 반환한다."""
        with self._lock:
            bucket = record.epoch_ns // self._interval_ns
            far_behind = (
                self._next_bucket is not None
                and bucket < self._next_bucket - self._resync_buckets
            )
            if far_behind:
                # 한두 개는 잘못된 타임스탬프로 보고 버린다. 연속으로 오면 DX1 시계가
                # 앞으로 점프했다가 보정된 것이므로, 계속 버리면 점프한 시간만큼 데이터가
                # 끊긴다 -> 새 시각을 기준으로 재동기화한다.
                self._behind_streak += 1
                if self._behind_streak < _RESYNC_AFTER_FAR_BEHIND:
                    self._late.add()
                    return []
                self._resync()
            else:
                self._behind_streak = 0
                if self._next_bucket is not None and bucket < self._next_bucket:
                    self._late.add()
                    return []
            tracked = {t: v for t, v in record.values.items() if t in self._tag_set}
            if not tracked:
                return []
            slot = self._pending.setdefault(bucket, {})
            for tag, value in tracked.items():
                current = slot.get(tag)
                if current is None or record.epoch_ns >= current[0]:
                    slot[tag] = (record.epoch_ns, value)
            if self._max_ns is None or record.epoch_ns > self._max_ns:
                self._max_ns = record.epoch_ns
            return self._drain()

    def _resync(self) -> None:
        print(
            f"[resampler] 이벤트 시각이 {_RESYNC_AFTER_FAR_BEHIND}개 연속으로 이전 시각보다 "
            "크게 뒤로 돌아가 시계가 보정된 것으로 보고 재동기화한다"
        )
        self._pending.clear()
        self._max_ns = None
        self._next_bucket = None
        self._reset_pending = True
        self._behind_streak = 0

    def _drain(self) -> list[ResampledStep]:
        closed_below = (self._max_ns - self._lateness_ns) // self._interval_ns
        if self._next_bucket is None:
            first = min(self._pending)
            if first >= closed_below:
                return []
            self._next_bucket = first
        steps: list[ResampledStep] = []
        while True:
            next_data = min(self._pending) if self._pending else None
            if next_data is not None and next_data - self._next_bucket > self._window_size:
                # 윈도우 길이를 넘는 공백(통신 단절 등): ffill 칸을 만들지 않고 건너뛴다.
                self._next_bucket = next_data
                self._reset_pending = True
            limit = closed_below if next_data is None else min(closed_below, next_data)
            for bucket in range(self._next_bucket, limit):
                self._append_step(steps, bucket, {})
            self._next_bucket = max(self._next_bucket, limit)
            if next_data is not None and next_data < closed_below:
                self._append_step(steps, next_data, self._pending.pop(next_data))
                self._next_bucket = next_data + 1
                continue
            break
        return steps

    def _append_step(
        self,
        steps: list[ResampledStep],
        bucket: int,
        slot: dict[str, tuple[int, float | int]],
    ) -> None:
        for tag, (_, value) in slot.items():
            self._last_values[tag] = value
        values = {tag: self._last_values.get(tag) for tag in self._tags}
        if all(v is None for v in values.values()):
            return  # 아직 어떤 태그도 값을 받지 못한 칸은 방출하지 않는다
        steps.append(
            ResampledStep(
                epoch_ns=bucket * self._interval_ns,
                snapshot=Snapshot(values=values),
                reset_window=self._reset_pending,
            )
        )
        self._reset_pending = False
