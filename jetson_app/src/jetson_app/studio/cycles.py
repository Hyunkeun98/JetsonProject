"""수집한 원본 데이터에서 사이클 수를 센다. 태그 이름을 가정하지 않는다.

방식 A(0/1 신호의 상승 에지)와 B(정수 카운터의 증가 횟수)는 정확하다. 둘 다 없으면 D(연속 신호의
움직임이 반복되는 주기를 자기상관으로 추정)를 쓰고, 그것도 안 되면 사이클 수를 모른다고 답한다.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from operator import mul
from pathlib import Path

from ..timeparse import parse_dx1_timestamp
from .payloads import iter_records, numeric_value

CYCLE_TARGET = 10
_BUCKET_NS = 500_000_000
_BUCKET_SECONDS = 0.5
_MAX_DISTINCT = 12          # 이보다 많은 값을 가지면 연속 신호로 본다
_MAX_BUCKETS = 2400         # 주기 추정에는 최근 20분만 쓴다
_MIN_LAG = 6                # 3초보다 짧은 주기는 찾지 않는다
_MIN_PERIODS = 3            # 수집 시간이 주기의 3배는 되어야 한다
_MIN_PEAK = 0.3
_MIN_PROMINENCE = 0.1
_AGREEMENT = 0.2            # 태그들이 구한 주기가 서로 20% 안에서 맞아야 한다
_MIN_AGREEING_TAGS = 2      # 한 태그만의 주기는 믿지 않는다(run4: 한 태그가 3.5초짜리 가짜 주기를 냈다)
_MAX_PERIOD_TAGS = 8


@dataclass(frozen=True)
class CycleCandidate:
    tag: str
    method: str     # "edge" | "counter"
    cycles: int


@dataclass(frozen=True)
class CycleReport:
    method: str                 # "edge" | "counter" | "period" | "none"
    cycles: object              # float, 세지 못하면 None
    exact: bool
    tag: object                 # edge/counter일 때 기준 태그
    period_seconds: object
    duration_seconds: float
    candidates: tuple
    message: str

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "cycles": self.cycles,
            "exact": self.exact,
            "tag": self.tag,
            "period_seconds": self.period_seconds,
            "duration_seconds": self.duration_seconds,
            "candidates": [{"tag": c.tag, "method": c.method, "cycles": c.cycles} for c in self.candidates],
            "message": self.message,
            "warning": cycle_warning(self),
        }


def cycle_warning(report: CycleReport):
    """권장 사이클 수보다 적을 때의 경고. 학습을 막지는 않는다."""
    if report.cycles is None:
        return "사이클 수를 알 수 없습니다. 정상 사이클이 10회 이상 들어갔는지 직접 확인하세요."
    if report.cycles < CYCLE_TARGET:
        suffix = "" if report.exact else " (추정값이라 정확하지 않을 수 있습니다)"
        return (
            f"수집된 사이클이 약 {report.cycles:.0f}개로 권장 {CYCLE_TARGET}개보다 적습니다. "
            f"학습 데이터가 적어 알람 기준이 불안정할 수 있습니다.{suffix}"
        )
    return None


class _Tag:
    __slots__ = ("n", "last", "distinct", "ups", "downs", "all_int", "lo", "hi")

    def __init__(self) -> None:
        self.n = 0
        self.last = None
        self.distinct = set()
        self.ups = 0
        self.downs = 0
        self.all_int = True
        self.lo = None
        self.hi = None

    def add(self, value) -> None:
        self.n += 1
        if len(self.distinct) <= _MAX_DISTINCT:
            self.distinct.add(value)
        if self.last is not None:
            if value > self.last:
                self.ups += 1
            elif value < self.last:
                self.downs += 1
        if self.all_int and not float(value).is_integer():
            self.all_int = False
        self.lo = value if self.lo is None or value < self.lo else self.lo
        self.hi = value if self.hi is None or value > self.hi else self.hi
        self.last = value

    def is_continuous(self) -> bool:
        return len(self.distinct) > _MAX_DISTINCT


class CycleCounter:
    def __init__(self) -> None:
        self._tags = {}
        self._activity = {}        # 태그 -> {0.5초 칸: 그 칸에서 값이 움직인 총량}
        self._t_min = None
        self._t_max = None

    @property
    def duration_seconds(self) -> float:
        if self._t_min is None:
            return 0.0
        return (self._t_max - self._t_min) / 1e9

    def feed_payload(self, payload: bytes) -> None:
        for record in iter_records(payload):
            ts = parse_dx1_timestamp(record.get("timestamp"))
            if ts is None:
                continue
            self._t_min = ts if self._t_min is None or ts < self._t_min else self._t_min
            self._t_max = ts if self._t_max is None or ts > self._t_max else self._t_max
            bucket = ts // _BUCKET_NS
            for key, raw in record.items():
                if key == "timestamp":
                    continue
                value = numeric_value(raw)
                if value is None:
                    continue
                tag = self._tags.get(key)
                if tag is None:
                    tag = self._tags[key] = _Tag()
                if tag.last is not None and value != tag.last:
                    series = self._activity.setdefault(key, {})
                    series[bucket] = series.get(bucket, 0.0) + abs(value - tag.last)
                tag.add(value)

    def candidates(self) -> list:
        found = []
        for name, tag in self._tags.items():
            if len(tag.distinct) == 2:
                if tag.ups >= 2:
                    found.append(CycleCandidate(name, "edge", tag.ups))
            elif (
                len(tag.distinct) >= 3
                and tag.downs == 0
                and tag.all_int
                and tag.ups >= 2
                and tag.ups * 2 <= tag.n          # 거의 매 샘플 바뀌는 정수(타이머)는 제외
            ):
                found.append(CycleCandidate(name, "counter", tag.ups))
        found.sort(key=lambda c: (c.cycles, 0 if c.method == "edge" else 1, c.tag))
        return found

    def report(self, preferred_tag=None) -> CycleReport:
        duration = self.duration_seconds
        found = self.candidates()
        if found:
            chosen = next((c for c in found if c.tag == preferred_tag), found[0])
            return CycleReport(
                chosen.method, float(chosen.cycles), True, chosen.tag, None, duration, tuple(found),
                f"'{chosen.tag}' 신호 기준 {chosen.cycles}사이클",
            )
        period = self._period_seconds()
        if period is not None:
            cycles = duration / period
            return CycleReport(
                "period", cycles, False, None, period, duration, (),
                f"주기 약 {period:.0f}초로 추정한 약 {cycles:.0f}사이클 (정확하지 않을 수 있음)",
            )
        return CycleReport(
            "none", None, False, None, None, duration, (),
            "사이클 수를 구하지 못했습니다(사이클 신호나 반복되는 움직임이 없습니다). "
            f"수집 시간은 {duration / 60:.1f}분입니다.",
        )

    def _period_seconds(self):
        """연속 신호의 움직임이 되풀이되는 주기(초). 믿을 만한 주기가 없으면 None."""
        if self._t_min is None:
            return None
        last = self._t_max // _BUCKET_NS
        start = max(self._t_min // _BUCKET_NS, last - _MAX_BUCKETS + 1)
        series_by_tag = []
        for name, tag in self._tags.items():
            activity = self._activity.get(name)
            if not activity or not tag.is_continuous() or tag.hi == tag.lo:
                continue
            x = [activity.get(b, 0.0) for b in range(start, last + 1)]
            mean = sum(x) / len(x)
            if mean <= 0.0:
                continue
            burstiness = statistics.pstdev(x) / mean      # 쉬었다 움직이길 되풀이하면 크고, 늘 떨리는 잡음이면 작다
            series_by_tag.append((burstiness, [v - mean for v in x]))
        series_by_tag.sort(key=lambda item: item[0], reverse=True)
        periods = []
        for _, x in series_by_tag[:_MAX_PERIOD_TAGS]:
            lag = _first_peak(x)
            if lag is not None:
                periods.append(lag * _BUCKET_SECONDS)
        if not periods:
            return None
        middle = statistics.median(periods)
        agreeing = [p for p in periods if abs(p - middle) <= _AGREEMENT * middle]
        if len(agreeing) < _MIN_AGREEING_TAGS or len(agreeing) * 2 < len(periods):
            return None
        period = statistics.median(agreeing)
        return period if self.duration_seconds >= _MIN_PERIODS * period else None


def _first_peak(x):
    """평균을 뺀 열 x의 자기상관에서 처음 나오는 뚜렷한 봉우리의 지연(칸 수). 없으면 None."""
    max_lag = len(x) // _MIN_PERIODS
    if max_lag <= _MIN_LAG + 1:
        return None
    denominator = sum(map(mul, x, x))
    if denominator <= 0.0:
        return None
    r = [1.0] + [sum(map(mul, x, x[k:])) / denominator for k in range(1, max_lag + 2)]
    floor = min(r[1:_MIN_LAG])
    for k in range(_MIN_LAG, max_lag + 1):
        floor = min(floor, r[k - 1])
        if r[k] >= _MIN_PEAK and r[k] - floor >= _MIN_PROMINENCE and r[k] > r[k - 1] and r[k] >= r[k + 1]:
            return k
    return None


def scan_directory(data_dir) -> CycleCounter:
    """폴더의 *.json/*.jsonl(한 줄 = payload 하나)을 읽어 CycleCounter를 채운다."""
    counter = CycleCounter()
    for path in sorted(Path(data_dir).iterdir()):
        if path.suffix not in (".json", ".jsonl"):
            continue
        with path.open("rb") as f:
            for line in f:
                if line.strip():
                    counter.feed_payload(line)
    return counter
