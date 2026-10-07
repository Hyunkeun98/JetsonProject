from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .debounce import Debouncer
from .scores import ScoreRow
from .segments import KIND_ANOMALY, KIND_NORMAL_EVAL

DEFAULT_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)
DEFAULT_MARGIN = 1.2
MIN_THRESHOLD = 2.0  # 이보다 낮은 임계값은 추천하지 않는다(정상 변동만으로도 자주 넘는다)
MIN_ANOMALY_SEGMENTS = 3
_NEG_INF = float("-inf")
_SECONDS_PER_DAY = 86400.0


class ThresholdError(ValueError):
    pass


@dataclass(frozen=True)
class TableRow:
    threshold: float
    normal_alarms: int
    normal_alarms_per_day: float | None
    anomalies_detected: int
    anomalies_total: int
    mean_latency_s: float | None


@dataclass(frozen=True)
class ConfirmAnalysis:
    confirm: int
    t0: float  # 정상 구간에서 알람이 한 번도 나지 않는 최소 임계값
    t1: float | None  # 모든 이상 구간을 검출하는 최대 임계값(이상 구간이 없으면 None)
    group_t0: dict
    table: tuple[TableRow, ...]
    recommended: float | None
    missed_at_t0: tuple[str, ...]  # t0에서 놓치는 이상 구간 이름
    notes: tuple[str, ...] = field(default_factory=tuple)


def _split_segments(rows: list[ScoreRow]) -> list[tuple[str, str, list[ScoreRow]]]:
    """행을 (종류, 구간 이름, 행들)로 나눈다. 같은 구간의 행은 이어져 있어야 한다."""
    result: list[tuple[str, str, list[ScoreRow]]] = []
    for row in rows:
        if result and result[-1][0] == row.kind and result[-1][1] == row.segment:
            result[-1][2].append(row)
        else:
            result.append((row.kind, row.segment, [row]))
    return result


def _stretches(rows: list[ScoreRow], group: str) -> list[list[float]]:
    """그룹의 점수를 reset 행에서 끊은 연속 구간들로 나눈다. 점수 없는 행은 건너뛴다(앱도 갱신하지 않는다)."""
    stretches: list[list[float]] = [[]]
    for row in rows:
        if row.reset and stretches[-1]:
            stretches.append([])
        entry = row.groups.get(group)
        if entry is not None:
            stretches[-1].append(entry[0])
    return stretches


def _max_window_min(scores: list[float], k: int) -> float:
    """길이 k인 모든 창의 최솟값 중 최댓값. 창이 없으면 -inf."""
    if len(scores) < k:
        return _NEG_INF
    best = _NEG_INF
    window: deque = deque()  # 인덱스, 점수는 증가 순
    for i, score in enumerate(scores):
        while window and scores[window[-1]] >= score:
            window.pop()
        window.append(i)
        if window[0] <= i - k:
            window.popleft()
        if i >= k - 1 and scores[window[0]] > best:
            best = scores[window[0]]
    return best


def critical_score(rows: list[ScoreRow], group_names: tuple[str, ...], confirm: int) -> float:
    """임계값 T에서 이 행들이 알람을 한 번이라도 내는 조건은 T <= critical_score 이다.
    알람은 그룹마다 '점수 >= T가 confirm번 연속'이고 전체 알람은 그룹 알람의 OR이다."""
    best = _NEG_INF
    for group in group_names:
        for stretch in _stretches(rows, group):
            best = max(best, _max_window_min(stretch, confirm))
    return best


def _simulate_segment(
    rows: list[ScoreRow], group_names: tuple[str, ...], threshold: float, confirm: int
) -> tuple[int, int | None]:
    """(알람 건수, 첫 알람 시각 ns). 연속된 알람은 1건으로 센다. 앱의 디바운서를 그대로 쓴다."""
    debouncers = {g: Debouncer(threshold=threshold, confirm_ticks=confirm) for g in group_names}
    episodes = 0
    first_alarm_ns = None
    alarm_on = False
    for row in rows:
        if row.reset:
            for debouncer in debouncers.values():
                debouncer.reset()
            alarm_on = False
        alarm = False
        for group, (score, _top) in row.groups.items():
            if debouncers[group].update(score):
                alarm = True
        if alarm and not alarm_on:
            episodes += 1
            if first_alarm_ns is None:
                first_alarm_ns = row.epoch_ns
        alarm_on = alarm
    return episodes, first_alarm_ns


def _round_up_above(value: float) -> float:
    """value보다 엄격히 큰 0.1 단위 값 중 가장 작은 값."""
    return round(math.floor(value * 10 + 1e-9) / 10 + 0.1, 1)


def _round_down(value: float) -> float:
    return round(math.floor(value * 10 + 1e-9) / 10, 1)


def _round_up(value: float) -> float:
    return round(math.ceil(value * 10 - 1e-9) / 10, 1)


def analyze(
    rows: list[ScoreRow],
    group_names: tuple[str, ...],
    confirm: int,
    grid: tuple[float, ...] = DEFAULT_GRID,
    margin: float = DEFAULT_MARGIN,
    max_alarms_per_day: float | None = None,
) -> ConfirmAnalysis:
    segments = _split_segments(rows)
    normal = [(name, seg_rows) for kind, name, seg_rows in segments if kind == KIND_NORMAL_EVAL]
    anomalies = [(name, seg_rows) for kind, name, seg_rows in segments if kind == KIND_ANOMALY]
    if not normal:
        raise ThresholdError("정상 검증 구간(normal_eval)의 점수가 없습니다. 임계값을 추천할 수 없습니다")
    notes: list[str] = []

    normal_c = max(critical_score(seg_rows, group_names, confirm) for _name, seg_rows in normal)
    t0 = MIN_THRESHOLD if normal_c == _NEG_INF else max(MIN_THRESHOLD, _round_up_above(normal_c))
    group_t0 = {}
    for group in group_names:
        c = max(critical_score(seg_rows, (group,), confirm) for _name, seg_rows in normal)
        group_t0[group] = MIN_THRESHOLD if c == _NEG_INF else max(MIN_THRESHOLD, _round_up_above(c))

    anomaly_c = {name: critical_score(seg_rows, group_names, confirm) for name, seg_rows in anomalies}
    t1 = None
    if anomaly_c:
        lowest = min(anomaly_c.values())
        t1 = lowest if lowest == _NEG_INF else _round_down(lowest)
    missed_at_t0 = tuple(name for name, c in anomaly_c.items() if c < t0)

    normal_seconds = sum(
        (seg_rows[-1].epoch_ns - seg_rows[0].epoch_ns) / 1e9 for _name, seg_rows in normal
    )
    grid_values = tuple(sorted(set(grid)))
    table = tuple(
        _table_row(threshold, normal, anomalies, group_names, confirm, normal_seconds)
        for threshold in grid_values
    )

    recommended = None
    if not anomalies:
        notes.append("이상 구간이 없어 민감도(이상을 얼마나 잡는지)는 검증하지 못했습니다")
        if max_alarms_per_day is not None:
            fits = [
                row.threshold
                for row in table
                if row.normal_alarms_per_day is not None
                and row.normal_alarms_per_day <= max_alarms_per_day
            ]
            if fits:
                recommended = fits[0]
            else:
                notes.append(
                    f"격자의 어떤 임계값도 하루 {max_alarms_per_day:g}건 이하가 되지 않습니다"
                )
        else:
            recommended = max(t0, _round_up(t0 * margin))
    else:
        if len(anomalies) < MIN_ANOMALY_SEGMENTS:
            notes.append(
                f"이상 구간이 {len(anomalies)}개뿐이라 표본이 적습니다. 추천값이 이 데이터에 과적합일 수 있습니다"
            )
        if t1 is not None and t1 != _NEG_INF and t0 <= t1:
            middle = math.floor((t0 + t1) / 2 * 10 + 0.5) / 10
            recommended = round(min(max(middle, t0), t1), 1)
        else:
            notes.append(
                "정상 오탐 0건과 모든 이상 검출을 동시에 만족하는 임계값이 없어 추천하지 않습니다"
            )
    return ConfirmAnalysis(
        confirm=confirm,
        t0=t0,
        t1=t1,
        group_t0=group_t0,
        table=table,
        recommended=recommended,
        missed_at_t0=missed_at_t0,
        notes=tuple(notes),
    )


def _table_row(threshold, normal, anomalies, group_names, confirm, normal_seconds) -> TableRow:
    alarms = sum(
        _simulate_segment(seg_rows, group_names, threshold, confirm)[0] for _n, seg_rows in normal
    )
    per_day = alarms / normal_seconds * _SECONDS_PER_DAY if normal_seconds > 0 else None
    detected = 0
    latencies = []
    for _name, seg_rows in anomalies:
        episodes, first_ns = _simulate_segment(seg_rows, group_names, threshold, confirm)
        if episodes > 0:
            detected += 1
            latencies.append((first_ns - seg_rows[0].epoch_ns) / 1e9)
    mean_latency = sum(latencies) / len(latencies) if latencies else None
    return TableRow(
        threshold=threshold,
        normal_alarms=alarms,
        normal_alarms_per_day=per_day,
        anomalies_detected=detected,
        anomalies_total=len(anomalies),
        mean_latency_s=mean_latency,
    )


def format_report(analysis: ConfirmAnalysis) -> str:
    lines = [f"=== 연속 확정 횟수(confirm) {analysis.confirm} ==="]
    lines.append(f"정상 오탐 0건이 되는 최소 임계값: {analysis.t0}")
    for group, value in analysis.group_t0.items():
        lines.append(f"  그룹 {group}: {value}")
    if analysis.t1 is not None:
        lines.append(f"모든 이상을 검출하는 최대 임계값: {analysis.t1}")
    header = f"{'임계값':>7} {'정상알람':>8} {'건/일':>8} {'이상검출':>9} {'평균지연(s)':>11}"
    lines.append(header)
    for row in analysis.table:
        per_day = "-" if row.normal_alarms_per_day is None else f"{row.normal_alarms_per_day:.1f}"
        latency = "-" if row.mean_latency_s is None else f"{row.mean_latency_s:.1f}"
        detected = f"{row.anomalies_detected}/{row.anomalies_total}"
        lines.append(
            f"{row.threshold:>7g} {row.normal_alarms:>8d} {per_day:>8} {detected:>9} {latency:>11}"
        )
    if analysis.recommended is not None:
        lines.append(f"추천 임계값: {analysis.recommended}")
    else:
        lines.append("추천 임계값: 없음")
    if analysis.missed_at_t0:
        lines.append(f"{analysis.t0}에서 놓치는 이상 구간: {', '.join(analysis.missed_at_t0)}")
    for note in analysis.notes:
        lines.append(f"* {note}")
    return "\n".join(lines)
