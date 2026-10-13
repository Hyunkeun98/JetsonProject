from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .scores import ScoreRow

MAX_TAGS_PER_GROUP = 10  # 조합 수가 2^n - 1이라 너무 큰 그룹은 막는다
FEW_ANOMALIES = 3


class TagSelectError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    tags: tuple
    normal_c: float
    anomaly_c: dict  # 이상 구간 이름 -> 임계 점수
    detected: int
    gap: float | None  # 이상 최소 - 정상 최대. 이상 구간이 없으면 None


@dataclass(frozen=True)
class GroupSelection:
    group: str
    all_tags: Candidate  # 모든 태그를 쓴 현재 상태
    candidates: tuple  # 좋은 순서
    single_normal_c: dict  # 태그 하나만 썼을 때의 정상 임계 점수(정상 구간만으로 볼 수 있는 기여도)
    anomalies: tuple  # 이 그룹에 배정된 이상 구간 이름

    @property
    def best(self) -> Candidate:
        return self.candidates[0]


def _critical(rows: list[ScoreRow], tags: tuple, confirm: int) -> float | None:
    """rows(시간 순서)에서, 태그 조합의 점수가 연속 confirm번 넘는 가장 높은 임계값.
    reset 행이나 태그 값이 빠진 행에서 연속이 끊긴다. 계산할 수 있는 행이 없으면 None."""
    best = None
    run: list[float] = []
    for row in rows:
        if row.reset:
            run = []
        values = [row.tag_z.get(t) for t in tags]
        if any(v is None for v in values):
            run = []
            continue
        run.append(max(values))
        if len(run) >= confirm:
            window_min = min(run[-confirm:])
            if best is None or window_min > best:
                best = window_min
    return best


def _evaluate(
    tags: tuple, normal_rows: list[ScoreRow], anomaly_rows: dict, confirm: int
) -> Candidate:
    normal_c = _critical(normal_rows, tags, confirm)
    if normal_c is None:
        raise TagSelectError("정상 구간에서 점수를 낼 수 있는 행이 없습니다")
    anomaly_c = {}
    for name, rows in anomaly_rows.items():
        c = _critical(rows, tags, confirm)
        if c is not None:
            anomaly_c[name] = c
    detected = sum(1 for c in anomaly_c.values() if c > normal_c)
    gap = (min(anomaly_c.values()) - normal_c) if anomaly_c else None
    return Candidate(tags=tags, normal_c=normal_c, anomaly_c=anomaly_c, detected=detected, gap=gap)


def select_tags(
    rows: list[ScoreRow],
    group_tags: dict,
    anomaly_groups: dict,
    confirm: int,
) -> list[GroupSelection]:
    """그룹마다 태그 조합(공집합 제외)을 모두 평가해, 이상을 가장 잘 갈라 내는 조합을 고른다.

    group_tags: 그룹 이름 -> 태그 목록. anomaly_groups: 이상 구간 이름 -> 이 이상이 드러나야 하는 그룹들
    (비어 있으면 모든 그룹에 배정). 순위: 정상 최대보다 높게 나온 이상 수(많을수록) -> 간격(클수록) ->
    태그 수(많을수록, 불필요하게 태그를 빼지 않기 위해)."""
    if confirm <= 0:
        raise TagSelectError("confirm은 1 이상이어야 합니다")
    if not any(row.tag_z for row in rows):
        raise TagSelectError(
            "점수 파일에 태그별 점수(z:<태그> 열)가 없습니다. 이 기능이 들어간 jetson-replay score로 다시 점수를 내세요"
        )
    ordered = sorted(rows, key=lambda r: r.epoch_ns)
    normal_rows = [r for r in ordered if r.kind == "normal_eval"]
    if not normal_rows:
        raise TagSelectError("정상 검증(normal_eval) 구간이 없습니다")
    by_anomaly: dict[str, list[ScoreRow]] = {}
    for row in ordered:
        if row.kind == "anomaly":
            by_anomaly.setdefault(row.segment, []).append(row)

    selections = []
    for group, tags in group_tags.items():
        tags = tuple(tags)
        if len(tags) > MAX_TAGS_PER_GROUP:
            raise TagSelectError(f"그룹 '{group}'의 태그가 {MAX_TAGS_PER_GROUP}개를 넘어 조합을 모두 볼 수 없습니다")
        missing = [t for t in tags if not any(t in r.tag_z for r in normal_rows[:200])]
        if missing:
            raise TagSelectError(f"점수 파일에 이 태그의 점수가 없습니다: {missing}")
        assigned = {
            name: rs
            for name, rs in by_anomaly.items()
            if not anomaly_groups.get(name) or group in anomaly_groups[name]
        }
        cands = []
        for size in range(1, len(tags) + 1):
            for subset in combinations(tags, size):
                cands.append(_evaluate(subset, normal_rows, assigned, confirm))
        everything = next(c for c in cands if len(c.tags) == len(tags))
        single = {c.tags[0]: c.normal_c for c in cands if len(c.tags) == 1}
        cands.sort(
            key=lambda c: (
                -c.detected,
                -(c.gap if c.gap is not None else float("-inf")),
                c.normal_c if not assigned else 0.0,
                -len(c.tags),
            )
        )
        selections.append(
            GroupSelection(
                group=group,
                all_tags=everything,
                candidates=tuple(cands),
                single_normal_c=single,
                anomalies=tuple(assigned),
            )
        )
    return selections


def _short(tag: str) -> str:
    return tag.split(":")[-1]


def format_selection(selections: list[GroupSelection], confirm: int, top: int = 4) -> str:
    lines = [f"=== 태그 조합 선택 (연속 확정 {confirm}번) ==="]
    exclude: list[str] = []
    for sel in selections:
        lines.append(f"\n[그룹 {sel.group}]  배정된 이상 구간 {len(sel.anomalies)}개")
        lines.append("  정상 구간에서 태그 하나만 썼을 때의 최대 점수(오탐이 나지 않는 최소 임계값):")
        for tag, c in sorted(sel.single_normal_c.items(), key=lambda p: -p[1]):
            lines.append(f"    {_short(tag):<16} {c:8.2f}")
        base = sel.all_tags
        lines.append(f"  현재(모든 태그): 정상 최대 {base.normal_c:.2f}, " + _anomaly_text(base))
        if not sel.anomalies:
            lines.append("  이 그룹에 배정된 이상 구간이 없어 조합을 고르지 않았습니다. 위 표를 보고 직접 정하세요.")
            continue
        lines.append(f"  상위 {top}개 조합:")
        for c in sel.candidates[:top]:
            lines.append("    " + "+".join(_short(t) for t in c.tags).ljust(24) + f" 정상 최대 {c.normal_c:6.2f}, " + _anomaly_text(c))
        best = sel.best
        if best.detected == 0:
            lines.append("  -> 제안 없음: 어떤 조합도 정상 최대보다 높은 이상이 없어, 태그를 빼는 것만으로는 해결되지 않습니다")
            if len(sel.anomalies) < FEW_ANOMALIES:
                lines.append(f"  ! 이상 구간이 {len(sel.anomalies)}개뿐입니다")
            continue
        dropped = [t for t in base.tags if t not in best.tags]
        exclude += dropped
        if dropped:
            lines.append("  -> 제안: 점수에서 뺄 태그 " + ", ".join(_short(t) for t in dropped))
            lines.append("     빼면 이 태그에서만 드러나는 이상은 못 잡습니다.")
        else:
            lines.append("  -> 제안: 모든 태그를 그대로 씁니다")
        if len(sel.anomalies) < FEW_ANOMALIES:
            lines.append(f"  ! 이상 구간이 {len(sel.anomalies)}개뿐이라 이 데이터에 과적합한 선택일 수 있습니다. 다른 데이터로 다시 확인하세요")
    lines.append("")
    if exclude:
        lines.append("설정에 적을 값 (모델을 다시 학습할 필요 없음):")
        lines.append("score_exclude_tags:")
        for tag in exclude:
            lines.append(f"  - {tag}")
    else:
        lines.append("뺄 태그가 없습니다.")
    return "\n".join(lines)


def _anomaly_text(c: Candidate) -> str:
    if not c.anomaly_c:
        return "이상 구간 점수 없음"
    low = min(c.anomaly_c.values())
    return f"이상 {c.detected}/{len(c.anomaly_c)}건이 정상 최대보다 높음 (이상 최소 {low:.2f}, 간격 {c.gap:+.2f})"
