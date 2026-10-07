from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .timeparse import parse_dx1_timestamp

KIND_TRAIN = "train"
KIND_NORMAL_EVAL = "normal_eval"
KIND_ANOMALY = "anomaly"


class SegmentsError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    kind: str
    name: str
    start_ns: int
    end_ns: int  # 구간은 [start_ns, end_ns)


@dataclass(frozen=True)
class SegmentSet:
    train: tuple[Segment, ...]
    normal_eval: tuple[Segment, ...]
    anomaly: tuple[Segment, ...]

    def eval_segments(self) -> tuple[Segment, ...]:
        """점수를 내는 구간(정상 검증 + 이상). 정상 구간이 먼저 온다."""
        return self.normal_eval + self.anomaly

    def all_segments(self) -> tuple[Segment, ...]:
        return self.train + self.normal_eval + self.anomaly


def load_segments(path: str | Path) -> SegmentSet:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise SegmentsError(f"구간 파일을 읽을 수 없습니다: {e}") from e
    except yaml.YAMLError as e:
        raise SegmentsError(f"구간 파일이 올바른 YAML이 아닙니다: {e}") from e
    if not isinstance(raw, dict):
        raise SegmentsError("구간 파일의 최상위는 train/normal_eval/anomaly를 가진 매핑이어야 합니다")
    unknown = set(raw) - {KIND_TRAIN, KIND_NORMAL_EVAL, KIND_ANOMALY}
    if unknown:
        raise SegmentsError(f"알 수 없는 항목: {sorted(unknown)} (train, normal_eval, anomaly만 쓸 수 있습니다)")

    segment_set = SegmentSet(
        train=_parse_kind(raw, KIND_TRAIN),
        normal_eval=_parse_kind(raw, KIND_NORMAL_EVAL),
        anomaly=_parse_kind(raw, KIND_ANOMALY),
    )
    _check_names(segment_set.anomaly)
    _check_no_overlap(segment_set.all_segments())
    return segment_set


def _parse_kind(raw: dict, kind: str) -> tuple[Segment, ...]:
    items = raw.get(kind)
    if items is None:
        return ()
    if not isinstance(items, list):
        raise SegmentsError(f"{kind}는 구간의 목록이어야 합니다")
    segments = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise SegmentsError(f"{kind} {index}번째 항목이 start/end를 가진 매핑이 아닙니다")
        start_ns = _parse_time(item.get("start"), kind, index, "start")
        end_ns = _parse_time(item.get("end"), kind, index, "end")
        if start_ns >= end_ns:
            raise SegmentsError(f"{kind} {index}번째 구간의 start가 end보다 앞이어야 합니다")
        name = item.get("name") or f"{kind}_{index}"
        if not isinstance(name, str):
            raise SegmentsError(f"{kind} {index}번째 구간의 name은 문자열이어야 합니다")
        segments.append(Segment(kind=kind, name=name, start_ns=start_ns, end_ns=end_ns))
    return tuple(segments)


def _parse_time(value: object, kind: str, index: int, field: str) -> int:
    if not isinstance(value, str):
        raise SegmentsError(f"{kind} {index}번째 구간의 {field}가 없습니다")
    ns = parse_dx1_timestamp(value)
    if ns is None:
        raise SegmentsError(
            f"{kind} {index}번째 구간의 {field} '{value}'를 읽을 수 없습니다 "
            "(시간대가 필요합니다. 예: 2026-10-02T11:15:00+09:00)"
        )
    return ns


def _check_names(segments: tuple[Segment, ...]) -> None:
    seen = set()
    for segment in segments:
        if segment.name in seen:
            raise SegmentsError(f"이상 구간 이름이 중복됩니다: {segment.name}")
        seen.add(segment.name)


def _check_no_overlap(segments: tuple[Segment, ...]) -> None:
    ordered = sorted(segments, key=lambda s: s.start_ns)
    for earlier, later in zip(ordered, ordered[1:]):
        if later.start_ns < earlier.end_ns:
            raise SegmentsError(
                f"구간이 겹칩니다: {earlier.kind}({earlier.name})와 {later.kind}({later.name}). "
                "학습과 검증 구간이 겹치면 오탐이 적게 나와 임계값이 낮게 추천됩니다"
            )
