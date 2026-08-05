from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum

from .calibration import CalibrationSample

_BINARY_VALUES = {0, 1, 0.0, 1.0}


class TagType(Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float


def detect_tag_types(
    samples: list[CalibrationSample], tags: tuple[str, ...]
) -> dict[str, TagType]:
    result: dict[str, TagType] = {}
    for tag in tags:
        observed = [
            s.values[tag] for s in samples if tag in s.values and s.values[tag] is not None
        ]
        # 상수 태그(캘리브레이션 내내 항상 0 또는 항상 1)를 BINARY로 오분류하면
        # 정규화 통계 없이 BCE/sigmoid head가 붙어 운영 중 값이 바뀌는 순간
        # 영구적으로 오알람이 난다. 서로 다른 값이 최소 2개 관측된 경우에만
        # BINARY로 보고, 나머지는 안전한 기본값인 CONTINUOUS로 떨어뜨린다.
        distinct = set(observed)
        if len(distinct) >= 2 and all(v in _BINARY_VALUES for v in distinct):
            result[tag] = TagType.BINARY
        else:
            result[tag] = TagType.CONTINUOUS
    return result


def compute_normalization_stats(
    samples: list[CalibrationSample],
    tags: tuple[str, ...],
    tag_types: dict[str, TagType],
) -> dict[str, NormalizationStats]:
    result: dict[str, NormalizationStats] = {}
    for tag in tags:
        if tag_types.get(tag) != TagType.CONTINUOUS:
            continue
        observed = [
            float(s.values[tag])
            for s in samples
            if tag in s.values and s.values[tag] is not None
        ]
        if not observed:
            result[tag] = NormalizationStats(mean=0.0, std=1.0)
            continue
        mean = statistics.fmean(observed)
        std = statistics.pstdev(observed) if len(observed) > 1 else 0.0
        result[tag] = NormalizationStats(mean=mean, std=std if std > 0 else 1.0)
    return result


def type_indices(tags: tuple[str, ...], tag_types: dict[str, str]) -> tuple[list[int], list[int]]:
    """tag_types(문자열 값 — ModelArtifact에 저장되는 형태)를 기준으로, tags 순서에
    맞는 연속/이진 태그의 인덱스 목록을 만든다. 학습(training.py)과 실시간 추론
    (inference.py)이 태그 인덱스를 항상 이 함수로만 파생시켜, 두 군데서 파생 로직이
    어긋나는 것을 방지한다."""
    continuous_indices = [i for i, t in enumerate(tags) if tag_types[t] == TagType.CONTINUOUS.value]
    binary_indices = [i for i, t in enumerate(tags) if tag_types[t] == TagType.BINARY.value]
    return continuous_indices, binary_indices
