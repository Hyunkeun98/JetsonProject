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
        if observed and all(v in _BINARY_VALUES for v in observed):
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
