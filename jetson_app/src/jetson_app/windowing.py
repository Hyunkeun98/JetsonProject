from __future__ import annotations

import torch

from .calibration import CalibrationSample


def build_windows(
    samples: list[CalibrationSample],
    tags: tuple[str, ...],
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """samples를 (window_size, target) 슬라이딩 윈도우로 잘라 학습용 텐서로 만든다.
    윈도우 안이나 타깃에 None이 하나라도 있으면 그 윈도우는 통째로 건너뛴다."""
    n_tags = len(tags)
    window_rows: list[list[list[float]]] = []
    target_rows: list[list[float]] = []

    for start in range(len(samples) - window_size):
        window = samples[start : start + window_size]
        target = samples[start + window_size]

        window_values: list[list[float]] = []
        window_has_none = False
        for sample in window:
            row = [sample.values.get(tag) for tag in tags]
            if any(v is None for v in row):
                window_has_none = True
                break
            window_values.append([float(v) for v in row])
        if window_has_none:
            continue

        target_row = [target.values.get(tag) for tag in tags]
        if any(v is None for v in target_row):
            continue

        window_rows.append(window_values)
        target_rows.append([float(v) for v in target_row])

    if not window_rows:
        return (
            torch.empty((0, window_size, n_tags), dtype=torch.float32),
            torch.empty((0, n_tags), dtype=torch.float32),
        )

    X = torch.tensor(window_rows, dtype=torch.float32)
    y = torch.tensor(target_rows, dtype=torch.float32)
    return X, y
