from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .calibration import CalibrationSample, TrainFn
from .model import AnomalyGRU
from .tag_stats import TagType, compute_normalization_stats, detect_tag_types
from .windowing import build_windows

DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64


@dataclass(frozen=True)
class ModelArtifact:
    tags: tuple[str, ...]
    tag_types: dict[str, str]
    norm_stats: dict[str, tuple[float, float]]
    error_stats: dict[str, tuple[float, float]]
    window_size: int
    hidden_size: int
    num_layers: int
    state_dict: dict


def train_model(
    samples: list[CalibrationSample],
    tags: tuple[str, ...],
    window_size: int,
    epochs: int = DEFAULT_EPOCHS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ModelArtifact:
    tag_types = detect_tag_types(samples, tags)
    norm_stats = compute_normalization_stats(samples, tags, tag_types)

    continuous_indices = [i for i, t in enumerate(tags) if tag_types[t] == TagType.CONTINUOUS]
    binary_indices = [i for i, t in enumerate(tags) if tag_types[t] == TagType.BINARY]

    X, y = build_windows(samples, tags, window_size)
    if X.shape[0] == 0:
        raise ValueError(
            f"not enough calibration samples to build any training window "
            f"(have {len(samples)} samples, need > {window_size})"
        )

    for idx in continuous_indices:
        tag = tags[idx]
        stats = norm_stats[tag]
        X[:, :, idx] = (X[:, :, idx] - stats.mean) / stats.std
        y[:, idx] = (y[:, idx] - stats.mean) / stats.std

    model = AnomalyGRU(
        num_tags=len(tags),
        continuous_indices=continuous_indices,
        binary_indices=binary_indices,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    n_windows = X.shape[0]
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(n_windows)
        for start in range(0, n_windows, batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch_X = X[batch_idx]
            batch_y = y[batch_idx]

            optimizer.zero_grad()
            continuous_out, binary_logits = model(batch_X)

            loss = torch.tensor(0.0)
            if continuous_indices:
                loss = loss + mse_loss_fn(continuous_out, batch_y[:, continuous_indices])
            if binary_indices:
                loss = loss + bce_loss_fn(binary_logits, batch_y[:, binary_indices])

            loss.backward()
            optimizer.step()

    error_stats = _compute_error_stats(model, X, y, tags, continuous_indices, binary_indices)

    return ModelArtifact(
        tags=tags,
        tag_types={t: tag_types[t].value for t in tags},
        norm_stats={t: (s.mean, s.std) for t, s in norm_stats.items()},
        error_stats=error_stats,
        window_size=window_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        state_dict=model.state_dict(),
    )


def _compute_error_stats(
    model: AnomalyGRU,
    X: torch.Tensor,
    y: torch.Tensor,
    tags: tuple[str, ...],
    continuous_indices: list[int],
    binary_indices: list[int],
) -> dict[str, tuple[float, float]]:
    """학습 완료 후 캘리브레이션 데이터 전체에 대한 태그별 "정상 오차" 평균/표준편차.
    실시간 이상 점수 계산(다음 계획)에서 원본 오차를 재정규화하는 기준으로 쓰인다."""
    model.eval()
    with torch.no_grad():
        continuous_out, binary_logits = model(X)

    errors: dict[str, torch.Tensor] = {}
    for pos, idx in enumerate(continuous_indices):
        errors[tags[idx]] = torch.abs(y[:, idx] - continuous_out[:, pos])
    for pos, idx in enumerate(binary_indices):
        pred_prob = torch.sigmoid(binary_logits[:, pos])
        errors[tags[idx]] = torch.abs(y[:, idx] - pred_prob)

    result: dict[str, tuple[float, float]] = {}
    for tag, err in errors.items():
        mean = err.mean().item()
        if err.numel() > 1:
            variance = ((err - err.mean()) ** 2).mean().item()
            std = variance ** 0.5
        else:
            std = 0.0
        result[tag] = (mean, std if std > 0 else 1.0)
    return result


def save_artifact(path: str | Path, artifact: ModelArtifact) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tags": artifact.tags,
            "tag_types": artifact.tag_types,
            "norm_stats": artifact.norm_stats,
            "error_stats": artifact.error_stats,
            "window_size": artifact.window_size,
            "hidden_size": artifact.hidden_size,
            "num_layers": artifact.num_layers,
            "state_dict": artifact.state_dict,
        },
        path,
    )


def load_artifact(path: str | Path) -> ModelArtifact:
    data = torch.load(Path(path), weights_only=False)
    return ModelArtifact(
        tags=tuple(data["tags"]),
        tag_types=data["tag_types"],
        norm_stats=data["norm_stats"],
        error_stats=data["error_stats"],
        window_size=data["window_size"],
        hidden_size=data["hidden_size"],
        num_layers=data["num_layers"],
        state_dict=data["state_dict"],
    )


def make_train_fn(
    tags: tuple[str, ...],
    window_size: int,
    model_path: str | Path,
    epochs: int = DEFAULT_EPOCHS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> TrainFn:
    def _train_fn(samples: list[CalibrationSample]) -> None:
        artifact = train_model(
            samples,
            tags=tags,
            window_size=window_size,
            epochs=epochs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        save_artifact(model_path, artifact)

    return _train_fn
