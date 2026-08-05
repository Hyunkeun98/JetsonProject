from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .calibration import CalibrationSample, TrainFn
from .model import AnomalyGRU
from .tag_stats import compute_normalization_stats, detect_tag_types, type_indices
from .windowing import build_windows

DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
# 캘리브레이션 버퍼는 calibration.max_duration까지 자라기 때문에(예: 7d @ 50ms ≈ 12M 샘플)
# 전체를 윈도잉하면 수 GB 텐서가 되어 4GB Jetson에서 OOM이 난다.
# 학습에는 가장 최근 구간만 쓴다.
DEFAULT_MAX_TRAINING_SAMPLES = 20_000


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
    max_training_samples: int = DEFAULT_MAX_TRAINING_SAMPLES,
) -> ModelArtifact:
    if len(samples) > max_training_samples:
        samples = samples[-max_training_samples:]

    tag_types = detect_tag_types(samples, tags)
    norm_stats = compute_normalization_stats(samples, tags, tag_types)
    tag_types_str = {t: tag_types[t].value for t in tags}
    continuous_indices, binary_indices = type_indices(tags, tag_types_str)

    X, y = build_windows(samples, tags, window_size)
    if X.shape[0] == 0:
        # 윈도우가 0개인 가장 흔한 실제 원인은 샘플 부족이 아니라 config tags 오탈자다
        # (DX1이 한 번도 발행하지 않는 태그 → 모든 샘플이 None → 모든 윈도우가 버려짐).
        never_observed_tags = [
            t for t in tags if not any(s.values.get(t) is not None for s in samples)
        ]
        if never_observed_tags:
            raise ValueError(
                f"tags never observed in calibration data: {never_observed_tags} "
                f"— check config tags against DX1 topic payloads"
            )
        raise ValueError(
            f"not enough calibration samples to build any training window "
            f"(have {len(samples)} samples, need > {window_size})"
        )

    norm_stats_tuples = {t: (s.mean, s.std) for t, s in norm_stats.items()}
    normalize_continuous_columns(X, y, tags, continuous_indices, norm_stats_tuples)

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

    error_stats = _compute_error_stats(
        model, X, y, tags, continuous_indices, binary_indices, batch_size
    )

    return ModelArtifact(
        tags=tags,
        tag_types=tag_types_str,
        norm_stats=norm_stats_tuples,
        error_stats=error_stats,
        window_size=window_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        state_dict=model.state_dict(),
    )


def normalize_continuous_columns(
    X: torch.Tensor,
    y: torch.Tensor,
    tags: tuple[str, ...],
    continuous_indices: list[int],
    norm_stats: dict[str, tuple[float, float]],
) -> None:
    """continuous_indices에 해당하는 열을 (값-평균)/표준편차로 정규화한다(in-place).
    binary 열은 건드리지 않는다. 학습(train_model)과 실시간 추론(inference.py) 양쪽이
    반드시 동일한 정규화를 쓰도록 공유한다."""
    for idx in continuous_indices:
        tag = tags[idx]
        mean, std = norm_stats[tag]
        X[:, :, idx] = (X[:, :, idx] - mean) / std
        y[:, idx] = (y[:, idx] - mean) / std


def _floor_std(mean: float, std: float) -> float:
    """오차 표준편차의 하한. 잘 학습된 모델의 잔차 std는 0.02처럼 아주 작을 수 있는데,
    이상 점수 계산이 이 std로 나누기 때문에 사소한 오차 변동도 z-score 20+로 튄다
    (임계값은 3 근처). 절대 하한 1e-3과 평균 대비 상대 하한 5%를 함께 적용한다."""
    floor = max(1e-3, 0.05 * abs(mean))
    return std if std > floor else floor


def compute_raw_errors(
    model: AnomalyGRU,
    X: torch.Tensor,
    y: torch.Tensor,
    tags: tuple[str, ...],
    continuous_indices: list[int],
    binary_indices: list[int],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, torch.Tensor]:
    """정규화된 X(윈도우 배치)로 다음 시점을 예측하고, 정규화된 실제값 y와의 절대오차를
    태그별로 반환한다(태그당 shape (n,) 텐서, n=배치 크기). 학습 시 "정상 오차" 통계
    산출(_compute_error_stats)과 실시간 추론(inference.py)의 단일 샘플 오차 계산이
    이 함수를 공유한다 — 두 곳에서 예측/오차 수식이 어긋나지 않도록 하기 위함이다.
    전체 X를 한 번에 forward하면 메모리가 무제한으로 커지므로 batch_size 단위로 나눠 돈다."""
    model.eval()
    continuous_chunks: list[torch.Tensor] = []
    binary_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            chunk_continuous, chunk_binary = model(X[start : start + batch_size])
            if chunk_continuous is not None:
                continuous_chunks.append(chunk_continuous)
            if chunk_binary is not None:
                binary_chunks.append(chunk_binary)
    continuous_out = torch.cat(continuous_chunks) if continuous_chunks else None
    binary_logits = torch.cat(binary_chunks) if binary_chunks else None

    errors: dict[str, torch.Tensor] = {}
    for pos, idx in enumerate(continuous_indices):
        errors[tags[idx]] = torch.abs(y[:, idx] - continuous_out[:, pos])
    for pos, idx in enumerate(binary_indices):
        pred_prob = torch.sigmoid(binary_logits[:, pos])
        errors[tags[idx]] = torch.abs(y[:, idx] - pred_prob)
    return errors


def _compute_error_stats(
    model: AnomalyGRU,
    X: torch.Tensor,
    y: torch.Tensor,
    tags: tuple[str, ...],
    continuous_indices: list[int],
    binary_indices: list[int],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, tuple[float, float]]:
    """학습 완료 후 캘리브레이션 데이터 전체에 대한 태그별 "정상 오차" 평균/표준편차.
    실시간 이상 점수 계산(inference.py)에서 원본 오차를 재정규화하는 기준으로 쓰인다."""
    errors = compute_raw_errors(model, X, y, tags, continuous_indices, binary_indices, batch_size)

    result: dict[str, tuple[float, float]] = {}
    for tag, err in errors.items():
        mean = err.mean().item()
        if err.numel() > 1:
            variance = ((err - err.mean()) ** 2).mean().item()
            std = variance ** 0.5
        else:
            std = 0.0
        result[tag] = (mean, _floor_std(mean, std))
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
    max_training_samples: int = DEFAULT_MAX_TRAINING_SAMPLES,
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
            max_training_samples=max_training_samples,
        )
        save_artifact(model_path, artifact)

    return _train_fn


def model_artifact_path(model_dir: str | Path, equipment_id: str) -> Path:
    return Path(model_dir) / f"{equipment_id}.pt"


def state_marker_path(model_dir: str | Path, equipment_id: str) -> Path:
    """캘리브레이션 상태 마커 파일 경로. 모델 아티팩트와 같은 디렉터리 규칙을 쓰므로
    model_artifact_path와 나란히 둔다 (경로 조립 규칙이 여러 곳에 흩어지지 않게)."""
    return Path(model_dir) / f"{equipment_id}.state"
