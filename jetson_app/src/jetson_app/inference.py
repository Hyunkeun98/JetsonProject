from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from .buffer import Snapshot
from .model import AnomalyGRU
from .tag_stats import type_indices
from .training import ModelArtifact, compute_raw_errors, normalize_continuous_columns


@dataclass(frozen=True)
class AnomalyResult:
    anomaly_score: float
    top_deviant_tag: str


class InferenceEngine:
    """저장된 ModelArtifact로 GRU를 복원해, 실시간 윈도우로 다음 시점을 예측하고
    실제값과의 오차를 캘리브레이션 구간 "정상 오차" 통계로 재정규화해 이상 점수를 낸다
    (설계 스펙 2026-08-04 문서 6절)."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self._artifact = artifact
        self._tags = artifact.tags
        self._continuous_indices, self._binary_indices = type_indices(
            artifact.tags, artifact.tag_types
        )
        self._model = AnomalyGRU(
            num_tags=len(artifact.tags),
            continuous_indices=self._continuous_indices,
            binary_indices=self._binary_indices,
            hidden_size=artifact.hidden_size,
            num_layers=artifact.num_layers,
        )
        self._model.load_state_dict(artifact.state_dict)
        self._model.eval()

    def score(self, window: list[Snapshot], actual: Snapshot) -> AnomalyResult | None:
        """window(길이 window_size, 오래된→최신 순)로 다음 시점을 예측하고, 실제로
        도착한 actual과 비교해 이상 점수를 계산한다. window나 actual에 한 번도 관측
        안 된(None) 태그가 있으면 None을 반환해 그 틱의 채점을 건너뛴다(학습 시
        build_windows가 None 포함 윈도우를 버리는 것과 동일한 정책)."""
        if len(window) != self._artifact.window_size:
            return None

        window_rows: list[list[float]] = []
        for snapshot in window:
            row = [snapshot.values.get(tag) for tag in self._tags]
            if any(v is None for v in row):
                return None
            window_rows.append([float(v) for v in row])

        actual_row = [actual.values.get(tag) for tag in self._tags]
        if any(v is None for v in actual_row):
            return None

        X = torch.tensor([window_rows], dtype=torch.float32)
        y = torch.tensor([[float(v) for v in actual_row]], dtype=torch.float32)
        normalize_continuous_columns(
            X, y, self._tags, self._continuous_indices, self._artifact.norm_stats
        )

        raw_errors = compute_raw_errors(
            self._model,
            X,
            y,
            self._tags,
            self._continuous_indices,
            self._binary_indices,
            batch_size=1,
        )

        best_tag: str | None = None
        best_z: float | None = None
        for tag, err_tensor in raw_errors.items():
            raw_error = err_tensor.item()
            error_mean, error_std = self._artifact.error_stats[tag]
            z = (raw_error - error_mean) / error_std
            if best_z is None or z > best_z:
                best_z = z
                best_tag = tag

        return AnomalyResult(anomaly_score=best_z, top_deviant_tag=best_tag)


class ActiveModelHolder:
    """PeriodicSnapshotter(매 틱, 읽기)와 학습 완료 콜백(train 명령 시, 쓰기)이 서로
    다른 스레드에서 접근하는 현재 InferenceEngine을 스레드세이프하게 공유한다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine: InferenceEngine | None = None

    def get(self) -> InferenceEngine | None:
        with self._lock:
            return self._engine

    def set(self, engine: InferenceEngine | None) -> None:
        with self._lock:
            self._engine = engine
