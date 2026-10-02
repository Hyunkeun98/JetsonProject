from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch

from .buffer import Snapshot
from .model import AnomalyGRU
from .tag_stats import type_indices
from .training import ModelArtifact, compute_raw_errors, normalize_continuous_columns


@dataclass(frozen=True)
class GroupResult:
    score: float
    top_tag: str


@dataclass(frozen=True)
class AnomalyResult:
    anomaly_score: float
    top_deviant_tag: str
    # 그룹 이름 -> 그룹 점수/원인 태그. 그룹 구성이 없으면 모든 태그를 한 그룹 "all"로 본다.
    group_results: dict = field(default_factory=dict)


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
        self._groups = artifact.groups or {"all": (None, artifact.tags)}
        # 태그 -> 그 태그가 속한 그룹의 상태 태그(동작 중 신호). 상태 태그가 없으면 항목 없음.
        self._state_tag_of = {
            tag: state_tag
            for state_tag, tags in self._groups.values()
            if state_tag is not None
            for tag in tags
        }
        # 점수를 내는 태그(그룹의 tags). 점수 밖 상태 태그와 입력 전용 태그는 예측만 하고 채점하지 않는다.
        self._scored_tags = tuple(tag for _state, tags in self._groups.values() for tag in tags)

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

        actual_values = dict(zip(self._tags, actual_row))
        z_by_tag: dict[str, float] = {}
        for tag in self._scored_tags:
            err_tensor = raw_errors.get(tag)
            if err_tensor is None:
                continue
            error_mean, error_std = self._stats_for(
                tag, actual_values.get(self._state_tag_of.get(tag))
            )
            z_by_tag[tag] = (err_tensor.item() - error_mean) / error_std

        group_results: dict[str, GroupResult] = {}
        for name, (_state_tag, group_tags) in self._groups.items():
            group_best_tag: str | None = None
            group_best_z: float | None = None
            for tag in group_tags:
                z = z_by_tag.get(tag)
                if z is not None and (group_best_z is None or z > group_best_z):
                    group_best_z = z
                    group_best_tag = tag
            if group_best_tag is not None:
                group_results[name] = GroupResult(score=group_best_z, top_tag=group_best_tag)

        if not group_results:
            return None

        # 전체 점수는 점수 대상 그룹의 점수 중 최댓값이다(점수 밖 태그는 영향을 주지 않는다).
        top_group = max(group_results.values(), key=lambda group: group.score)
        return AnomalyResult(
            anomaly_score=top_group.score,
            top_deviant_tag=top_group.top_tag,
            group_results=group_results,
        )

    def _stats_for(self, tag: str, state_value: float | None) -> tuple[float, float]:
        """태그의 정상 오차 (평균, 표준편차). 상태 태그가 있는 그룹의 태그는 현재 상태(동작 중/
        대기 중)에 맞는 통계를 쓰고, 상태를 모르거나 그 상태의 통계가 없으면(표본 부족으로
        학습 때 대체됨) 전체 통계를 쓴다."""
        regime = self._artifact.regime_error_stats.get(tag)
        if regime and state_value is not None:
            stats = regime.get("on" if state_value >= 0.5 else "off")
            if stats is not None:
                return stats[0], stats[1]
        return self._artifact.error_stats[tag]


class ActiveModelHolder:
    """SnapshotProcessor(매 스텝, 읽기)와 학습 완료 콜백(train 명령 시, 쓰기)이 서로
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
