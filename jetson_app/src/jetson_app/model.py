from __future__ import annotations

import torch
from torch import nn


class AnomalyGRU(nn.Module):
    """설비 통합 1개 GRU. 연속 태그는 회귀 head(MSE용), 이진 태그는
    시그모이드 이전 logit head(BCEWithLogitsLoss용)로 다음 시점 값을 예측한다."""

    def __init__(
        self,
        num_tags: int,
        continuous_indices: list[int],
        binary_indices: list[int],
        hidden_size: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.continuous_indices = continuous_indices
        self.binary_indices = binary_indices
        self.gru = nn.GRU(
            input_size=num_tags,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.continuous_head = (
            nn.Linear(hidden_size, len(continuous_indices)) if continuous_indices else None
        )
        self.binary_head = (
            nn.Linear(hidden_size, len(binary_indices)) if binary_indices else None
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]  # 마지막 레이어의 마지막 시점 hidden state, shape (batch, hidden_size)
        continuous_out = (
            self.continuous_head(last_hidden) if self.continuous_head is not None else None
        )
        binary_logits = (
            self.binary_head(last_hidden) if self.binary_head is not None else None
        )
        return continuous_out, binary_logits
