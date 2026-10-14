"""학습 결과 파일 경로 규칙(torch 불필요)."""
from __future__ import annotations

from pathlib import Path


def model_artifact_path(model_dir: str | Path, equipment_id: str) -> Path:
    return Path(model_dir) / f"{equipment_id}.pt"


def state_marker_path(model_dir: str | Path, equipment_id: str) -> Path:
    """캘리브레이션 상태 마커 파일 경로. 모델 아티팩트와 같은 디렉터리 규칙을 쓴다."""
    return Path(model_dir) / f"{equipment_id}.state"
