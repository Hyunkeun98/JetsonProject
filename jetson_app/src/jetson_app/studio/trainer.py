"""5단계 학습: 수집본(원본)에서 고른 태그의 값을 읽어 기준값 이탈 기준을 만든다. 파이프라인도 torch도 쓰지 않는다."""
from __future__ import annotations

from pathlib import Path

from ..artifact_paths import state_marker_path
from ..calibration import CalibrationSample, CalibrationState, StateStore
from ..deviation import deviation_artifact_path, make_deviation_train_fn
from ..replay import load_records, steps_for_segment
from ..segments import Segment
from ..timeparse import format_epoch_ns
from .errors import StudioError

_EVERYTHING = Segment(kind="train", name="all", start_ns=0, end_ns=2**62)


def train_from_collection(config, collection_dir, model_dir) -> int:
    """수집본으로 학습해 모델과 상태 마커(MONITORING)를 쓰고, 학습에 쓴 스텝 수를 돌려준다.
    학습이 실패하면 예외를 그대로 올리고 아무것도 쓰지 않는다."""
    records = load_records(Path(collection_dir), config.all_tags, (_EVERYTHING,))
    samples = []
    for step in steps_for_segment(config, records, _EVERYTHING):
        if step.reset_window and samples:
            samples.append(CalibrationSample(timestamp="", values={}))   # 공백 표시: 끊긴 구간을 이어 붙이지 않는다
        samples.append(CalibrationSample(timestamp=format_epoch_ns(step.epoch_ns), values=step.snapshot.values))
    if not samples:
        raise StudioError(
            "선택한 태그의 값이 수집 데이터에 없습니다: " + ", ".join(config.all_tags) + ". 태그 이름을 확인하세요."
        )
    artifact_path = deviation_artifact_path(model_dir, config.equipment_id)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    make_deviation_train_fn(config, artifact_path)(samples)
    StateStore(state_marker_path(model_dir, config.equipment_id)).write(CalibrationState.MONITORING)
    return len([s for s in samples if s.values])
