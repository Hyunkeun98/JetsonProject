from __future__ import annotations

from pathlib import Path

from .buffer import SlidingWindow
from .calibration import CalibrationSample, CalibrationState, StateStore
from .config import EquipmentConfig
from .inference import InferenceEngine
from .mqtt_subscriber import Record, parse_and_filter_records
from .resampler import EventTimeResampler, ResampledStep
from .scores import ScoreRow
from .segments import Segment, SegmentSet
from .timeparse import format_epoch_ns
from .training import (
    ModelArtifact,
    load_artifact,
    make_train_fn,
    model_artifact_path,
    state_marker_path,
)


class ReplayError(ValueError):
    pass


def load_records(
    data_dir: str | Path, tags: tuple[str, ...], segments: tuple[Segment, ...]
) -> list[Record]:
    """데이터 폴더의 *.json/*.jsonl을 읽어 구간 안의 레코드만 이벤트 시각 순으로 돌려준다.
    파일 한 줄이 DX1 메시지 하나이며, 토픽은 보지 않고 설정의 태그만 고른다."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise ReplayError(f"데이터 폴더를 찾을 수 없습니다: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix in (".json", ".jsonl"))
    if not files:
        raise ReplayError(f"데이터 폴더에 .json/.jsonl 파일이 없습니다: {directory}")
    ranges = [(s.start_ns, s.end_ns) for s in segments]
    records: list[Record] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for record in parse_and_filter_records(line.encode("utf-8"), tags):
                    if any(start <= record.epoch_ns < end for start, end in ranges):
                        records.append(record)
    records.sort(key=lambda r: r.epoch_ns)
    return records


def steps_for_segment(
    config: EquipmentConfig, records: list[Record], segment: Segment
) -> list[ResampledStep]:
    """구간 하나를 새 리샘플러에 통과시켜 확정된 스텝을 얻는다(구간 사이는 이어 붙이지 않는다)."""
    resampler = EventTimeResampler(
        tags=config.tags,
        interval_ms=config.resample_interval_ms,
        max_lateness_ms=config.max_lateness_ms,
        window_size=config.window_size,
    )
    steps: list[ResampledStep] = []
    for record in records:
        if segment.start_ns <= record.epoch_ns < segment.end_ns:
            steps.extend(resampler.add(record))
    return steps


def train_from_files(
    config: EquipmentConfig,
    data_dir: str | Path,
    segment_set: SegmentSet,
    model_dir: str | Path,
) -> tuple[Path, int]:
    """학습 구간의 스텝으로 모델을 학습해 저장하고 (모델 경로, 학습 스텝 수)를 돌려준다.
    앱이 같은 --model-dir로 바로 이어 쓰도록 상태 마커도 MONITORING으로 쓴다."""
    if not segment_set.train:
        raise ReplayError("구간 파일에 train 구간이 없습니다")
    records = load_records(data_dir, config.tags, segment_set.train)
    samples: list[CalibrationSample] = []
    for segment in segment_set.train:
        for step in steps_for_segment(config, records, segment):
            samples.append(
                CalibrationSample(
                    timestamp=format_epoch_ns(step.epoch_ns), values=step.snapshot.values
                )
            )
    if not samples:
        raise ReplayError("학습 구간에서 만들어진 스텝이 없습니다. 구간 시각과 데이터 파일을 확인하세요")
    model_path = model_artifact_path(model_dir, config.equipment_id)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    make_train_fn(
        tags=config.tags,
        window_size=config.window_size,
        model_path=model_path,
        resample_interval_ms=config.resample_interval_ms,
        groups=config.group_specs(),
        epochs=config.training.epochs,
        max_training_samples=config.training.max_samples,
    )(samples)
    StateStore(state_marker_path(model_dir, config.equipment_id)).write(CalibrationState.MONITORING)
    return model_path, len(samples)


def load_matching_artifact(config: EquipmentConfig, model_dir: str | Path) -> ModelArtifact:
    path = model_artifact_path(model_dir, config.equipment_id)
    if not path.exists():
        raise ReplayError(f"모델 파일이 없습니다: {path} (먼저 jetson-replay train을 실행하세요)")
    artifact = load_artifact(path)
    if (
        artifact.window_size != config.window_size
        or set(artifact.tags) != set(config.tags)
        or artifact.resample_interval_ms != config.resample_interval_ms
        or artifact.groups != config.group_specs()
    ):
        raise ReplayError(
            "모델이 현재 설정과 맞지 않습니다(window_size/tags/격자 간격/그룹). 설정을 바꿨다면 다시 학습하세요"
        )
    return artifact


def score_steps(
    engine: InferenceEngine,
    steps: list[ResampledStep],
    window_size: int,
    segment: Segment,
) -> list[ScoreRow]:
    """앱 처리기와 같은 순서(윈도우로 예측 → 점수 → 스냅샷 push)로 스텝마다 점수를 낸다."""
    window = SlidingWindow(window_size)
    rows: list[ScoreRow] = []
    pending_reset = True  # 구간의 첫 점수는 새 윈도우에서 시작한다
    for step in steps:
        if step.reset_window:
            window.clear()
            pending_reset = True
        if window.is_full():
            result = engine.score(window.to_list(), step.snapshot)
            if result is not None:
                groups = {
                    name: (group.score, group.top_tag)
                    for name, group in (result.group_results or {}).items()
                }
                rows.append(
                    ScoreRow(
                        epoch_ns=step.epoch_ns,
                        kind=segment.kind,
                        segment=segment.name,
                        reset=pending_reset,
                        groups=groups,
                    )
                )
                pending_reset = False
        window.push(step.snapshot)
    return rows


def score_from_files(
    config: EquipmentConfig,
    data_dir: str | Path,
    segment_set: SegmentSet,
    model_dir: str | Path,
) -> tuple[list[ScoreRow], tuple[str, ...]]:
    eval_segments = segment_set.eval_segments()
    if not eval_segments:
        raise ReplayError("구간 파일에 normal_eval 또는 anomaly 구간이 없습니다")
    artifact = load_matching_artifact(config, model_dir)
    engine = InferenceEngine(artifact)
    records = load_records(data_dir, config.tags, eval_segments)
    rows: list[ScoreRow] = []
    for segment in eval_segments:
        steps = steps_for_segment(config, records, segment)
        segment_rows = score_steps(engine, steps, config.window_size, segment)
        if not segment_rows:
            raise ReplayError(
                f"{segment.kind}({segment.name}) 구간에서 점수가 하나도 나오지 않았습니다. "
                "구간이 윈도우 길이보다 짧거나 데이터가 없습니다"
            )
        rows.extend(segment_rows)
    group_names = tuple(artifact.groups) if artifact.groups else ("all",)
    return rows, group_names
