"""기준값 이탈 점수: 신호가 직전 몇 초의 중앙값에서 얼마나 벗어났는지로 이상을 본다.

GRU(다음 값 예측)와 별개의 방식이다. 멈춰 있는 축의 위치(Pos)처럼 평소엔 거의 일정하고, 외력이
가해지면 잠깐 크게 벗어나는 신호에 맞다. torch를 쓰지 않아 가볍다.

- 점수 z = |값 - 직전 baseline 구간 중앙값| / sigma (sigma는 정상 데이터의 대기 잡음).
- 축이 실제로 움직이는 동안(최근 구간의 값 범위가 move_range를 넘음)은 값이 크게 바뀌는 게
  정상이므로 점수를 내지 않는다(0). 움직임이 막 시작되는 샘플은 충격과 구별되지 않아서
  판정을 settle 구간만큼 늦추고, 그 안에 움직임이 이어지면 움직임으로 본다.
- 학습은 정상 데이터에서 sigma, move_range, 알람 기준(limit)을 자동으로 정한다.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median, pstdev

from .calibration import CalibrationSample, TrainFn
from .config import DeviationConfig, EquipmentConfig

_MOTION_WINDOW_SECONDS = 3.0   # 이 시간 동안의 값 범위로 "움직이는 중"을 판단한다
_MAD_TO_SIGMA = 1.4826         # 중앙값 절대편차 -> 표준편차 환산(정규분포 기준)
_AUTO_MOVE_RANGE_FRACTION = 1e-4   # 자동 움직임 기준: 값 전체 범위의 1e-4 ...
_AUTO_MOVE_RANGE_NOISE_TIMES = 100  # ... 와 대기 잡음의 100배 중 큰 쪽
_MIN_SIGMA = 1e-12
_MIN_FIT_STEPS = 100


def derive_steps(interval_ms: int, baseline_seconds: float, settle_seconds: float) -> dict[str, int]:
    """초 단위 설정을 격자 스텝 수로 바꾼다(각각 최소 1스텝)."""

    def steps(seconds: float) -> int:
        return max(1, round(seconds * 1000 / interval_ms))

    baseline = steps(baseline_seconds)
    return {
        "baseline_steps": baseline,
        "settle_steps": steps(settle_seconds),
        "motion_window_steps": steps(_MOTION_WINDOW_SECONDS),
        "hold_steps": baseline,
    }


class DeviationTracker:
    """신호 하나의 스트리밍 계산. push()는 settle_steps만큼 늦게, 그 시점 샘플의
    (|기준값 이탈|, 움직임 여부)를 돌려준다. 아직 판정할 수 없으면 None."""

    def __init__(
        self,
        baseline_steps: int,
        settle_steps: int,
        motion_window_steps: int,
        hold_steps: int,
        move_range: float,
    ) -> None:
        self._baseline = baseline_steps
        self._settle = settle_steps
        self._motion_window = motion_window_steps
        self._hold = hold_steps
        self._move_range = move_range
        self._values: deque[float] = deque(maxlen=max(baseline_steps, motion_window_steps))
        self._entries: deque[tuple[float | None, bool]] = deque(maxlen=settle_steps + 1)
        self._since_move = hold_steps

    def reset(self) -> None:
        self._values.clear()
        self._entries.clear()
        self._since_move = self._hold

    def push(self, value: float) -> tuple[float, bool] | None:
        recent = list(self._values)
        deviation = value - median(recent[-self._baseline :]) if len(recent) >= self._baseline else None
        self._values.append(value)
        window = list(self._values)[-self._motion_window :]
        moving_now = max(window) - min(window) > self._move_range
        self._since_move = 0 if moving_now else self._since_move + 1
        # 움직임이 끝난 뒤에도 baseline 구간이 움직임 값으로 채워져 있는 동안은 움직임으로 본다.
        held = self._since_move < self._hold
        self._entries.append((deviation, held))
        if len(self._entries) < self._settle + 1:
            return None
        oldest_deviation, oldest_held = self._entries[0]
        if oldest_deviation is None:
            return None
        return abs(oldest_deviation), oldest_held or held


@dataclass(frozen=True)
class DeviationSignal:
    tag: str
    sigma: float       # 정상 데이터의 대기 잡음(이탈량 단위)
    limit: float       # 알람 기준: z >= limit (정상 최대 z x margin)
    move_range: float  # 이 값보다 크게 움직이는 동안은 움직이는 중으로 보고 점수를 내지 않는다


@dataclass(frozen=True)
class DeviationArtifact:
    resample_interval_ms: int
    baseline_steps: int
    settle_steps: int
    motion_window_steps: int
    hold_steps: int
    signals: dict  # 그룹 이름 -> DeviationSignal


@dataclass(frozen=True)
class DeviationScore:
    score: float    # z / limit x 알람 임계값: 알람 임계값에 닿으면 limit에 닿은 것
    top_tag: str
    z: float


def _new_tracker(artifact_like, move_range: float) -> DeviationTracker:
    return DeviationTracker(
        baseline_steps=artifact_like["baseline_steps"],
        settle_steps=artifact_like["settle_steps"],
        motion_window_steps=artifact_like["motion_window_steps"],
        hold_steps=artifact_like["hold_steps"],
        move_range=move_range,
    )


def _run_series(series: list, steps: dict[str, int], move_range: float) -> list[tuple[float, bool]]:
    tracker = _new_tracker(steps, move_range)
    out: list[tuple[float, bool]] = []
    for value in series:
        if value is None:
            tracker.reset()
            continue
        result = tracker.push(float(value))
        if result is not None:
            out.append(result)
    return out


def fit_deviation(
    samples: list[CalibrationSample], config: DeviationConfig, interval_ms: int
) -> DeviationArtifact:
    """정상 데이터로 신호마다 sigma, move_range, limit를 정한다."""
    steps = derive_steps(interval_ms, config.baseline_seconds, config.settle_seconds)
    needed = max(_MIN_FIT_STEPS, 4 * (steps["baseline_steps"] + steps["settle_steps"]))
    if len(samples) < needed:
        raise ValueError(f"기준값 이탈 학습에는 샘플이 최소 {needed}개 필요합니다(현재 {len(samples)}개)")
    signals: dict[str, DeviationSignal] = {}
    for spec in config.signals:
        series = [s.values.get(spec.tag) for s in samples]
        observed = [float(v) for v in series if v is not None]
        if not observed:
            raise ValueError(f"학습 데이터에서 태그 '{spec.tag}' 값을 한 번도 받지 못했습니다(설정의 태그 이름을 확인하세요)")
        everything = _run_series(series, steps, move_range=float("inf"))
        if not everything:
            raise ValueError(f"태그 '{spec.tag}'의 연속 구간이 너무 짧아 기준을 정할 수 없습니다")
        noise_all = _MAD_TO_SIGMA * median(d for d, _ in everything)
        if spec.move_range is not None:
            move_range = spec.move_range
        else:
            move_range = max(
                _AUTO_MOVE_RANGE_FRACTION * (max(observed) - min(observed)),
                _AUTO_MOVE_RANGE_NOISE_TIMES * noise_all,
            )
        idle = [d for d, moving in _run_series(series, steps, move_range) if not moving]
        if not idle:
            raise ValueError(f"태그 '{spec.tag}'는 학습 데이터 내내 움직이는 중이어서 대기 잡음을 알 수 없습니다")
        sigma = _MAD_TO_SIGMA * median(idle)
        if sigma <= 0.0:
            # 값이 계단처럼 가끔만 바뀌어 절반 넘게 이탈이 정확히 0이면 중앙값 기준 잡음이 0이 된다.
            sigma = pstdev(idle) if len(idle) > 1 else 0.0
        if sigma <= 0.0:
            raise ValueError(f"태그 '{spec.tag}'는 대기 중 값이 전혀 변하지 않아 잡음 크기를 알 수 없습니다")
        sigma = max(sigma, _MIN_SIGMA)
        limit = config.margin * max(idle) / sigma
        signals[spec.name] = DeviationSignal(tag=spec.tag, sigma=sigma, limit=limit, move_range=move_range)
    return DeviationArtifact(
        resample_interval_ms=interval_ms,
        signals=signals,
        **steps,
    )


class DeviationEngine:
    """학습된 DeviationArtifact로 스텝마다 신호별 점수를 낸다. 점수는 z를 알람 임계값 기준으로
    환산해(z가 limit에 닿으면 임계값) 기존 그룹 점수와 같은 임계선을 쓸 수 있게 한다."""

    def __init__(self, artifact: DeviationArtifact, alarm_threshold: float) -> None:
        self._signals = artifact.signals
        self._alarm_threshold = alarm_threshold
        steps = {
            "baseline_steps": artifact.baseline_steps,
            "settle_steps": artifact.settle_steps,
            "motion_window_steps": artifact.motion_window_steps,
            "hold_steps": artifact.hold_steps,
        }
        self._trackers = {
            name: _new_tracker(steps, signal.move_range) for name, signal in artifact.signals.items()
        }

    def reset(self) -> None:
        for tracker in self._trackers.values():
            tracker.reset()

    def update(self, values: dict) -> dict[str, DeviationScore] | None:
        """이 스텝의 값을 반영하고, 판정이 나온 신호의 점수를 돌려준다(없으면 None).
        판정은 settle 시간만큼 늦다: 돌려주는 점수는 그만큼 이전 샘플의 것이다."""
        out: dict[str, DeviationScore] = {}
        for name, signal in self._signals.items():
            value = values.get(signal.tag)
            tracker = self._trackers[name]
            if value is None:
                tracker.reset()
                continue
            result = tracker.push(float(value))
            if result is None:
                continue
            deviation, moving = result
            z = 0.0 if moving else deviation / signal.sigma
            out[name] = DeviationScore(
                score=z / signal.limit * self._alarm_threshold, top_tag=signal.tag, z=z
            )
        return out or None


def save_deviation_artifact(path: str | Path, artifact: DeviationArtifact) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "resample_interval_ms": artifact.resample_interval_ms,
        "baseline_steps": artifact.baseline_steps,
        "settle_steps": artifact.settle_steps,
        "motion_window_steps": artifact.motion_window_steps,
        "hold_steps": artifact.hold_steps,
        "signals": {
            name: {"tag": s.tag, "sigma": s.sigma, "limit": s.limit, "move_range": s.move_range}
            for name, s in artifact.signals.items()
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_deviation_artifact(path: str | Path) -> DeviationArtifact:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DeviationArtifact(
        resample_interval_ms=data["resample_interval_ms"],
        baseline_steps=data["baseline_steps"],
        settle_steps=data["settle_steps"],
        motion_window_steps=data["motion_window_steps"],
        hold_steps=data["hold_steps"],
        signals={
            name: DeviationSignal(
                tag=s["tag"], sigma=s["sigma"], limit=s["limit"], move_range=s["move_range"]
            )
            for name, s in data["signals"].items()
        },
    )


def deviation_artifact_path(model_dir: str | Path, equipment_id: str) -> Path:
    return Path(model_dir) / f"{equipment_id}.deviation.json"


def check_deviation_artifact(artifact: DeviationArtifact, config: EquipmentConfig) -> None:
    """저장된 기준값 이탈 학습 결과가 현재 설정과 맞는지 검사한다. 맞지 않으면 ValueError.
    설정이 학습 이후 바뀌면 점수의 의미가 달라지므로 GRU 모델과 같이 재학습을 유도한다."""
    deviation = config.deviation
    if deviation is None:
        raise ValueError("설정에 deviation 섹션이 없습니다")
    expected = derive_steps(config.resample_interval_ms, deviation.baseline_seconds, deviation.settle_seconds)
    actual = {
        "baseline_steps": artifact.baseline_steps,
        "settle_steps": artifact.settle_steps,
        "motion_window_steps": artifact.motion_window_steps,
        "hold_steps": artifact.hold_steps,
    }
    if artifact.resample_interval_ms != config.resample_interval_ms or actual != expected:
        raise ValueError(
            f"deviation 학습 결과가 현재 설정과 맞지 않습니다(격자 간격/baseline_seconds/settle_seconds): "
            f"{artifact.resample_interval_ms}ms {actual} vs {config.resample_interval_ms}ms {expected}"
        )
    saved = {name: signal.tag for name, signal in artifact.signals.items()}
    wanted = {spec.name: spec.tag for spec in deviation.signals}
    if saved != wanted:
        raise ValueError(f"deviation 신호 구성이 학습 때와 다릅니다: {saved} vs {wanted}")
    for spec in deviation.signals:
        if spec.move_range is not None and artifact.signals[spec.name].move_range != spec.move_range:
            raise ValueError(f"deviation 신호 '{spec.name}'의 move_range가 학습 때와 다릅니다")


def make_deviation_train_fn(config: EquipmentConfig, path: str | Path) -> TrainFn:
    """캘리브레이션 샘플로 기준값 이탈 기준을 학습해 path에 저장하는 학습 함수."""
    if config.deviation is None:
        raise ValueError("설정에 deviation 섹션이 없습니다")

    def _train_fn(samples: list[CalibrationSample]) -> None:
        artifact = fit_deviation(samples, config.deviation, config.resample_interval_ms)
        save_deviation_artifact(path, artifact)
        for name, signal in artifact.signals.items():
            print(
                f"[train] deviation {name}: 대기 잡음 {signal.sigma:.3g}, 알람 기준 z={signal.limit:.1f}, "
                f"움직임 기준 {signal.move_range:.3g}"
            )

    return _train_fn
