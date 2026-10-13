"""저장해 둔 정상 데이터를 훑어서, 기준값 이탈 점수(deviation)에 맞는 태그를 추천한다.

맞는 신호는 "평소(멈춰 있을 때)의 잡음이 작고 값의 전체 범위는 넓은" 신호다(예: 서보 위치).
그런 신호는 외력이 가해져 조금만 벗어나도 잡음과 구별된다. 속도·토크처럼 평소에도 크게
흔들리는 신호는 맞지 않는다. 이 도구는 태그마다 정상 데이터로 대기 잡음, 움직이는 비율,
값 범위를 재서 그 기준으로 추천하고 이유를 보여 준다. 추천은 어디까지나 참고이며, 실제로
충격을 잡는지는 이상 데이터로 확인해야 한다.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .calibration import CalibrationSample
from .config import DeviationConfig, DeviationSignalConfig
from .deviation import _run_series, derive_steps, fit_deviation
from .replay import ReplayError, load_records
from .resampler import EventTimeResampler
from .segments import Segment

# 범위/잡음이 이 값(10의 거듭제곱 지수) 이상이어야 추천한다. NX5 서보 위치는 10^6.4~10^7.2, 속도는 10^4.1 이하였다.
MIN_DYNAMIC_RANGE = 5.0
# 대기 중 이탈이 정확히 0인 비율이 이보다 크면 값이 계단처럼 가끔만 바뀌는 신호로 본다(카운터, 타이머, 거친 양자화).
MAX_ZERO_FRACTION = 0.5
MAX_MOVING_FRACTION = 0.5
_MAX_DISCRETE_VALUES = 10
_MAX_LATENESS_MS = 500
_ALL_TIME = Segment(kind="train", name="all", start_ns=0, end_ns=2**62)


@dataclass(frozen=True)
class TagReport:
    tag: str
    kind: str                 # continuous / discrete / binary / constant
    steps: int
    value_range: float
    sigma: float | None       # 대기 잡음(기준값 이탈 단위). 연속 신호만
    limit: float | None       # 학습으로 정해지는 알람 기준 z
    moving_fraction: float | None
    dynamic_range: float      # log10(값 범위 / 대기 잡음). 클수록 작은 이탈도 드러난다
    recommended: bool
    reason: str


def _data_files(data_dir: str | Path) -> list[Path]:
    directory = Path(data_dir)
    if not directory.is_dir():
        raise ReplayError(f"데이터 폴더를 찾을 수 없습니다: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix in (".json", ".jsonl"))
    if not files:
        raise ReplayError(f"데이터 폴더에 .json/.jsonl 파일이 없습니다: {directory}")
    return files


def discover_numeric_tags(data_dir: str | Path) -> list[str]:
    """데이터 폴더의 DX1 JSON에서 숫자 값을 가진 태그 이름을 모은다(timestamp, 문자열, True/False 제외)."""
    tags: set[str] = set()
    for path in _data_files(data_dir):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                records = data.get("records", []) if isinstance(data, dict) else []
                for record in records if isinstance(records, list) else []:
                    if not isinstance(record, dict):
                        continue
                    for key, value in record.items():
                        if key == "timestamp" or isinstance(value, bool):
                            continue
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            tags.add(key)
    return sorted(tags)


def _resample(
    data_dir: str | Path, tags: list[str], interval_ms: int, segments: tuple[Segment, ...]
) -> list[CalibrationSample]:
    records = load_records(data_dir, tuple(tags), segments)
    resampler = EventTimeResampler(
        tags=tuple(tags),
        interval_ms=interval_ms,
        max_lateness_ms=_MAX_LATENESS_MS,
        window_size=max(1, round(3000 / interval_ms)),
    )
    steps = []
    for record in records:
        steps.extend(resampler.add(record))
    samples: list[CalibrationSample] = []
    for step in steps:
        if step.reset_window and samples:
            samples.append(CalibrationSample(timestamp="", values={}))   # 공백 표시: 이어 붙이지 않는다
        samples.append(CalibrationSample(timestamp="", values=step.snapshot.values))
    return samples


def analyze_tags(
    data_dir: str | Path,
    interval_ms: int = 100,
    baseline_seconds: float = 2.0,
    settle_seconds: float = 1.0,
    segments: tuple[Segment, ...] | None = None,
) -> list[TagReport]:
    """숫자 태그마다 기준값 이탈 점수에 맞는지 재서 보고서를 만든다(추천 먼저, 범위/잡음 큰 순)."""
    tags = discover_numeric_tags(data_dir)
    samples = _resample(data_dir, tags, interval_ms, segments or (_ALL_TIME,))
    steps = derive_steps(interval_ms, baseline_seconds, settle_seconds)
    reports = [_analyze_one(tag, samples, interval_ms, baseline_seconds, settle_seconds, steps) for tag in tags]
    reports.sort(key=lambda r: (not r.recommended, -r.dynamic_range))
    return reports


def _analyze_one(
    tag: str,
    samples: list[CalibrationSample],
    interval_ms: int,
    baseline_seconds: float,
    settle_seconds: float,
    steps: dict[str, int],
) -> TagReport:
    values = [s.values.get(tag) for s in samples]
    observed = [float(v) for v in values if v is not None]
    if not observed:
        return TagReport(tag, "constant", 0, 0.0, None, None, None, 0.0, False, "값을 받지 못했습니다")
    value_range = max(observed) - min(observed)
    distinct = len(set(observed))

    def rejected(kind: str, reason: str) -> TagReport:
        return TagReport(tag, kind, len(observed), value_range, None, None, None, 0.0, False, reason)

    if distinct <= 1:
        return rejected("constant", "값이 변하지 않습니다")
    if distinct == 2:
        return rejected("binary", "0/1 같은 두 값뿐인 신호라 기준값 이탈로 볼 수 없습니다")
    if distinct <= _MAX_DISCRETE_VALUES:
        return rejected("discrete", "값이 몇 가지뿐인 신호라 기준값 이탈로 볼 수 없습니다")
    spec = DeviationSignalConfig(name="t", tag=tag)
    config = DeviationConfig(signals=(spec,), baseline_seconds=baseline_seconds, settle_seconds=settle_seconds)
    try:
        signal = fit_deviation(samples, config, interval_ms).signals["t"]
    except ValueError as e:
        return rejected("continuous", f"학습 기준을 정하지 못했습니다: {e}")
    finalized = _run_series(values, steps, signal.move_range)
    moving_fraction = sum(1 for _, moving in finalized if moving) / len(finalized)
    idle = [deviation for deviation, moving in finalized if not moving]
    zero_fraction = sum(1 for d in idle if d <= 1e-9 * value_range) / len(idle) if idle else 0.0
    dynamic_range = math.log10(value_range / signal.sigma) if value_range > 0 else 0.0
    if zero_fraction > MAX_ZERO_FRACTION:
        reason = (
            f"대기 중 {zero_fraction:.0%}는 값이 전혀 안 변하고 가끔만 계단처럼 바뀌는 신호(카운터·타이머·거친 양자화)라 "
            "잡음 크기를 믿기 어렵습니다"
        )
        ok = False
    elif dynamic_range < MIN_DYNAMIC_RANGE:
        reason = (
            f"대기 잡음({signal.sigma:.3g})이 값의 전체 범위({value_range:.3g})에 비해 커서 "
            f"작은 이탈을 구분하기 어렵습니다(범위/잡음 약 10^{dynamic_range:.1f}, 10^{MIN_DYNAMIC_RANGE:.0f} 이상을 추천)"
        )
        ok = False
    elif moving_fraction > MAX_MOVING_FRACTION:
        reason = f"데이터의 {moving_fraction:.0%}가 움직이는 구간이라 점수를 내는 시간이 적습니다"
        ok = False
    else:
        reason = f"대기 잡음이 작고(σ={signal.sigma:.3g}) 값의 범위가 넓어(범위/잡음 약 10^{dynamic_range:.1f}) 작은 이탈도 드러납니다"
        ok = True
    return TagReport(
        tag, "continuous", len(observed), value_range, signal.sigma, signal.limit,
        moving_fraction, dynamic_range, ok, reason,
    )


def _signal_name(tag: str, used: set[str]) -> str:
    base = re.sub(r"[^0-9a-z]+", "_", tag.split(":")[-1].lower()).strip("_") or "signal"
    name, n = base, 2
    while name in used:
        name, n = f"{base}_{n}", n + 1
    used.add(name)
    return name


def format_report(reports: list[TagReport]) -> str:
    lines = ["숫자 태그 분석(정상 데이터 기준)", ""]
    lines.append(f"{'태그':44s} {'종류':10s} {'움직임':>6s} {'대기잡음':>10s} {'값 범위':>10s} {'범위/잡음':>9s}  판정")
    for r in reports:
        moving = "-" if r.moving_fraction is None else f"{r.moving_fraction:.0%}"
        sigma = "-" if r.sigma is None else f"{r.sigma:.3g}"
        dyn = "-" if r.sigma is None else f"10^{r.dynamic_range:.1f}"
        verdict = "추천" if r.recommended else "-"
        lines.append(f"{r.tag:44s} {r.kind:10s} {moving:>6s} {sigma:>10s} {r.value_range:>10.3g} {dyn:>9s}  {verdict}")
    lines.append("")
    recommended = [r for r in reports if r.recommended]
    skipped = [r for r in reports if not r.recommended and r.kind in ("continuous", "discrete")]
    if skipped:
        lines.append("추천하지 않은 연속 신호와 이유:")
        lines += [f"  {r.tag}: {r.reason}" for r in skipped]
        lines.append("")
    if not recommended:
        lines.append("추천할 태그가 없습니다. 이 데이터에는 평소 조용하고 값 범위가 넓은 신호(예: 서보 위치)가 보이지 않습니다.")
        lines.append("그래도 쓰고 싶은 태그가 있다면 설정에 직접 적고, 이상 데이터로 잡히는지 확인하세요.")
        return "\n".join(lines)
    lines.append("추천 이유:")
    for r in recommended:
        lines.append(f"  {r.tag}: {r.reason}")
    lines += ["", "설정 조각(설비 config YAML에 붙여 넣으세요. 숫자는 학습할 때 자동으로 정해집니다):", "", "deviation:", "  signals:"]
    used: set[str] = set()
    for r in recommended:
        lines.append(f'    {_signal_name(r.tag, used)}: "{r.tag}"')
    lines += [
        "",
        "참고: 추천은 정상 데이터의 잡음과 범위로 본 것입니다. 실제로 충격을 잡는지는 이상 데이터를 넣어",
        "jetson-replay score로 점수를 내서 확인하세요(README의 '기준값 이탈 점수' 절).",
    ]
    return "\n".join(lines)
