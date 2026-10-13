from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import yaml

from .debounce import DEFAULT_CONFIRM_TICKS, DEFAULT_THRESHOLD

DEFAULT_MAX_LATENESS_MS = 2000
# 학습 설정 기본값. training.py(torch를 import함)가 아니라 여기에 두어 config/도구가 torch를 끌어오지 않게 한다.
DEFAULT_EPOCHS = 20
# 캘리브레이션 버퍼는 calibration.max_duration까지 자라기 때문에(예: 7d @ 50ms ≈ 12M 샘플)
# 전체를 윈도잉하면 수 GB 텐서가 되어 4GB Jetson에서 OOM이 난다. 학습에는 가장 최근 구간만 쓴다.
DEFAULT_MAX_TRAINING_SAMPLES = 20_000


class ConfigError(ValueError):
    pass


_DURATION_PATTERN = re.compile(r"^(\d+)([smhd])$")
_DURATION_UNIT_KEYWORDS = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
}


def parse_duration(text: str) -> timedelta:
    match = _DURATION_PATTERN.match(text.strip())
    if not match:
        raise ConfigError(
            f"invalid duration '{text}': expected format like '7d', '12h', '30m', '45s'"
        )
    amount = int(match.group(1))
    unit_keyword = _DURATION_UNIT_KEYWORDS[match.group(2)]
    return timedelta(**{unit_keyword: amount})


@dataclass(frozen=True)
class CalibrationConfig:
    max_duration: timedelta
    min_samples: int


@dataclass(frozen=True)
class GroupConfig:
    """태그 묶음. state_tag(동작 중 신호)가 있으면 그 그룹의 태그는 동작/대기 상태별 기준으로 판정한다."""

    name: str
    state_tag: str | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class AlarmConfig:
    threshold: float = DEFAULT_THRESHOLD
    confirm_steps: int = DEFAULT_CONFIRM_TICKS


@dataclass(frozen=True)
class TrainingConfig:
    max_samples: int = DEFAULT_MAX_TRAINING_SAMPLES
    epochs: int = DEFAULT_EPOCHS


DEFAULT_DEVIATION_BASELINE_SECONDS = 2.0
DEFAULT_DEVIATION_SETTLE_SECONDS = 1.0
DEFAULT_DEVIATION_MARGIN = 1.2
DEFAULT_DEVIATION_CONFIRM_STEPS = 1


@dataclass(frozen=True)
class DeviationSignalConfig:
    """기준값 이탈로 점수를 낼 신호 하나. name은 점수/알람의 그룹 이름이 된다.
    move_range가 None이면 학습 데이터에서 자동으로 정한다."""

    name: str
    tag: str
    move_range: float | None = None


@dataclass(frozen=True)
class DeviationConfig:
    """신호가 직전 기준값에서 얼마나 벗어났는지로 점수를 내는 방식(GRU와 별개)의 설정."""

    signals: tuple[DeviationSignalConfig, ...]
    baseline_seconds: float = DEFAULT_DEVIATION_BASELINE_SECONDS
    settle_seconds: float = DEFAULT_DEVIATION_SETTLE_SECONDS
    margin: float = DEFAULT_DEVIATION_MARGIN
    confirm_steps: int = DEFAULT_DEVIATION_CONFIRM_STEPS


@dataclass(frozen=True)
class EquipmentConfig:
    equipment_id: str
    subscribe_topics: tuple[str, ...]
    publish_topic: str
    command_topic: str
    tags: tuple[str, ...]
    resample_interval_ms: int
    window_size: int
    calibration: CalibrationConfig
    max_lateness_ms: int = DEFAULT_MAX_LATENESS_MS
    groups: tuple[GroupConfig, ...] = ()
    alarm: AlarmConfig = AlarmConfig()
    training: TrainingConfig = TrainingConfig()
    # 모델 입력으로만 쓰고 점수에는 반영하지 않는 태그. `tags`는 점수 태그, 점수 밖 상태 태그,
    # 이 태그를 모두 합한 모델 입력 태그다(학습/MQTT 필터/모델 호환성 검사가 이 값을 쓴다).
    context_tags: tuple[str, ...] = ()
    # 모델 입력과 학습에는 그대로 두고, 점수(그룹 최댓값)에서만 뺄 그룹 태그. 모델을 다시 학습할 필요가 없다.
    score_exclude_tags: tuple[str, ...] = ()
    # GRU와 별개로 신호의 기준값 이탈로 점수를 내는 설정. 없으면 None.
    deviation: DeviationConfig | None = None

    @property
    def all_tags(self) -> tuple[str, ...]:
        """구독/리샘플링할 태그 전체: GRU 입력 태그(tags)와 기준값 이탈 신호의 태그."""
        extra = tuple(s.tag for s in self.deviation.signals) if self.deviation else ()
        return self.tags + tuple(t for t in extra if t not in self.tags)

    def resolved_groups(self) -> tuple[GroupConfig, ...]:
        """groups가 있으면 그대로, 평면 tags 설정이면 상태 태그 없는 단일 그룹 'all'.
        GRU를 쓰지 않는 설정(기준값 이탈만)이면 빈 튜플."""
        if self.groups:
            return self.groups
        if not self.tags:
            return ()
        return (GroupConfig(name="all", state_tag=None, tags=self.tags),)

    def group_specs(self) -> dict[str, tuple[str | None, tuple[str, ...]]]:
        """그룹 이름 -> (상태 태그, 태그 목록). 학습 artifact에 저장되어 config와 비교된다."""
        return {g.name: (g.state_tag, g.tags) for g in self.resolved_groups()}


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{label} must be a positive integer")
    return value


def _parse_groups(raw: object) -> tuple[GroupConfig, ...]:
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("groups must be a non-empty mapping")
    groups: list[GroupConfig] = []
    owner: dict[str, str] = {}
    for name, body in raw.items():
        name = str(name)
        if not isinstance(body, dict):
            raise ConfigError(f"group '{name}' must be a mapping")
        tags = body.get("tags")
        if not isinstance(tags, list) or not tags:
            raise ConfigError(f"group '{name}': tags must be a non-empty list")
        state_tag = body.get("state_tag")
        if state_tag is not None and (not isinstance(state_tag, str) or not state_tag):
            raise ConfigError(f"group '{name}': state_tag must be a non-empty string")
        for tag in tags:
            if tag in owner:
                raise ConfigError(
                    f"tag '{tag}' is listed more than once (groups '{owner[tag]}' and '{name}')"
                )
            owner[tag] = name
        groups.append(GroupConfig(name=name, state_tag=state_tag, tags=tuple(tags)))
    return tuple(groups)


def _parse_context_tags(raw: object, groups: tuple[GroupConfig, ...]) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw or not all(isinstance(t, str) and t for t in raw):
        raise ConfigError("context_tags must be a non-empty list of tag names")
    seen: set[str] = set()
    for tag in raw:
        if tag in seen:
            raise ConfigError(f"context_tags: tag '{tag}' is listed more than once")
        seen.add(tag)
    taken = {tag for g in groups for tag in g.tags} | {g.state_tag for g in groups if g.state_tag}
    for tag in raw:
        if tag in taken:
            raise ConfigError(
                f"context_tags: tag '{tag}' is already used by a group (tags or state_tag)"
            )
    return tuple(raw)


def _parse_score_exclude_tags(raw: object, groups: tuple[GroupConfig, ...]) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw or not all(isinstance(t, str) and t for t in raw):
        raise ConfigError("score_exclude_tags must be a non-empty list of tag names")
    if len(set(raw)) != len(raw):
        raise ConfigError("score_exclude_tags: a tag is listed more than once")
    scored = {tag for g in groups for tag in g.tags}
    for tag in raw:
        if tag not in scored:
            raise ConfigError(f"score_exclude_tags: tag '{tag}' is not a scored tag of any group")
    for g in groups:
        if all(tag in raw for tag in g.tags):
            raise ConfigError(f"score_exclude_tags: group '{g.name}' would have no scored tag left")
    return tuple(raw)


def _model_input_tags(groups: tuple[GroupConfig, ...], context_tags: tuple[str, ...]) -> tuple[str, ...]:
    """점수 태그(그룹 순서), 점수 밖 상태 태그, 입력 전용 태그 순서로 중복 없이 합친다."""
    ordered: list[str] = []
    for source in (
        [tag for g in groups for tag in g.tags],
        [g.state_tag for g in groups if g.state_tag],
        list(context_tags),
    ):
        for tag in source:
            if tag not in ordered:
                ordered.append(tag)
    return tuple(ordered)


_DEVIATION_OPTIONS = {"baseline_seconds", "settle_seconds", "margin", "confirm_steps", "signals"}


def _positive_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{label} must be a positive number")
    return float(value)


def _parse_deviation(raw: object, taken_names: set[str]) -> DeviationConfig:
    if not isinstance(raw, dict):
        raise ConfigError("deviation section must be a mapping")
    unknown = set(raw) - _DEVIATION_OPTIONS
    if unknown:
        raise ConfigError(f"deviation: unknown option {sorted(unknown)}")
    if "signals" not in raw:
        raise ConfigError("deviation: missing required field: signals")
    raw_signals = raw["signals"]
    if not isinstance(raw_signals, dict) or not raw_signals:
        raise ConfigError("deviation.signals must be a non-empty mapping (name -> tag)")
    signals: list[DeviationSignalConfig] = []
    seen_tags: set[str] = set()
    for name, body in raw_signals.items():
        name = str(name)
        if name in taken_names:
            raise ConfigError(f"deviation signal '{name}' has the same name as a group already defined")
        move_range = None
        if isinstance(body, dict):
            tag = body.get("tag")
            if "move_range" in body:
                move_range = _positive_number(body["move_range"], f"deviation signal '{name}': move_range")
        else:
            tag = body
        if not isinstance(tag, str) or not tag:
            raise ConfigError(f"deviation signal '{name}': tag must be a non-empty string")
        if tag in seen_tags:
            raise ConfigError(f"deviation: tag '{tag}' is listed more than once")
        seen_tags.add(tag)
        signals.append(DeviationSignalConfig(name=name, tag=tag, move_range=move_range))
    return DeviationConfig(
        signals=tuple(signals),
        baseline_seconds=_positive_number(
            raw.get("baseline_seconds", DEFAULT_DEVIATION_BASELINE_SECONDS), "deviation.baseline_seconds"
        ),
        settle_seconds=_positive_number(
            raw.get("settle_seconds", DEFAULT_DEVIATION_SETTLE_SECONDS), "deviation.settle_seconds"
        ),
        margin=_positive_number(raw.get("margin", DEFAULT_DEVIATION_MARGIN), "deviation.margin"),
        confirm_steps=_positive_int(
            raw.get("confirm_steps", DEFAULT_DEVIATION_CONFIRM_STEPS), "deviation.confirm_steps"
        ),
    )


def _parse_alarm(raw: object) -> AlarmConfig:
    if not isinstance(raw, dict):
        raise ConfigError("alarm section must be a mapping")
    threshold = raw.get("threshold", DEFAULT_THRESHOLD)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold <= 0:
        raise ConfigError("alarm.threshold must be a positive number")
    confirm_steps = _positive_int(raw.get("confirm_steps", DEFAULT_CONFIRM_TICKS), "alarm.confirm_steps")
    return AlarmConfig(threshold=float(threshold), confirm_steps=confirm_steps)


def _parse_training(raw: object) -> TrainingConfig:
    if not isinstance(raw, dict):
        raise ConfigError("training section must be a mapping")
    max_samples = _positive_int(raw.get("max_samples", DEFAULT_MAX_TRAINING_SAMPLES), "training.max_samples")
    epochs = _positive_int(raw.get("epochs", DEFAULT_EPOCHS), "training.epochs")
    return TrainingConfig(max_samples=max_samples, epochs=epochs)


def load_equipment_config(path: str | Path) -> EquipmentConfig:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read config file {path}: {e}") from e

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError("config file must contain a YAML mapping")

    for field in ("equipment_id", "mqtt", "resample_interval_ms", "window_size", "calibration"):
        if field not in data:
            raise ConfigError(f"missing required field: {field}")

    mqtt_section = data["mqtt"]
    if not isinstance(mqtt_section, dict):
        raise ConfigError("mqtt section must be a mapping")

    for field in ("subscribe_topics", "publish_topic", "command_topic"):
        if field not in mqtt_section:
            raise ConfigError(f"missing required field: mqtt.{field}")

    subscribe_topics = mqtt_section["subscribe_topics"]
    if not isinstance(subscribe_topics, list) or not subscribe_topics:
        raise ConfigError("mqtt.subscribe_topics must be a non-empty list")

    if "tags" in data and "groups" in data:
        raise ConfigError("use either 'tags' or 'groups', not both")
    if "tags" not in data and "groups" not in data and "deviation" not in data:
        raise ConfigError("missing required field: tags (or groups, or deviation)")
    if "context_tags" in data and "groups" not in data:
        raise ConfigError("context_tags can only be used together with 'groups'")
    if "score_exclude_tags" in data and "groups" not in data:
        raise ConfigError("score_exclude_tags can only be used together with 'groups'")
    score_exclude_tags: tuple[str, ...] = ()
    if "groups" in data:
        groups = _parse_groups(data["groups"])
        if "score_exclude_tags" in data:
            score_exclude_tags = _parse_score_exclude_tags(data["score_exclude_tags"], groups)
        context_tags = (
            _parse_context_tags(data["context_tags"], groups) if "context_tags" in data else ()
        )
        tags = _model_input_tags(groups, context_tags)
    elif "tags" in data:
        groups = ()
        context_tags = ()
        tags = data["tags"]
        if not isinstance(tags, list) or not tags:
            raise ConfigError("tags must be a non-empty list")
    else:  # 기준값 이탈(deviation)만 쓰는 설정: GRU 입력 태그가 없다
        groups = ()
        context_tags = ()
        tags = ()
    taken_names = {g.name for g in groups} or ({"all"} if tags else set())
    deviation = _parse_deviation(data["deviation"], taken_names) if "deviation" in data else None

    resample_interval_ms = data["resample_interval_ms"]
    if not isinstance(resample_interval_ms, int) or isinstance(resample_interval_ms, bool) or resample_interval_ms <= 0:
        raise ConfigError("resample_interval_ms must be a positive integer")

    window_size = data["window_size"]
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
        raise ConfigError("window_size must be a positive integer")

    max_lateness_ms = data.get("max_lateness_ms", DEFAULT_MAX_LATENESS_MS)
    if not isinstance(max_lateness_ms, int) or isinstance(max_lateness_ms, bool) or max_lateness_ms < 0:
        raise ConfigError("max_lateness_ms must be a non-negative integer")

    alarm = _parse_alarm(data["alarm"]) if "alarm" in data else AlarmConfig()
    training = _parse_training(data["training"]) if "training" in data else TrainingConfig()

    calibration_section = data["calibration"]
    if not isinstance(calibration_section, dict):
        raise ConfigError("calibration section must be a mapping")

    for field in ("max_duration", "min_samples"):
        if field not in calibration_section:
            raise ConfigError(f"missing required field: calibration.{field}")

    max_duration = parse_duration(str(calibration_section["max_duration"]))

    min_samples = calibration_section["min_samples"]
    if not isinstance(min_samples, int) or isinstance(min_samples, bool) or min_samples <= 0:
        raise ConfigError("calibration.min_samples must be a positive integer")

    return EquipmentConfig(
        equipment_id=data["equipment_id"],
        subscribe_topics=tuple(subscribe_topics),
        publish_topic=mqtt_section["publish_topic"],
        command_topic=mqtt_section["command_topic"],
        tags=tuple(tags),
        resample_interval_ms=resample_interval_ms,
        window_size=window_size,
        calibration=CalibrationConfig(max_duration=max_duration, min_samples=min_samples),
        max_lateness_ms=max_lateness_ms,
        groups=groups,
        alarm=alarm,
        training=training,
        context_tags=context_tags,
        score_exclude_tags=score_exclude_tags,
        deviation=deviation,
    )
