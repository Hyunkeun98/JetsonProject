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

    def resolved_groups(self) -> tuple[GroupConfig, ...]:
        """groups가 있으면 그대로, 평면 tags 설정이면 상태 태그 없는 단일 그룹 'all'."""
        if self.groups:
            return self.groups
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
        if state_tag is not None and (not isinstance(state_tag, str) or state_tag not in tags):
            raise ConfigError(f"group '{name}': state_tag must be one of the group's tags")
        for tag in tags:
            if tag in owner:
                raise ConfigError(
                    f"tag '{tag}' is listed more than once (groups '{owner[tag]}' and '{name}')"
                )
            owner[tag] = name
        groups.append(GroupConfig(name=name, state_tag=state_tag, tags=tuple(tags)))
    return tuple(groups)


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
    if "tags" not in data and "groups" not in data:
        raise ConfigError("missing required field: tags (or groups)")
    if "groups" in data:
        groups = _parse_groups(data["groups"])
        tags = [tag for group in groups for tag in group.tags]
    else:
        groups = ()
        tags = data["tags"]
        if not isinstance(tags, list) or not tags:
            raise ConfigError("tags must be a non-empty list")

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
    )
