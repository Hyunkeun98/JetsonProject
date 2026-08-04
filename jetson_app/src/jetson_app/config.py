from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import yaml


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
class EquipmentConfig:
    equipment_id: str
    subscribe_topics: tuple[str, ...]
    publish_topic: str
    command_topic: str
    tags: tuple[str, ...]
    resample_interval_ms: int
    window_size: int
    calibration: CalibrationConfig


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

    for field in ("equipment_id", "mqtt", "tags", "resample_interval_ms", "window_size", "calibration"):
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

    tags = data["tags"]
    if not isinstance(tags, list) or not tags:
        raise ConfigError("tags must be a non-empty list")

    resample_interval_ms = data["resample_interval_ms"]
    if not isinstance(resample_interval_ms, int) or isinstance(resample_interval_ms, bool) or resample_interval_ms <= 0:
        raise ConfigError("resample_interval_ms must be a positive integer")

    window_size = data["window_size"]
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
        raise ConfigError("window_size must be a positive integer")

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
    )
