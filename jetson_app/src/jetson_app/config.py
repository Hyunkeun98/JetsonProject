from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EquipmentConfig:
    equipment_id: str
    subscribe_topic: str
    publish_topic: str
    tags: tuple[str, ...]


def load_equipment_config(path: str | Path) -> EquipmentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    for field in ("equipment_id", "mqtt", "tags"):
        if field not in data:
            raise ConfigError(f"missing required field: {field}")

    mqtt_section = data["mqtt"]
    for field in ("subscribe_topic", "publish_topic"):
        if field not in mqtt_section:
            raise ConfigError(f"missing required field: mqtt.{field}")

    tags = data["tags"]
    if not isinstance(tags, list) or not tags:
        raise ConfigError("tags must be a non-empty list")

    return EquipmentConfig(
        equipment_id=data["equipment_id"],
        subscribe_topic=mqtt_section["subscribe_topic"],
        publish_topic=mqtt_section["publish_topic"],
        tags=tuple(tags),
    )
