from pathlib import Path

import pytest

from jetson_app.config import ConfigError, load_equipment_config


SAMPLE_YAML = """\
equipment_id: "test_dx1"
mqtt:
  subscribe_topic: "dx1/test_dx1/telemetry"
  publish_topic: "jetson/test_dx1/anomaly"
tags:
  - "servo1:torque"
  - "sensor:A_L_01"
"""


def test_load_equipment_config_parses_valid_yaml(tmp_path):
    config_path = tmp_path / "test_dx1.yaml"
    config_path.write_text(SAMPLE_YAML, encoding="utf-8")

    config = load_equipment_config(config_path)

    assert config.equipment_id == "test_dx1"
    assert config.subscribe_topic == "dx1/test_dx1/telemetry"
    assert config.publish_topic == "jetson/test_dx1/anomaly"
    assert config.tags == ("servo1:torque", "sensor:A_L_01")


def test_load_equipment_config_missing_tags_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        'equipment_id: "x"\nmqtt:\n  subscribe_topic: "a"\n  publish_topic: "b"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_missing_mqtt_section_raises(tmp_path):
    config_path = tmp_path / "bad2.yaml"
    config_path.write_text(
        'equipment_id: "x"\ntags:\n  - "a"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_empty_yaml_raises(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_malformed_yaml_raises(tmp_path):
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("{ invalid yaml: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_mqtt_not_mapping_raises(tmp_path):
    config_path = tmp_path / "bad_mqtt.yaml"
    config_path.write_text(
        'equipment_id: "x"\nmqtt: "not a dict"\ntags:\n  - "a"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_missing_file_raises_config_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(ConfigError):
        load_equipment_config(missing_path)


def test_load_equipment_config_loads_shipped_example_config():
    example_path = Path(__file__).resolve().parent.parent / "configs" / "test_dx1.example.yaml"

    config = load_equipment_config(example_path)

    assert config.tags != ()
