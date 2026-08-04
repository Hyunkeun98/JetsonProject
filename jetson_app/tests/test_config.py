import pytest

from jetson_app.config import ConfigError, load_equipment_config, parse_duration


SAMPLE_YAML = """\
equipment_id: "test_dx1"
mqtt:
  subscribe_topics:
    - "dx1/test_dx1/actuator_1"
    - "dx1/test_dx1/axis_status_3"
  publish_topic: "jetson/test_dx1/anomaly"
  command_topic: "jetson/test_dx1/cmd"
tags:
  - "PLC_Collector_Actuator_1:AirBlower.Cmd[0]"
  - "PLC_Collector_Axis_Status_3:AxCV_VelDemVal"
resample_interval_ms: 50
window_size: 100
calibration:
  max_duration: "7d"
  min_samples: 10000
"""


def test_load_equipment_config_parses_valid_yaml(tmp_path):
    config_path = tmp_path / "test_dx1.yaml"
    config_path.write_text(SAMPLE_YAML, encoding="utf-8")

    config = load_equipment_config(config_path)

    assert config.equipment_id == "test_dx1"
    assert config.subscribe_topics == (
        "dx1/test_dx1/actuator_1",
        "dx1/test_dx1/axis_status_3",
    )
    assert config.publish_topic == "jetson/test_dx1/anomaly"
    assert config.command_topic == "jetson/test_dx1/cmd"
    assert config.tags == (
        "PLC_Collector_Actuator_1:AirBlower.Cmd[0]",
        "PLC_Collector_Axis_Status_3:AxCV_VelDemVal",
    )
    assert config.resample_interval_ms == 50
    assert config.window_size == 100
    assert config.calibration.min_samples == 10000
    from datetime import timedelta

    assert config.calibration.max_duration == timedelta(days=7)


def test_load_equipment_config_missing_subscribe_topics_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        'equipment_id: "x"\n'
        'mqtt:\n  publish_topic: "b"\n  command_topic: "c"\n'
        'tags:\n  - "a"\n'
        "resample_interval_ms: 50\nwindow_size: 100\n"
        'calibration:\n  max_duration: "7d"\n  min_samples: 10\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_empty_subscribe_topics_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        'equipment_id: "x"\n'
        'mqtt:\n  subscribe_topics: []\n  publish_topic: "b"\n  command_topic: "c"\n'
        'tags:\n  - "a"\n'
        "resample_interval_ms: 50\nwindow_size: 100\n"
        'calibration:\n  max_duration: "7d"\n  min_samples: 10\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_missing_calibration_section_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        'equipment_id: "x"\n'
        'mqtt:\n  subscribe_topics: ["t"]\n  publish_topic: "b"\n  command_topic: "c"\n'
        'tags:\n  - "a"\n'
        "resample_interval_ms: 50\nwindow_size: 100\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_invalid_min_samples_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        'equipment_id: "x"\n'
        'mqtt:\n  subscribe_topics: ["t"]\n  publish_topic: "b"\n  command_topic: "c"\n'
        'tags:\n  - "a"\n'
        "resample_interval_ms: 50\nwindow_size: 100\n"
        'calibration:\n  max_duration: "7d"\n  min_samples: "not-a-number"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_invalid_duration_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        'equipment_id: "x"\n'
        'mqtt:\n  subscribe_topics: ["t"]\n  publish_topic: "b"\n  command_topic: "c"\n'
        'tags:\n  - "a"\n'
        "resample_interval_ms: 50\nwindow_size: 100\n"
        'calibration:\n  max_duration: "banana"\n  min_samples: 10\n',
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
    config_path.write_text("equipment_id: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_equipment_config(tmp_path / "does-not-exist.yaml")


def test_load_equipment_config_loads_shipped_example_config():
    from pathlib import Path

    example_path = Path(__file__).parent.parent / "configs" / "test_dx1.example.yaml"
    config = load_equipment_config(example_path)

    assert len(config.subscribe_topics) >= 1
    assert len(config.tags) > 0


def test_parse_duration_parses_days():
    from datetime import timedelta

    assert parse_duration("7d") == timedelta(days=7)


def test_parse_duration_parses_hours_minutes_seconds():
    from datetime import timedelta

    assert parse_duration("12h") == timedelta(hours=12)
    assert parse_duration("30m") == timedelta(minutes=30)
    assert parse_duration("45s") == timedelta(seconds=45)


def test_parse_duration_invalid_format_raises():
    with pytest.raises(ConfigError):
        parse_duration("banana")
    with pytest.raises(ConfigError):
        parse_duration("7")
    with pytest.raises(ConfigError):
        parse_duration("7x")
