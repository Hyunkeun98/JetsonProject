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


def test_max_lateness_ms_defaults_to_2000_when_omitted(tmp_path):
    config_path = tmp_path / "test_dx1.yaml"
    config_path.write_text(SAMPLE_YAML, encoding="utf-8")

    assert load_equipment_config(config_path).max_lateness_ms == 2000


def test_max_lateness_ms_is_read_from_config(tmp_path):
    config_path = tmp_path / "test_dx1.yaml"
    config_path.write_text(SAMPLE_YAML + "max_lateness_ms: 500\n", encoding="utf-8")

    assert load_equipment_config(config_path).max_lateness_ms == 500


def test_max_lateness_ms_zero_is_allowed(tmp_path):
    config_path = tmp_path / "test_dx1.yaml"
    config_path.write_text(SAMPLE_YAML + "max_lateness_ms: 0\n", encoding="utf-8")

    assert load_equipment_config(config_path).max_lateness_ms == 0


@pytest.mark.parametrize("bad_value", ["-1", "1.5", '"2s"', "true", "null"])
def test_max_lateness_ms_rejects_invalid_values(tmp_path, bad_value):
    config_path = tmp_path / "test_dx1.yaml"
    config_path.write_text(SAMPLE_YAML + "max_lateness_ms: %s\n" % bad_value, encoding="utf-8")

    with pytest.raises(ConfigError, match="max_lateness_ms"):
        load_equipment_config(config_path)


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


GROUPS_YAML = """\
equipment_id: "nx5"
mqtt:
  subscribe_topics: ["dx1/AxisData", "dx1/ProcStart"]
  publish_topic: "jetson/nx5/anomaly"
  command_topic: "jetson/nx5/cmd"
groups:
  process_0:
    state_tag: "P:U0_ProcStart"
    tags: ["P:U0_ProcStart", "A:AxX_Act_Pos"]
  process_4:
    state_tag: "P:U4_ProcStart"
    tags: ["P:U4_ProcStart", "A:AxZ_Act_Pos", "T:U4_TaktTime"]
  general:
    tags: ["S:ConvSensor0"]
resample_interval_ms: 100
window_size: 30
calibration:
  max_duration: "7d"
  min_samples: 3000
"""


def _load_text(tmp_path, text):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(text, encoding="utf-8")
    return load_equipment_config(config_path)


def _groups_yaml_with(replacement_groups_block):
    head, _, tail = GROUPS_YAML.partition("groups:\n")
    _, _, rest = tail.partition("resample_interval_ms")
    return head + replacement_groups_block + "resample_interval_ms" + rest


def test_groups_are_parsed_in_declaration_order_and_tags_are_concatenated(tmp_path):
    config = _load_text(tmp_path, GROUPS_YAML)

    assert [g.name for g in config.groups] == ["process_0", "process_4", "general"]
    assert config.groups[0].state_tag == "P:U0_ProcStart"
    assert config.groups[0].tags == ("P:U0_ProcStart", "A:AxX_Act_Pos")
    assert config.groups[2].state_tag is None
    assert config.tags == (
        "P:U0_ProcStart",
        "A:AxX_Act_Pos",
        "P:U4_ProcStart",
        "A:AxZ_Act_Pos",
        "T:U4_TaktTime",
        "S:ConvSensor0",
    )
    assert config.resolved_groups() == config.groups


def test_flat_tags_config_resolves_to_a_single_all_group(tmp_path):
    config = _load_text(tmp_path, SAMPLE_YAML)

    assert config.groups == ()
    resolved = config.resolved_groups()
    assert len(resolved) == 1
    assert (resolved[0].name, resolved[0].state_tag, resolved[0].tags) == ("all", None, config.tags)


def test_tags_and_groups_together_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="tags.*groups|groups.*tags"):
        _load_text(tmp_path, GROUPS_YAML + 'tags: ["x"]\n')


def test_neither_tags_nor_groups_is_rejected(tmp_path):
    text = GROUPS_YAML.replace("groups:", "ignored_key:")
    with pytest.raises(ConfigError, match="tags.*groups|groups.*tags"):
        _load_text(tmp_path, text)


@pytest.mark.parametrize("block", ["groups: []\n", "groups: {}\n", "groups: 3\n"])
def test_groups_must_be_a_non_empty_mapping(tmp_path, block):
    with pytest.raises(ConfigError, match="groups"):
        _load_text(tmp_path, _groups_yaml_with(block))


def test_group_with_empty_or_missing_tags_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="process_0"):
        _load_text(tmp_path, _groups_yaml_with("groups:\n  process_0:\n    tags: []\n"))
    with pytest.raises(ConfigError, match="process_0"):
        _load_text(tmp_path, _groups_yaml_with("groups:\n  process_0:\n    state_tag: x\n"))


def test_tag_in_two_groups_is_rejected(tmp_path):
    block = 'groups:\n  a:\n    tags: ["x", "y"]\n  b:\n    tags: ["y"]\n'
    with pytest.raises(ConfigError, match="y"):
        _load_text(tmp_path, _groups_yaml_with(block))


def test_state_tag_must_be_one_of_the_group_tags(tmp_path):
    block = 'groups:\n  a:\n    state_tag: "z"\n    tags: ["x", "y"]\n'
    with pytest.raises(ConfigError, match="state_tag"):
        _load_text(tmp_path, _groups_yaml_with(block))


def test_alarm_and_training_defaults(tmp_path):
    config = _load_text(tmp_path, SAMPLE_YAML)

    assert config.alarm.threshold == 3.0
    assert config.alarm.confirm_steps == 3
    assert config.training.max_samples == 20_000
    assert config.training.epochs == 20


def test_alarm_and_training_values_are_read(tmp_path):
    text = SAMPLE_YAML + "alarm: {threshold: 4.5, confirm_steps: 5}\ntraining: {max_samples: 60000, epochs: 8}\n"
    config = _load_text(tmp_path, text)

    assert (config.alarm.threshold, config.alarm.confirm_steps) == (4.5, 5)
    assert (config.training.max_samples, config.training.epochs) == (60000, 8)


@pytest.mark.parametrize(
    "extra,needle",
    [
        ("alarm: {threshold: 0}", "alarm.threshold"),
        ("alarm: {threshold: -1}", "alarm.threshold"),
        ('alarm: {threshold: "x"}', "alarm.threshold"),
        ("alarm: {threshold: true}", "alarm.threshold"),
        ("alarm: {confirm_steps: 0}", "alarm.confirm_steps"),
        ("alarm: {confirm_steps: 1.5}", "alarm.confirm_steps"),
        ("alarm: 3", "alarm"),
        ("training: {max_samples: 0}", "training.max_samples"),
        ("training: {epochs: false}", "training.epochs"),
        ("training: {epochs: -2}", "training.epochs"),
        ("training: []", "training"),
    ],
)
def test_invalid_alarm_or_training_values_are_rejected(tmp_path, extra, needle):
    with pytest.raises(ConfigError, match=needle):
        _load_text(tmp_path, SAMPLE_YAML + extra + "\n")
