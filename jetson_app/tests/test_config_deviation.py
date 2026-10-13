import pytest

from jetson_app.config import (
    ConfigError,
    DeviationConfig,
    DeviationSignalConfig,
    load_equipment_config,
)

_BASE = """\
equipment_id: "nx5"
mqtt:
  subscribe_topics: ["dx1/AxisData"]
  publish_topic: "jetson/nx5/anomaly"
  command_topic: "jetson/nx5/cmd"
resample_interval_ms: 100
window_size: 30
calibration:
  max_duration: "7d"
  min_samples: 3000
"""
_GROUPS = """\
groups:
  axis_x:
    tags: ["A:AxX_Act_Vel", "A:AxX_Act_Trq"]
"""


def _load(tmp_path, text):
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return load_equipment_config(path)


def test_deviation_only_config_has_no_model_tags(tmp_path):
    config = _load(tmp_path, _BASE + "deviation:\n  signals:\n    x_pos: \"A:AxX_Act_Pos\"\n")

    assert config.tags == () and config.groups == ()
    assert config.resolved_groups() == () and config.group_specs() == {}
    assert config.all_tags == ("A:AxX_Act_Pos",)
    assert config.deviation == DeviationConfig(
        signals=(DeviationSignalConfig(name="x_pos", tag="A:AxX_Act_Pos", move_range=None),)
    )


def test_deviation_defaults(tmp_path):
    config = _load(tmp_path, _BASE + 'deviation:\n  signals:\n    x_pos: "t"\n')

    assert config.deviation.baseline_seconds == 2.0
    assert config.deviation.settle_seconds == 1.0
    assert config.deviation.margin == 1.2
    assert config.deviation.confirm_steps == 1


def test_deviation_options_and_signal_mapping_form(tmp_path):
    text = (
        _BASE
        + "deviation:\n  baseline_seconds: 3\n  settle_seconds: 0.5\n  margin: 1.5\n  confirm_steps: 2\n"
        + "  signals:\n    x_pos: {tag: \"t1\", move_range: 0.02}\n    z_pos: \"t2\"\n"
    )
    deviation = _load(tmp_path, text).deviation

    assert (deviation.baseline_seconds, deviation.settle_seconds) == (3.0, 0.5)
    assert (deviation.margin, deviation.confirm_steps) == (1.5, 2)
    assert deviation.signals == (
        DeviationSignalConfig("x_pos", "t1", 0.02),
        DeviationSignalConfig("z_pos", "t2", None),
    )


def test_deviation_can_be_combined_with_groups(tmp_path):
    config = _load(tmp_path, _BASE + _GROUPS + 'deviation:\n  signals:\n    x_pos: "A:AxX_Act_Pos"\n')

    assert config.tags == ("A:AxX_Act_Vel", "A:AxX_Act_Trq")           # GRU 입력 태그는 그대로
    assert config.all_tags == ("A:AxX_Act_Vel", "A:AxX_Act_Trq", "A:AxX_Act_Pos")
    assert config.group_specs() == {"axis_x": (None, ("A:AxX_Act_Vel", "A:AxX_Act_Trq"))}


def test_all_tags_does_not_repeat_a_tag_used_by_both(tmp_path):
    config = _load(tmp_path, _BASE + _GROUPS + 'deviation:\n  signals:\n    x_vel: "A:AxX_Act_Vel"\n')

    assert config.all_tags == ("A:AxX_Act_Vel", "A:AxX_Act_Trq")


def test_config_without_deviation_is_unchanged(tmp_path):
    config = _load(tmp_path, _BASE + _GROUPS)

    assert config.deviation is None and config.all_tags == config.tags


@pytest.mark.parametrize(
    "block, message",
    [
        ("deviation: 3\n", "deviation section must be a mapping"),
        ("deviation: {}\n", "signals"),
        ("deviation:\n  signals: {}\n", "non-empty mapping"),
        ("deviation:\n  signals:\n    a: 3\n", "tag"),
        ("deviation:\n  signals:\n    a: {}\n", "tag"),
        ("deviation:\n  signals:\n    a: \"t\"\n    b: \"t\"\n", "more than once"),
        ("deviation:\n  signals:\n    axis_x: \"t\"\n", "same name as a group"),
        ("deviation:\n  baseline_seconds: 0\n  signals:\n    a: \"t\"\n", "baseline_seconds"),
        ("deviation:\n  settle_seconds: -1\n  signals:\n    a: \"t\"\n", "settle_seconds"),
        ("deviation:\n  margin: 0\n  signals:\n    a: \"t\"\n", "margin"),
        ("deviation:\n  confirm_steps: 0\n  signals:\n    a: \"t\"\n", "confirm_steps"),
        ("deviation:\n  signals:\n    a: {tag: \"t\", move_range: 0}\n", "move_range"),
        ("deviation:\n  unknown_option: 1\n  signals:\n    a: \"t\"\n", "unknown"),
    ],
)
def test_deviation_validation(tmp_path, block, message):
    with pytest.raises(ConfigError, match=message):
        _load(tmp_path, _BASE + _GROUPS + block)


def test_neither_tags_groups_nor_deviation_is_still_rejected(tmp_path):
    with pytest.raises(ConfigError, match="tags"):
        _load(tmp_path, _BASE)
