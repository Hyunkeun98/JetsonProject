from __future__ import annotations

import json

import openpyxl
import pytest

from jetson_app.config import load_equipment_config
from jetson_app.datalist_to_config import main

HEADER = ["Variables", "Variables Type", "Description", "MQTT Topic"]

NX5_LIKE_ROWS = [
    ["U0_TaktTime", "LREAL", "공정0 Takt Time (ms)", "dx1/TaktTime"],
    ["U0_ProcStart", "BOOL", "공정0 동작중 신호", "dx1/ProcStart"],
    ["AxX.Act.Pos", "LREAL", "공정0 서보 실제 위치", "dx1/AxisData"],
    ["AxX.Act.Vel[", "LREAL", "공정0 서보 실제 속도", "dx1/AxisData"],  # 끝의 [는 오탈자
    ["U1_TaktTime", "LREAL", "공정1 Takt Time (ms)", "dx1/TaktTime"],
    ["U1_ProcStart", "BOOL", "공정1 동작중 신호", "dx1/ProcStart"],
    [None, None, None, None],  # 빈 행
    [None, None, None, "Conveyor:ConvSen.Sensor[0..10]"],  # 변수 이름이 없는 메모 행
    ["ConvSensor0", "BOOL", "Conveyor Sensor 0", "dx1/SenData"],
]


def _make_xlsx(tmp_path, header, rows, name="list.xlsx"):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    path = tmp_path / name
    workbook.save(str(path))
    return path


def _nx5_args(xlsx, out, *extra):
    return [
        str(xlsx),
        "--equipment-id",
        "nx5",
        "--collector-prefix",
        "NX5",
        "--group-regex",
        r"공정(\d+)",
        "--group-template",
        "process_{}",
        "--state-pattern",
        "U{}_ProcStart",
        "--output",
        str(out),
        *extra,
    ]


def _write_samples(folder, tag_values, name="DX_sample.json", step_ms=100):
    """tag_values: {태그: [값, ...]} — 같은 길이의 리스트. 줄마다 record 하나(DX1 형식)."""
    folder.mkdir(parents=True, exist_ok=True)
    n = len(next(iter(tag_values.values())))
    lines = []
    for i in range(n):
        record = {"timestamp": "2026-10-01T05:32:20.%03d000000+0000" % (i * step_ms)}
        record.update({tag: values[i] for tag, values in tag_values.items()})
        lines.append(json.dumps({"records": [record]}))
    (folder / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_groups_state_tags_and_names_are_derived_from_description_and_patterns(tmp_path, capsys):
    xlsx = _make_xlsx(tmp_path, HEADER, NX5_LIKE_ROWS)
    out = tmp_path / "nx5.yaml"

    code = main(_nx5_args(xlsx, out))

    assert code == 0
    config = load_equipment_config(out)
    assert [g.name for g in config.groups] == ["process_0", "process_1", "general"]
    process_0, process_1, general = config.groups
    assert process_0.state_tag == "NX5_ProcStart:U0_ProcStart"
    assert process_0.tags == (
        "NX5_TaktTime:U0_TaktTime",
        "NX5_ProcStart:U0_ProcStart",
        "NX5_AxisData:AxX_Act_Pos",
        "NX5_AxisData:AxX_Act_Vel",  # 끝의 [ 제거
    )
    assert process_1.state_tag == "NX5_ProcStart:U1_ProcStart"
    assert general.state_tag is None
    assert general.tags == ("NX5_SenData:ConvSensor0",)
    # 구독 토픽은 엑셀 순서대로 중복 없이
    assert config.subscribe_topics == (
        "dx1/TaktTime",
        "dx1/ProcStart",
        "dx1/AxisData",
        "dx1/SenData",
    )
    assert config.publish_topic == "jetson/nx5/anomaly"
    assert config.command_topic == "jetson/nx5/cmd"
    assert "AxX.Act.Vel[" in capsys.readouterr().out  # 오탈자 경고


def test_explicit_tag_group_and_statetag_columns_are_used_as_written(tmp_path):
    rows = [
        ["a1", "t/x", "P:a1", "g1", "Y"],
        ["a2", "t/x", "P:a2", "g1", None],
        ["b1", "t/y", "Q:b1", None, None],
    ]
    xlsx = _make_xlsx(tmp_path, ["Variables", "MQTT Topic", "Tag", "Group", "StateTag"], rows)
    out = tmp_path / "cfg.yaml"

    code = main([str(xlsx), "--equipment-id", "eq", "--output", str(out)])

    assert code == 0
    config = load_equipment_config(out)
    assert [(g.name, g.state_tag, g.tags) for g in config.groups] == [
        ("g1", "P:a1", ("P:a1", "P:a2")),
        ("general", None, ("Q:b1",)),
    ]


def test_without_group_information_all_tags_go_to_general(tmp_path):
    xlsx = _make_xlsx(tmp_path, ["Variables", "MQTT Topic"], [["x", "dx1/A"], ["y", "dx1/A"]])
    out = tmp_path / "cfg.yaml"

    code = main([str(xlsx), "--equipment-id", "eq", "--collector-prefix", "M", "--output", str(out)])

    assert code == 0
    config = load_equipment_config(out)
    assert [(g.name, g.tags) for g in config.groups] == [("general", ("M_A:x", "M_A:y"))]


def test_two_state_tags_in_one_group_are_rejected(tmp_path, capsys):
    rows = [["a", "t", "P:a", "g", "Y"], ["b", "t", "P:b", "g", "Y"]]
    xlsx = _make_xlsx(tmp_path, ["Variables", "MQTT Topic", "Tag", "Group", "StateTag"], rows)
    out = tmp_path / "cfg.yaml"

    code = main([str(xlsx), "--equipment-id", "eq", "--output", str(out)])

    assert code == 1
    assert not out.exists()
    assert "g" in capsys.readouterr().err


def test_missing_required_column_is_an_input_error(tmp_path):
    xlsx = _make_xlsx(tmp_path, ["Variables", "Description"], [["x", "d"]])

    assert main([str(xlsx), "--equipment-id", "eq", "--output", str(tmp_path / "o.yaml")]) == 2


def test_duplicate_tags_are_rejected(tmp_path):
    rows = [["x", "t/A", "공정0"], ["x", "t/A", "공정0"]]
    xlsx = _make_xlsx(tmp_path, ["Variables", "MQTT Topic", "Description"], rows)
    out = tmp_path / "cfg.yaml"

    assert main([str(xlsx), "--equipment-id", "eq", "--output", str(out)]) == 1
    assert not out.exists()


def _sample_rows():
    return [
        ["U0_TaktTime", "LREAL", "공정0 Takt Time", "dx1/TaktTime"],
        ["U0_ProcStart", "BOOL", "공정0 동작중 신호", "dx1/ProcStart"],
        ["ConvSensor0", "BOOL", "Conveyor Sensor 0", "dx1/SenData"],
    ]


def _sample_values():
    return {
        "NX5_TaktTime:U0_TaktTime": [0.0, 0.0, 0.0],  # 변하지 않음
        "NX5_ProcStart:U0_ProcStart": [0, 1, 0],
        "NX5_SenData:ConvSensor0": [1, 1, 0],
        "NX5_Other:NotListed": [5, 6, 7],  # 목록에 없는 태그
    }


def test_samples_validate_tags_suggest_interval_and_never_add_unlisted_tags(tmp_path, capsys):
    xlsx = _make_xlsx(tmp_path, HEADER, _sample_rows())
    samples = tmp_path / "samples"
    _write_samples(samples, _sample_values(), step_ms=100)
    out = tmp_path / "nx5.yaml"

    code = main(_nx5_args(xlsx, out, "--samples", str(samples)))

    printed = capsys.readouterr().out
    assert code == 0
    assert load_equipment_config(out).resample_interval_ms == 100  # 샘플 간격 100ms
    assert "NotListed" not in out.read_text(encoding="utf-8")  # 목록에 없는 태그는 설정에 넣지 않는다
    assert "NX5_Other:NotListed" in printed  # 참고로만 출력
    assert "권장" in printed and "100" in printed
    assert "NX5_TaktTime:U0_TaktTime" in printed  # 값이 변하지 않은 태그 경고


def test_interval_suggestion_follows_the_sample_spacing(tmp_path):
    xlsx = _make_xlsx(tmp_path, HEADER, _sample_rows())
    samples = tmp_path / "samples"
    _write_samples(samples, _sample_values(), step_ms=200)
    out = tmp_path / "nx5.yaml"

    main(_nx5_args(xlsx, out, "--samples", str(samples)))

    assert load_equipment_config(out).resample_interval_ms == 200


def test_explicit_interval_argument_wins_over_the_suggestion(tmp_path):
    xlsx = _make_xlsx(tmp_path, HEADER, _sample_rows())
    samples = tmp_path / "samples"
    _write_samples(samples, _sample_values(), step_ms=100)
    out = tmp_path / "nx5.yaml"

    main(_nx5_args(xlsx, out, "--samples", str(samples), "--resample-interval-ms", "50"))

    assert load_equipment_config(out).resample_interval_ms == 50


def test_listed_tag_missing_from_samples_is_an_error_and_writes_nothing(tmp_path, capsys):
    xlsx = _make_xlsx(tmp_path, HEADER, _sample_rows())
    values = _sample_values()
    del values["NX5_SenData:ConvSensor0"]
    samples = tmp_path / "samples"
    _write_samples(samples, values)
    out = tmp_path / "nx5.yaml"

    code = main(_nx5_args(xlsx, out, "--samples", str(samples)))

    assert code == 1
    assert not out.exists()
    assert "NX5_SenData:ConvSensor0" in capsys.readouterr().err


def test_allow_missing_turns_the_error_into_a_warning(tmp_path, capsys):
    xlsx = _make_xlsx(tmp_path, HEADER, _sample_rows())
    values = _sample_values()
    del values["NX5_SenData:ConvSensor0"]
    samples = tmp_path / "samples"
    _write_samples(samples, values)
    out = tmp_path / "nx5.yaml"

    code = main(_nx5_args(xlsx, out, "--samples", str(samples), "--allow-missing"))

    assert code == 0
    assert out.exists()
    assert "NX5_SenData:ConvSensor0" in capsys.readouterr().out


def test_defaults_are_used_without_samples_and_can_be_overridden(tmp_path):
    xlsx = _make_xlsx(tmp_path, HEADER, _sample_rows())
    out = tmp_path / "nx5.yaml"

    main(_nx5_args(xlsx, out))
    defaults = load_equipment_config(out)
    main(
        _nx5_args(
            xlsx,
            out,
            "--resample-interval-ms",
            "20",
            "--max-lateness-ms",
            "300",
            "--window-size",
            "50",
            "--min-samples",
            "1234",
        )
    )
    custom = load_equipment_config(out)

    assert (defaults.resample_interval_ms, defaults.max_lateness_ms, defaults.window_size) == (
        100,
        500,
        30,
    )
    assert defaults.calibration.min_samples == 3000
    assert (custom.resample_interval_ms, custom.max_lateness_ms, custom.window_size) == (20, 300, 50)
    assert custom.calibration.min_samples == 1234


def test_without_output_the_yaml_goes_to_stdout(tmp_path, capsys):
    xlsx = _make_xlsx(tmp_path, HEADER, _sample_rows())

    code = main(
        [
            str(xlsx),
            "--equipment-id",
            "nx5",
            "--collector-prefix",
            "NX5",
        ]
    )

    assert code == 0
    assert "groups:" in capsys.readouterr().out
