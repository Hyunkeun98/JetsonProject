import json
import math

import pytest

from jetson_app.studio import wizard
from jetson_app.studio.errors import StudioError
from jetson_app.timeparse import format_epoch_ns, parse_dx1_timestamp

BASE_NS = parse_dx1_timestamp("2026-10-02T00:00:00Z")


def _timestamp(second):
    return format_epoch_ns(BASE_NS + int(second * 1_000_000_000)).replace("+00:00", "+0000")


def _noise(n, scale):
    return scale * (((n * 7919) % 13) - 6) / 6.0


def _write_data(directory, seconds=300):
    """pos는 가끔 크게 움직이고 평소엔 거의 일정(추천 대상), vel은 늘 흔들림(비추천)."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "DX_0.json").open("w", encoding="utf-8") as f:
        for n in range(seconds * 10):
            moving = (n // 600) % 2 == 1 and (n % 600) < 80
            pos = 100.0 * ((n % 600) / 80.0 if moving else (1.0 if (n // 600) % 2 == 1 else 0.0))
            record = {
                "timestamp": _timestamp(n / 10),
                "m:pos": pos + _noise(n, 1e-5),
                "m:vel": 5.0 * math.sin(n / 3.0) + _noise(n, 0.5),
            }
            f.write(json.dumps({"records": [record]}) + "\n")


def _form(**overrides):
    form = {
        "equipment_id": "nx5",
        "host": "localhost",
        "port": 1883,
        "subscribe_topics": ["dx1/AxisData"],
        "tags": ["m:pos"],
        "min_samples": 200,
    }
    form.update(overrides)
    return form


def test_analyze_recommends_the_position_tag(tmp_path):
    _write_data(tmp_path / "d")

    result = wizard.analyze(str(tmp_path / "d"))

    by_tag = {t["tag"]: t for t in result["tags"]}
    assert by_tag["m:pos"]["recommended"] is True
    assert by_tag["m:vel"]["recommended"] is False and by_tag["m:vel"]["reason"]
    json.dumps(result)                                   # 그대로 JSON 응답으로 보낼 수 있어야 한다


def test_analyze_accepts_a_path_pasted_with_quotes_and_spaces(tmp_path):
    _write_data(tmp_path / "my data")

    result = wizard.analyze(f'  "{tmp_path / "my data"}"  ')

    assert any(t["recommended"] for t in result["tags"])


def test_analyze_reports_a_missing_folder_as_studio_error(tmp_path):
    with pytest.raises(StudioError, match="찾을 수 없습니다"):
        wizard.analyze(str(tmp_path / "nope"))


def test_save_config_writes_a_valid_deviation_config_and_settings(tmp_path):
    config = wizard.save_config(tmp_path, _form())

    assert config.equipment_id == "nx5"
    assert config.deviation is not None and config.deviation.signals[0].tag == "m:pos"
    assert config.tags == () and config.calibration.min_samples == 200
    assert (tmp_path / "configs" / "nx5.yaml").exists()
    assert wizard.load_settings(tmp_path) == {"config": "nx5.yaml", "host": "localhost", "port": 1883}


def test_saved_form_restores_what_was_entered(tmp_path):
    wizard.save_config(tmp_path, _form(tags=["m:pos", "m:pos2"]))

    form = wizard.saved_form(tmp_path)

    assert form["equipment_id"] == "nx5" and form["host"] == "localhost" and form["port"] == 1883
    assert form["subscribe_topics"] == ["dx1/AxisData"]
    assert form["tags"] == ["m:pos", "m:pos2"] and form["min_samples"] == 200


def test_saved_form_is_empty_before_anything_is_saved(tmp_path):
    assert wizard.saved_form(tmp_path) == {}
    assert wizard.load_settings(tmp_path) is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"equipment_id": "../evil"}, "장비 이름"),
        ({"equipment_id": ""}, "장비 이름"),
        ({"host": "localhost:1883"}, "포트"),
        ({"host": ""}, "브로커 주소"),
        ({"port": 0}, "포트"),
        ({"port": "abc"}, "포트"),
        ({"subscribe_topics": []}, "토픽"),
        ({"tags": []}, "태그"),
        ({"min_samples": 0}, "수집"),
    ],
)
def test_save_config_rejects_bad_input_without_writing_anything(tmp_path, overrides, message):
    with pytest.raises(StudioError, match=message):
        wizard.save_config(tmp_path, _form(**overrides))

    assert not (tmp_path / "configs").exists() or not list((tmp_path / "configs").iterdir())
    assert wizard.load_settings(tmp_path) is None
