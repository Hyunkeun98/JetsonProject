import json
import math

import pytest

from jetson_app.deviation_suggest import analyze_tags, discover_numeric_tags, format_report
from jetson_app.deviation_suggest_cli import main
from jetson_app.timeparse import format_epoch_ns, parse_dx1_timestamp

BASE_NS = parse_dx1_timestamp("2026-10-02T00:00:00Z")


def _timestamp(second):
    return format_epoch_ns(BASE_NS + int(second * 1_000_000_000)).replace("+00:00", "+0000")


def _noise(n, scale):
    return scale * (((n * 7919) % 13) - 6) / 6.0


def _write(directory, seconds=300):
    """0.1초 간격. pos: 가끔 먼 거리를 오가고 평소엔 거의 일정(잡음 1e-5), vel: 늘 크게 흔들림,
    flag: 0/1, const: 상수, label: 숫자가 아닌 값."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "DX_0.json").open("w", encoding="utf-8") as f:
        for n in range(seconds * 10):
            moving = (n // 600) % 2 == 1 and (n % 600) < 80          # 60초마다 8초 이동
            pos = 100.0 * ((n % 600) / 80.0 if moving else (1.0 if (n // 600) % 2 == 1 else 0.0))
            record = {
                "timestamp": _timestamp(n / 10),
                "m:pos": pos + _noise(n, 1e-5),
                "m:vel": 5.0 * math.sin(n / 3.0) + _noise(n, 0.5),
                "m:flag": (n // 100) % 2,
                "m:const": 7.0,
                "m:text": "ok",
                # 계단형: 거의 변하지 않다가 가끔 큰 폭으로 바뀌고, 평소엔 아주 작게만 흔들린다.
                "m:step": 10.0 * ((n // 100) % 7) + (0.001 if n % 50 == 0 else 0.0),
            }
            f.write(json.dumps({"records": [record]}) + "\n")


def test_discover_numeric_tags_skips_timestamp_text_and_booleans(tmp_path):
    _write(tmp_path / "d", seconds=5)
    (tmp_path / "d" / "extra.json").write_text(
        json.dumps({"records": [{"timestamp": _timestamp(1), "m:bool": True, "m:other": 1.5}]}) + "\n",
        encoding="utf-8",
    )

    assert discover_numeric_tags(tmp_path / "d") == ["m:const", "m:flag", "m:other", "m:pos", "m:step", "m:vel"]


def test_analyze_recommends_the_flat_wide_ranging_signal_and_explains_the_others(tmp_path):
    _write(tmp_path / "d")

    reports = {r.tag: r for r in analyze_tags(tmp_path / "d")}

    assert reports["m:pos"].recommended
    assert not reports["m:vel"].recommended and "잡음" in reports["m:vel"].reason
    assert not reports["m:flag"].recommended and reports["m:flag"].kind == "binary"
    assert not reports["m:const"].recommended and reports["m:const"].kind == "constant"
    assert reports["m:pos"].dynamic_range > reports["m:vel"].dynamic_range
    assert 0.0 < reports["m:pos"].moving_fraction < 0.5


def test_analyze_does_not_recommend_a_step_like_signal(tmp_path):
    _write(tmp_path / "d")

    report = {r.tag: r for r in analyze_tags(tmp_path / "d")}["m:step"]

    assert not report.recommended and "계단" in report.reason


def test_report_text_lists_recommended_first_and_gives_a_config_snippet(tmp_path):
    _write(tmp_path / "d")

    text = format_report(analyze_tags(tmp_path / "d"))

    assert text.index("m:pos") < text.index("m:vel")
    assert "deviation:" in text and "signals:" in text and "m:pos" in text
    assert text.count("추천") >= 1


def test_report_explains_why_continuous_signals_were_not_recommended(tmp_path):
    _write(tmp_path / "d")

    text = format_report(analyze_tags(tmp_path / "d"))

    not_recommended = text.split("추천하지 않은 연속 신호")[1]
    assert "m:vel" in not_recommended and "m:step" in not_recommended
    assert "m:flag" not in not_recommended           # 0/1, 상수는 종류 열로 충분하다


def test_report_says_so_when_nothing_is_recommended(tmp_path):
    tmp_path.joinpath("d").mkdir()
    with (tmp_path / "d" / "x.json").open("w", encoding="utf-8") as f:
        for n in range(3000):
            f.write(json.dumps({"records": [{"timestamp": _timestamp(n / 10), "m:vel": 5.0 * math.sin(n / 3.0) + _noise(n, 0.5)}]}) + "\n")

    text = format_report(analyze_tags(tmp_path / "d"))

    assert "추천할 태그가 없습니다" in text and "deviation:" not in text


def test_cli_prints_the_report(tmp_path, capsys):
    _write(tmp_path / "d")

    main(["--data-dir", str(tmp_path / "d")])

    out = capsys.readouterr().out
    assert "m:pos" in out and "deviation:" in out


def test_cli_reports_missing_folder(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--data-dir", str(tmp_path / "nope")])

    assert exc.value.code == 2
    assert "찾을 수 없습니다" in capsys.readouterr().err
