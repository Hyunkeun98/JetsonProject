import json
import math
from datetime import timedelta

import pytest

from jetson_app.config import (
    CalibrationConfig,
    DeviationConfig,
    DeviationSignalConfig,
    EquipmentConfig,
)
from jetson_app.deviation import deviation_artifact_path
from jetson_app.replay import ReplayError, score_from_files, train_from_files
from jetson_app.scores import read_scores_csv, write_scores_csv
from jetson_app.segments import KIND_NORMAL_EVAL, Segment, SegmentSet
from jetson_app.timeparse import format_epoch_ns, parse_dx1_timestamp

BASE_NS = parse_dx1_timestamp("2026-10-02T00:00:00Z")
SPIKE_SECOND = 50.0


def _ns(second):
    return BASE_NS + int(second * 1_000_000_000)


def _timestamp(second):
    return format_epoch_ns(_ns(second)).replace("+00:00", "+0000")


def _write_data(directory, seconds=60, spike_second=SPIKE_SECOND):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "DX_0.json").open("w", encoding="utf-8") as f:
        for n in range(seconds * 10):
            noise = 0.01 * (((n * 7919) % 13) - 6) / 6.0
            spike = 0.3 if n == int(spike_second * 10) else 0.0
            record = {"timestamp": _timestamp(n / 10), "pos": noise + spike, "g": math.sin(n / 5.0)}
            f.write(json.dumps({"records": [record]}) + "\n")


def _config(with_gru=False):
    return EquipmentConfig(
        equipment_id="rep_dev",
        subscribe_topics=("t/data",),
        publish_topic="p",
        command_topic="c",
        tags=("g",) if with_gru else (),
        resample_interval_ms=100,
        window_size=3,
        calibration=CalibrationConfig(max_duration=timedelta(days=1), min_samples=10),
        max_lateness_ms=0,
        deviation=DeviationConfig(
            signals=(DeviationSignalConfig(name="x_pos", tag="pos", move_range=1.0),)
        ),
    )


def _segments():
    return SegmentSet(
        train=(Segment("train", "t", _ns(0), _ns(40)),),
        normal_eval=(),
        anomaly=(Segment("anomaly", "spike", _ns(40), _ns(60)),),
    )


def test_deviation_only_train_then_score_finds_the_spike(tmp_path):
    _write_data(tmp_path / "d")
    config = _config()

    path, steps = train_from_files(config, tmp_path / "d", _segments(), tmp_path / "m")
    rows, group_names = score_from_files(config, tmp_path / "d", _segments(), tmp_path / "m")

    assert path == deviation_artifact_path(tmp_path / "m", "rep_dev") and path.exists()
    assert steps > 300 and group_names == ("x_pos",)
    assert rows[0].reset and not any(r.reset for r in rows[1:])
    hits = [r for r in rows if r.groups["x_pos"][0] >= config.alarm.threshold]
    assert len(hits) == 1
    # 점수는 settle(1초) 늦게 나온다: 충격 시각 + 1초의 행에 충격 샘플의 점수가 실린다.
    assert hits[0].epoch_ns == _ns(SPIKE_SECOND + 1.0)
    assert hits[0].groups["x_pos"][1] == "pos"


def test_gru_and_deviation_scores_share_rows_and_group_names(tmp_path):
    _write_data(tmp_path / "d")
    config = _config(with_gru=True)

    train_from_files(config, tmp_path / "d", _segments(), tmp_path / "m")
    rows, group_names = score_from_files(config, tmp_path / "d", _segments(), tmp_path / "m")

    assert group_names == ("all", "x_pos")
    both = [r for r in rows if {"all", "x_pos"} <= set(r.groups)]
    assert both and all(set(r.groups) <= {"all", "x_pos"} for r in rows)


def test_scores_csv_roundtrip_with_deviation_group(tmp_path):
    _write_data(tmp_path / "d")
    config = _config()
    train_from_files(config, tmp_path / "d", _segments(), tmp_path / "m")
    rows, group_names = score_from_files(config, tmp_path / "d", _segments(), tmp_path / "m")

    write_scores_csv(tmp_path / "s.csv", rows, group_names)
    read_rows, read_names = read_scores_csv(tmp_path / "s.csv")

    assert read_names == ("x_pos",) and len(read_rows) == len(rows)


def test_score_without_training_explains_what_to_do(tmp_path):
    _write_data(tmp_path / "d")

    with pytest.raises(ReplayError, match="jetson-replay train"):
        score_from_files(_config(), tmp_path / "d", _segments(), tmp_path / "m")


def test_score_after_config_change_asks_for_retraining(tmp_path):
    _write_data(tmp_path / "d")
    config = _config()
    train_from_files(config, tmp_path / "d", _segments(), tmp_path / "m")
    from dataclasses import replace

    changed = replace(config, deviation=replace(config.deviation, baseline_seconds=3.0))

    with pytest.raises(ReplayError, match="다시 학습"):
        score_from_files(changed, tmp_path / "d", _segments(), tmp_path / "m")
