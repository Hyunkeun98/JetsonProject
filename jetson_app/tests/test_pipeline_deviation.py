import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from jetson_app.calibration import CalibrationState
from jetson_app.config import (
    CalibrationConfig,
    DeviationConfig,
    DeviationSignalConfig,
    EquipmentConfig,
)
from jetson_app.deviation import deviation_artifact_path
from jetson_app.pipeline import build_pipeline
from jetson_app.training import make_equipment_train_fn

TAG = "pos"


def _config(equipment_id, move_range=0.05, with_gru=False):
    # 격자 5ms, 지연 허용 0ms. 기준 구간 10스텝, 판정 지연 5스텝이 되도록 초 단위를 맞춘다.
    return EquipmentConfig(
        equipment_id=equipment_id,
        subscribe_topics=(f"dx1/{equipment_id}/data",),
        publish_topic=f"jetson/{equipment_id}/anomaly",
        command_topic=f"jetson/{equipment_id}/cmd",
        tags=("g",) if with_gru else (),
        resample_interval_ms=5,
        window_size=3,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=200),
        max_lateness_ms=0,
        deviation=DeviationConfig(
            signals=(DeviationSignalConfig(name="x_pos", tag=TAG, move_range=move_range),),
            baseline_seconds=0.05,
            settle_seconds=0.025,
        ),
    )


def _ts(i):
    moment = datetime(2026, 8, 4, tzinfo=timezone.utc) + timedelta(milliseconds=i * 5)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000+0000"


def _noise(i):
    return 0.001 * (((i * 7919) % 13) - 6) / 6.0     # 결정적인 작은 잡음(최대 +-0.001)


def _send(pipeline, i, value, with_gru=False):
    record = {"timestamp": _ts(i), TAG: value}
    if with_gru:
        record["g"] = float(i % 2)
    payload = json.dumps({"records": [record]}).encode("utf-8")
    pipeline.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))


def _feed(pipeline, start, stop, spike_at=None, spike=0.01, with_gru=False):
    for i in range(start, stop):
        _send(pipeline, i, _noise(i) + (spike if i == spike_at else 0.0), with_gru)
    pipeline.snapshotter.process_pending()


def _train(pipeline):
    pipeline.command_subscriber._handle_command_message(
        None, None, SimpleNamespace(payload=b'{"command": "train"}')
    )


def _build(config, tmp_path):
    model_dir = tmp_path / "model_data"
    return build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=make_equipment_train_fn(config, model_dir, epochs=1, hidden_size=4, num_layers=1),
    )


def _capture(pipeline):
    published = []

    def _fake_publish(topic, payload):
        published.append(json.loads(payload)["records"][0])
        return SimpleNamespace(rc=0)

    pipeline.mqtt_subscriber.client.publish = _fake_publish
    return published


def test_deviation_only_pipeline_trains_and_starts_monitoring(tmp_path):
    config = _config("dev_train")
    pipeline = _build(config, tmp_path)
    assert pipeline.inference_engine_holder.get() is None

    _feed(pipeline, 0, 400)
    _train(pipeline)

    assert pipeline.calibration_manager.state == CalibrationState.MONITORING
    assert pipeline.deviation_holder.get() is not None
    assert pipeline.inference_engine_holder.get() is None        # GRU는 쓰지 않는다
    assert deviation_artifact_path(tmp_path / "model_data", "dev_train").exists()


def test_deviation_pipeline_raises_alarm_on_spike_and_not_on_noise(tmp_path):
    config = _config("dev_alarm")
    pipeline = _build(config, tmp_path)
    _feed(pipeline, 0, 400)
    _train(pipeline)
    published = _capture(pipeline)

    _feed(pipeline, 400, 700, spike_at=500)

    assert published
    spike_records = [r for r in published if r["jetson:alarm"]]
    assert len(spike_records) == 1                                  # 충격 샘플 하나만 알람
    assert spike_records[0]["jetson:top_deviant_tag"] == TAG
    assert max(r["jetson:anomaly_score"] for r in published if not r["jetson:alarm"]) < 3.0


def test_deviation_pipeline_resumes_monitoring_after_restart(tmp_path):
    config = _config("dev_resume")
    first = _build(config, tmp_path)
    _feed(first, 0, 400)
    _train(first)

    second = _build(config, tmp_path)

    assert second.calibration_manager.state == CalibrationState.MONITORING
    assert second.deviation_holder.get() is not None


def test_deviation_pipeline_falls_back_to_calibrating_when_signal_tag_changed(tmp_path):
    config = _config("dev_tag")
    first = _build(config, tmp_path)
    _feed(first, 0, 400)
    _train(first)

    changed = replace(
        config,
        deviation=replace(config.deviation, signals=(DeviationSignalConfig("x_pos", "other"),)),
    )
    second = _build(changed, tmp_path)

    assert second.calibration_manager.state == CalibrationState.CALIBRATING
    assert second.deviation_holder.get() is None


def test_deviation_pipeline_falls_back_to_calibrating_when_baseline_changed(tmp_path):
    config = _config("dev_baseline")
    first = _build(config, tmp_path)
    _feed(first, 0, 400)
    _train(first)

    changed = replace(config, deviation=replace(config.deviation, baseline_seconds=0.1))
    second = _build(changed, tmp_path)

    assert second.calibration_manager.state == CalibrationState.CALIBRATING


def test_deviation_pipeline_falls_back_to_calibrating_when_artifact_file_missing(tmp_path):
    config = _config("dev_missing")
    first = _build(config, tmp_path)
    _feed(first, 0, 400)
    _train(first)
    deviation_artifact_path(tmp_path / "model_data", "dev_missing").unlink()

    second = _build(config, tmp_path)

    assert second.calibration_manager.state == CalibrationState.CALIBRATING


def test_gru_and_deviation_pipeline_publishes_both_group_kinds(tmp_path):
    config = _config("dev_both", with_gru=True)
    pipeline = _build(config, tmp_path)
    _feed(pipeline, 0, 400, with_gru=True)
    _train(pipeline)
    assert pipeline.inference_engine_holder.get() is not None
    assert pipeline.deviation_holder.get() is not None
    published = _capture(pipeline)

    _feed(pipeline, 400, 450, with_gru=True)

    record = published[-1]
    assert "jetson:all:score" in record and "jetson:x_pos:score" in record
    assert isinstance(record["jetson:x_pos:alarm"], bool)


def test_all_tags_are_subscribed_and_resampled(tmp_path):
    config = _config("dev_tags", with_gru=True)
    pipeline = _build(config, tmp_path)

    assert set(pipeline.resampler._tags) == {"g", TAG}
