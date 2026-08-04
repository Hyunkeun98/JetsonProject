import json
import time
from datetime import timedelta
from types import SimpleNamespace

from jetson_app.calibration import CalibrationState
from jetson_app.config import CalibrationConfig, EquipmentConfig
from jetson_app.pipeline import build_pipeline


def _make_config(tmp_path):
    return EquipmentConfig(
        equipment_id="test_dx1",
        subscribe_topics=("dx1/test_dx1/actuator_1",),
        publish_topic="jetson/test_dx1/anomaly",
        command_topic="jetson/test_dx1/cmd",
        tags=("PLC_Collector_Actuator_1:AirBlower.Cmd[0]",),
        resample_interval_ms=50,
        window_size=10,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=10),
    )


def test_build_pipeline_wires_all_components(tmp_path):
    config = _make_config(tmp_path)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        train_fn=lambda samples: None,
    )

    assert pipeline.config is config
    assert pipeline.calibration_manager.state == CalibrationState.CALIBRATING
    assert pipeline.mqtt_subscriber.client is not None


def test_build_pipeline_command_subscriber_shares_calibration_manager(tmp_path):
    config = _make_config(tmp_path)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        train_fn=lambda samples: None,
    )

    assert pipeline.command_subscriber._calibration_manager is pipeline.calibration_manager


def test_build_pipeline_on_record_updates_tag_buffer(tmp_path):
    config = _make_config(tmp_path)
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        train_fn=lambda samples: None,
    )

    from jetson_app.mqtt_subscriber import Record

    pipeline.mqtt_subscriber._on_record(
        Record(
            timestamp="2026-08-04T00:00:00+00:00",
            values={"PLC_Collector_Actuator_1:AirBlower.Cmd[0]": 1},
        )
    )

    snapshot = pipeline.tag_buffer.snapshot()
    assert snapshot.values["PLC_Collector_Actuator_1:AirBlower.Cmd[0]"] == 1


def test_pipeline_end_to_end_message_through_training(tmp_path):
    config = EquipmentConfig(
        equipment_id="e2e_test",
        subscribe_topics=("dx1/e2e_test/data",),
        publish_topic="jetson/e2e_test/anomaly",
        command_topic="jetson/e2e_test/cmd",
        tags=("tag_a",),
        resample_interval_ms=5,
        window_size=10,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=3),
    )
    train_calls = []

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        train_fn=lambda samples: train_calls.append(samples),
    )

    payload = json.dumps(
        {"records": [{"timestamp": "2026-08-04T00:00:00+0000", "tag_a": 42}]}
    ).encode("utf-8")
    pipeline.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))

    pipeline.snapshotter.start()
    time.sleep(0.1)
    pipeline.snapshotter.stop()

    buffer_path = tmp_path / "calibration_data" / "e2e_test.jsonl"
    assert buffer_path.exists()
    recorded_lines = buffer_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(recorded_lines) >= 3

    pipeline.command_subscriber._handle_command_message(
        None, None, SimpleNamespace(payload=b'{"command": "train"}')
    )

    assert len(train_calls) == 1
    assert pipeline.calibration_manager.state == CalibrationState.MONITORING
    assert not buffer_path.exists()
