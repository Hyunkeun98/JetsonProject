import json
import time
from datetime import timedelta
from types import SimpleNamespace

from jetson_app.calibration import CalibrationState
from jetson_app.config import CalibrationConfig, EquipmentConfig
from jetson_app.pipeline import build_pipeline
from jetson_app.training import load_artifact, make_train_fn


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


def test_pipeline_end_to_end_with_real_train_fn(tmp_path, capsys):
    """subscriber_cli가 하는 것과 동일한 방식으로 make_train_fn을 배선해,
    MQTT 메시지 → 스냅샷 → train 명령 → 모델 파일 저장까지 실제로 도는지 확인한다.
    (command_subscriber가 예외를 삼키므로 배선 버그는 이 경로 없이는 안 잡힌다.)"""
    config = EquipmentConfig(
        equipment_id="e2e_train",
        subscribe_topics=("dx1/e2e_train/data",),
        publish_topic="jetson/e2e_train/anomaly",
        command_topic="jetson/e2e_train/cmd",
        tags=("tag_a", "tag_b"),
        resample_interval_ms=5,
        window_size=3,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=10),
    )
    model_path = tmp_path / "m.pt"
    train_fn = make_train_fn(
        tags=config.tags,
        window_size=config.window_size,
        model_path=model_path,
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        train_fn=train_fn,
    )

    pipeline.snapshotter.start()
    try:
        for i in range(20):
            payload = json.dumps(
                {
                    "records": [
                        {
                            "timestamp": "2026-08-04T00:00:00+0000",
                            "tag_a": float(i),
                            "tag_b": i % 2,
                        }
                    ]
                }
            ).encode("utf-8")
            pipeline.mqtt_subscriber._handle_message(
                None, None, SimpleNamespace(payload=payload)
            )
            time.sleep(0.01)
    finally:
        pipeline.snapshotter.stop()

    buffer_path = tmp_path / "calibration_data" / "e2e_train.jsonl"
    recorded_lines = buffer_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(recorded_lines) >= config.calibration.min_samples

    pipeline.command_subscriber._handle_command_message(
        None, None, SimpleNamespace(payload=b'{"command": "train"}')
    )

    # 학습이 실패하면 command_subscriber가 예외를 삼키므로 원인을 출력에서 보여준다
    assert model_path.exists(), capsys.readouterr().out
    assert pipeline.calibration_manager.state == CalibrationState.MONITORING
    assert load_artifact(model_path).tags == config.tags
