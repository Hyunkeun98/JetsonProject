import json
import time
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

from jetson_app.calibration import CalibrationState, StateStore
from jetson_app.config import CalibrationConfig, EquipmentConfig
from jetson_app.model import AnomalyGRU
from jetson_app.pipeline import build_pipeline
from jetson_app.training import ModelArtifact, load_artifact, make_train_fn, save_artifact


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
        model_dir=tmp_path / "model_data",
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
        model_dir=tmp_path / "model_data",
        train_fn=lambda samples: None,
    )

    assert pipeline.command_subscriber._calibration_manager is pipeline.calibration_manager


def test_build_pipeline_on_record_updates_tag_buffer(tmp_path):
    config = _make_config(tmp_path)
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=tmp_path / "model_data",
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
    model_dir = tmp_path / "model_data"

    def _fake_train_fn(samples):
        # 실제 학습 대신, wrapped_train_fn(pipeline.py)이 학습 직후 곧바로
        # load_artifact로 다시 불러올 수 있도록 최소한의 유효한 아티팩트만 저장한다.
        train_calls.append(samples)
        model = AnomalyGRU(
            num_tags=1, continuous_indices=[0], binary_indices=[], hidden_size=2, num_layers=1
        )
        artifact = ModelArtifact(
            tags=config.tags,
            tag_types={"tag_a": "continuous"},
            norm_stats={"tag_a": (0.0, 1.0)},
            error_stats={"tag_a": (0.0, 1.0)},
            window_size=config.window_size,
            hidden_size=2,
            num_layers=1,
            state_dict=model.state_dict(),
        )
        save_artifact(model_dir / f"{config.equipment_id}.pt", artifact)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=_fake_train_fn,
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
    model_dir = tmp_path / "model_data"
    # wrapped_train_fn(pipeline.py)이 학습 직후 model_artifact_path(model_dir, equipment_id)에서
    # 즉시 다시 읽어들이므로, 여기 model_path도 그 규칙과 일치해야 한다.
    model_path = model_dir / f"{config.equipment_id}.pt"
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
        model_dir=model_dir,
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


def test_pipeline_scores_and_publishes_after_training(tmp_path):
    config = EquipmentConfig(
        equipment_id="e2e_score",
        subscribe_topics=("dx1/e2e_score/data",),
        publish_topic="jetson/e2e_score/anomaly",
        command_topic="jetson/e2e_score/cmd",
        tags=("tag_a", "tag_b"),
        resample_interval_ms=5,
        window_size=3,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=10),
    )
    model_dir = tmp_path / "model_data"
    train_fn = make_train_fn(
        tags=config.tags,
        window_size=config.window_size,
        model_path=model_dir / f"{config.equipment_id}.pt",
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )

    def _send(i):
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
        pipeline.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))

    pipeline.snapshotter.start()
    try:
        for i in range(20):
            _send(i)
            time.sleep(0.01)
    finally:
        pipeline.snapshotter.stop()

    pipeline.command_subscriber._handle_command_message(
        None, None, SimpleNamespace(payload=b'{"command": "train"}')
    )
    assert pipeline.calibration_manager.state == CalibrationState.MONITORING
    assert pipeline.inference_engine_holder.get() is not None

    published = []

    def _fake_publish(topic, payload):
        published.append((topic, payload))
        return SimpleNamespace(rc=0)  # ResultPublisher가 rc를 확인한다

    pipeline.mqtt_subscriber.client.publish = _fake_publish

    pipeline.snapshotter.start()
    try:
        for i in range(20, 30):
            _send(i)
            time.sleep(0.01)
    finally:
        pipeline.snapshotter.stop()

    assert published, "MONITORING 진입 후에도 이상 점수가 발행되지 않았다"
    topic, payload = published[0]
    assert topic == config.publish_topic
    record = json.loads(payload)["records"][0]
    assert "jetson:anomaly_score" in record
    assert "jetson:alarm" in record
    assert record["jetson:top_deviant_tag"] in config.tags


def test_pipeline_resumes_monitoring_after_restart(tmp_path):
    config = EquipmentConfig(
        equipment_id="e2e_resume",
        subscribe_topics=("dx1/e2e_resume/data",),
        publish_topic="jetson/e2e_resume/anomaly",
        command_topic="jetson/e2e_resume/cmd",
        tags=("tag_a", "tag_b"),
        resample_interval_ms=5,
        window_size=3,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=10),
    )
    model_dir = tmp_path / "model_data"
    train_fn = make_train_fn(
        tags=config.tags,
        window_size=config.window_size,
        model_path=model_dir / f"{config.equipment_id}.pt",
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    first.snapshotter.start()
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
            first.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))
            time.sleep(0.01)
    finally:
        first.snapshotter.stop()
    first.command_subscriber._handle_command_message(
        None, None, SimpleNamespace(payload=b'{"command": "train"}')
    )
    assert first.calibration_manager.state == CalibrationState.MONITORING

    # "재시작": 같은 calibration_dir/model_dir로 파이프라인을 새로 만든다
    second = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    assert second.calibration_manager.state == CalibrationState.MONITORING
    assert second.inference_engine_holder.get() is not None


def _feed_and_train(pipeline, tags, count=20):
    """MQTT 메시지를 흘려 캘리브레이션 버퍼를 채운 뒤 train 명령까지 보낸다
    (test_pipeline_resumes_monitoring_after_restart의 절차를 그대로 따른다)."""
    pipeline.snapshotter.start()
    try:
        for i in range(count):
            record = {"timestamp": "2026-08-04T00:00:00+0000"}
            for j, tag in enumerate(tags):
                record[tag] = float(i) if j == 0 else i % 2
            payload = json.dumps({"records": [record]}).encode("utf-8")
            pipeline.mqtt_subscriber._handle_message(
                None, None, SimpleNamespace(payload=payload)
            )
            time.sleep(0.01)
    finally:
        pipeline.snapshotter.stop()
    pipeline.command_subscriber._handle_command_message(
        None, None, SimpleNamespace(payload=b'{"command": "train"}')
    )


def test_pipeline_falls_back_to_calibrating_when_config_window_size_changed(tmp_path):
    """모델 학습 후 config의 window_size가 바뀌면, 모델은 정상 로드되지만
    InferenceEngine.score()가 매 틱 None을 반환해 아무 것도 발행하지 않는다.
    손상된 모델 파일과 동일하게 CALIBRATING으로 폴백해야 한다."""
    config = EquipmentConfig(
        equipment_id="e2e_wsmismatch",
        subscribe_topics=("dx1/e2e_wsmismatch/data",),
        publish_topic="jetson/e2e_wsmismatch/anomaly",
        command_topic="jetson/e2e_wsmismatch/cmd",
        tags=("tag_a", "tag_b"),
        resample_interval_ms=5,
        window_size=3,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=10),
    )
    model_dir = tmp_path / "model_data"
    train_fn = make_train_fn(
        tags=config.tags,
        window_size=config.window_size,
        model_path=model_dir / f"{config.equipment_id}.pt",
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed_and_train(first, config.tags)
    assert first.calibration_manager.state == CalibrationState.MONITORING

    # "재시작": config YAML의 window_size만 바뀐 채로 같은 model_dir을 다시 연다
    changed = replace(config, window_size=5)
    second = build_pipeline(
        config=changed,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )

    assert second.calibration_manager.state == CalibrationState.CALIBRATING
    assert second.inference_engine_holder.get() is None


def test_pipeline_falls_back_to_calibrating_when_config_tags_changed(tmp_path):
    config = EquipmentConfig(
        equipment_id="e2e_tagmismatch",
        subscribe_topics=("dx1/e2e_tagmismatch/data",),
        publish_topic="jetson/e2e_tagmismatch/anomaly",
        command_topic="jetson/e2e_tagmismatch/cmd",
        tags=("tag_a", "tag_b"),
        resample_interval_ms=5,
        window_size=3,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=10),
    )
    model_dir = tmp_path / "model_data"
    train_fn = make_train_fn(
        tags=config.tags,
        window_size=config.window_size,
        model_path=model_dir / f"{config.equipment_id}.pt",
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed_and_train(first, config.tags)
    assert first.calibration_manager.state == CalibrationState.MONITORING

    # "재시작": tag_b가 tag_c로 교체된 config
    changed = replace(config, tags=("tag_a", "tag_c"))
    second = build_pipeline(
        config=changed,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )

    assert second.calibration_manager.state == CalibrationState.CALIBRATING
    assert second.inference_engine_holder.get() is None


def test_pipeline_stays_calibrating_after_recalibrate_even_though_model_file_remains(tmp_path):
    """recalibrate는 의도적으로 모델 파일을 지우지 않고 상태 마커만 되돌린다.
    그 직후 재시작하면 남아있는 모델로 MONITORING을 재개하는 게 아니라
    CALIBRATING에 머물러야 한다."""
    config = EquipmentConfig(
        equipment_id="e2e_recal_restart",
        subscribe_topics=("dx1/e2e_recal_restart/data",),
        publish_topic="jetson/e2e_recal_restart/anomaly",
        command_topic="jetson/e2e_recal_restart/cmd",
        tags=("tag_a", "tag_b"),
        resample_interval_ms=5,
        window_size=3,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=10),
    )
    model_dir = tmp_path / "model_data"
    train_fn = make_train_fn(
        tags=config.tags,
        window_size=config.window_size,
        model_path=model_dir / f"{config.equipment_id}.pt",
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed_and_train(first, config.tags)
    assert first.calibration_manager.state == CalibrationState.MONITORING

    first.command_subscriber._handle_command_message(
        None, None, SimpleNamespace(payload=b'{"command": "recalibrate"}')
    )
    assert first.calibration_manager.state == CalibrationState.CALIBRATING

    model_path = model_dir / f"{config.equipment_id}.pt"
    assert model_path.exists()  # 모델 파일은 의도적으로 남아있어야 한다

    second = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    assert second.calibration_manager.state == CalibrationState.CALIBRATING
    assert second.inference_engine_holder.get() is None


def test_pipeline_falls_back_to_calibrating_when_model_file_corrupted(tmp_path):
    config = _make_config(tmp_path)
    model_dir = tmp_path / "model_data"
    model_dir.mkdir(parents=True)
    (model_dir / f"{config.equipment_id}.pt").write_bytes(b"not a real torch file")
    StateStore(model_dir / f"{config.equipment_id}.state").write(CalibrationState.MONITORING)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=lambda samples: None,
    )

    assert pipeline.calibration_manager.state == CalibrationState.CALIBRATING
    assert pipeline.inference_engine_holder.get() is None
