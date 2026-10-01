import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from jetson_app.calibration import CalibrationState, StateStore
from jetson_app.config import CalibrationConfig, EquipmentConfig
from jetson_app.model import AnomalyGRU
from jetson_app.mqtt_subscriber import Record
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


def _e2e_config(equipment_id, tags=("tag_a", "tag_b"), window_size=3, min_samples=10):
    # 격자 5ms, 지연 허용 0ms: 테스트에서 칸이 다음 record가 도착하는 즉시 확정된다.
    return EquipmentConfig(
        equipment_id=equipment_id,
        subscribe_topics=(f"dx1/{equipment_id}/data",),
        publish_topic=f"jetson/{equipment_id}/anomaly",
        command_topic=f"jetson/{equipment_id}/cmd",
        tags=tags,
        resample_interval_ms=5,
        window_size=window_size,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=min_samples),
        max_lateness_ms=0,
    )


def _ts(i, step_ms=5):
    """i번째 격자 칸의 DX1 형식 타임스탬프(나노초 9자리, +0000)."""
    moment = datetime(2026, 8, 4, tzinfo=timezone.utc) + timedelta(milliseconds=i * step_ms)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000+0000"


def _send(pipeline, i, tags):
    """i번째 시점의 record 하나를 담은 MQTT 메시지를 구독자에 직접 전달한다.
    첫 태그는 연속값(float(i)), 나머지는 0/1(i % 2)."""
    record = {"timestamp": _ts(i)}
    for j, tag in enumerate(tags):
        record[tag] = float(i) if j == 0 else i % 2
    payload = json.dumps({"records": [record]}).encode("utf-8")
    pipeline.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))


def _feed(pipeline, tags, start, stop):
    for i in range(start, stop):
        _send(pipeline, i, tags)
    pipeline.snapshotter.process_pending()


def _train_fn_for(config, model_dir):
    return make_train_fn(
        tags=config.tags,
        window_size=config.window_size,
        model_path=model_dir / f"{config.equipment_id}.pt",
        resample_interval_ms=config.resample_interval_ms,
        groups=config.group_specs(),
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )


def _send_train(pipeline):
    pipeline.command_subscriber._handle_command_message(
        None, None, SimpleNamespace(payload=b'{"command": "train"}')
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


def test_build_pipeline_on_record_feeds_resampler_and_processor(tmp_path):
    config = replace(_make_config(tmp_path), max_lateness_ms=0)
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=tmp_path / "model_data",
        train_fn=lambda samples: None,
    )
    tag = "PLC_Collector_Actuator_1:AirBlower.Cmd[0]"

    pipeline.mqtt_subscriber._on_record(
        Record(timestamp="2026-08-04T00:00:00+00:00", values={tag: 1}, epoch_ns=0)
    )
    pipeline.mqtt_subscriber._on_record(
        Record(timestamp="2026-08-04T00:00:00.050+00:00", values={tag: 0}, epoch_ns=50_000_000)
    )
    pipeline.snapshotter.process_pending()

    # 두 번째 record가 첫 칸(0~50ms)을 확정시켰고, 그 칸의 값은 1이다.
    assert [s.values[tag] for s in pipeline.sliding_window.to_list()] == [1]


def test_pipeline_end_to_end_message_through_training(tmp_path):
    config = _e2e_config("e2e_test", tags=("tag_a",), min_samples=3)
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
            resample_interval_ms=config.resample_interval_ms,
            groups=config.group_specs(),
        )
        save_artifact(model_dir / f"{config.equipment_id}.pt", artifact)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=_fake_train_fn,
    )

    # 한 메시지에 시점이 다른 record 5개(배치) -> 마지막 칸을 뺀 4칸이 확정된다
    payload = json.dumps(
        {"records": [{"timestamp": _ts(i), "tag_a": 42} for i in range(5)]}
    ).encode("utf-8")
    pipeline.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))
    pipeline.snapshotter.process_pending()

    buffer_path = tmp_path / "calibration_data" / "e2e_test.jsonl"
    assert buffer_path.exists()
    recorded_lines = buffer_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(recorded_lines) == 4

    _send_train(pipeline)

    assert len(train_calls) == 1
    assert pipeline.calibration_manager.state == CalibrationState.MONITORING
    assert not buffer_path.exists()


def test_pipeline_batch_message_keeps_every_record_at_its_own_timestamp(tmp_path):
    config = replace(
        _e2e_config("e2e_batch", tags=("tag_a",), min_samples=1000),
        resample_interval_ms=100,
    )
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=tmp_path / "model_data",
        train_fn=lambda samples: None,
    )
    # DX1가 100ms 주기로 취득한 값 10개를 한 메시지로 묶어 보내고, 다음 메시지가 이어진다
    first = {"records": [{"timestamp": _ts(i, 100), "tag_a": i} for i in range(10)]}
    second = {"records": [{"timestamp": _ts(10, 100), "tag_a": 10}]}
    for message in (first, second):
        payload = json.dumps(message).encode("utf-8")
        pipeline.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))
    pipeline.snapshotter.process_pending()

    lines = (tmp_path / "calibration_data" / "e2e_batch.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["values"]["tag_a"] for row in rows] == list(range(10))
    stamps = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    assert [b - a for a, b in zip(stamps, stamps[1:])] == [timedelta(milliseconds=100)] * 9


def test_pipeline_worker_thread_processes_messages_end_to_end(tmp_path):
    config = _e2e_config("e2e_thread", tags=("tag_a",))
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=tmp_path / "model_data",
        train_fn=lambda samples: None,
    )

    pipeline.snapshotter.start()
    for i in range(6):
        _send(pipeline, i, config.tags)
    pipeline.snapshotter.stop()

    lines = (tmp_path / "calibration_data" / "e2e_thread.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 5


def test_pipeline_end_to_end_with_real_train_fn(tmp_path, capsys):
    """subscriber_cli가 하는 것과 동일한 방식으로 make_train_fn을 배선해,
    MQTT 메시지 → 스냅샷 → train 명령 → 모델 파일 저장까지 실제로 도는지 확인한다.
    (command_subscriber가 예외를 삼키므로 배선 버그는 이 경로 없이는 안 잡힌다.)"""
    config = _e2e_config("e2e_train")
    model_dir = tmp_path / "model_data"
    # wrapped_train_fn(pipeline.py)이 학습 직후 model_artifact_path(model_dir, equipment_id)에서
    # 즉시 다시 읽어들이므로, 여기 model_path도 그 규칙과 일치해야 한다.
    model_path = model_dir / f"{config.equipment_id}.pt"
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=_train_fn_for(config, model_dir),
    )

    _feed(pipeline, config.tags, 0, 20)

    buffer_path = tmp_path / "calibration_data" / "e2e_train.jsonl"
    recorded_lines = buffer_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(recorded_lines) >= config.calibration.min_samples

    _send_train(pipeline)

    # 학습이 실패하면 command_subscriber가 예외를 삼키므로 원인을 출력에서 보여준다
    assert model_path.exists(), capsys.readouterr().out
    assert pipeline.calibration_manager.state == CalibrationState.MONITORING
    artifact = load_artifact(model_path)
    assert artifact.tags == config.tags
    assert artifact.resample_interval_ms == config.resample_interval_ms


def test_pipeline_scores_and_publishes_after_training(tmp_path):
    config = _e2e_config("e2e_score")
    model_dir = tmp_path / "model_data"
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=_train_fn_for(config, model_dir),
    )

    _feed(pipeline, config.tags, 0, 20)
    _send_train(pipeline)
    assert pipeline.calibration_manager.state == CalibrationState.MONITORING
    assert pipeline.inference_engine_holder.get() is not None

    published = []

    def _fake_publish(topic, payload):
        published.append((topic, payload))
        return SimpleNamespace(rc=0)  # ResultPublisher가 rc를 확인한다

    pipeline.mqtt_subscriber.client.publish = _fake_publish

    _feed(pipeline, config.tags, 20, 30)

    assert published, "MONITORING 진입 후에도 이상 점수가 발행되지 않았다"
    topic, payload = published[0]
    assert topic == config.publish_topic
    record = json.loads(payload)["records"][0]
    assert "jetson:anomaly_score" in record
    assert "jetson:alarm" in record
    assert record["jetson:top_deviant_tag"] in config.tags


def test_pipeline_resumes_monitoring_after_restart(tmp_path):
    config = _e2e_config("e2e_resume")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed(first, config.tags, 0, 20)
    _send_train(first)
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
    """MQTT 메시지를 흘려 캘리브레이션 버퍼를 채운 뒤 train 명령까지 보낸다."""
    _feed(pipeline, tags, 0, count)
    _send_train(pipeline)


def test_pipeline_falls_back_to_calibrating_when_config_window_size_changed(tmp_path):
    """모델 학습 후 config의 window_size가 바뀌면, 모델은 정상 로드되지만
    InferenceEngine.score()가 매 스텝 None을 반환해 아무 것도 발행하지 않는다.
    손상된 모델 파일과 동일하게 CALIBRATING으로 폴백해야 한다."""
    config = _e2e_config("e2e_wsmismatch")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)

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
    config = _e2e_config("e2e_tagmismatch")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)

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


def test_pipeline_falls_back_to_calibrating_when_resample_interval_changed(tmp_path):
    """학습 때와 다른 격자 간격으로 윈도우를 만들면 모델 입력의 시간 간격이 달라져
    점수가 무의미해진다. window_size/tags가 바뀐 경우와 마찬가지로 CALIBRATING 폴백."""
    config = _e2e_config("e2e_gridmismatch")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed_and_train(first, config.tags)
    assert first.calibration_manager.state == CalibrationState.MONITORING

    changed = replace(config, resample_interval_ms=10)
    second = build_pipeline(
        config=changed,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )

    assert second.calibration_manager.state == CalibrationState.CALIBRATING
    assert second.inference_engine_holder.get() is None


def test_pipeline_falls_back_to_calibrating_for_artifact_without_grid_info(tmp_path):
    """격자 정보가 없는 기존 artifact(resample_interval_ms=0)는 어떤 config와도 호환되지
    않는 것으로 보고 재학습하게 한다."""
    config = _e2e_config("e2e_legacy_artifact", tags=("tag_a",))
    model_dir = tmp_path / "model_data"
    model = AnomalyGRU(
        num_tags=1, continuous_indices=[0], binary_indices=[], hidden_size=2, num_layers=1
    )
    save_artifact(
        model_dir / f"{config.equipment_id}.pt",
        ModelArtifact(
            tags=config.tags,
            tag_types={"tag_a": "continuous"},
            norm_stats={"tag_a": (0.0, 1.0)},
            error_stats={"tag_a": (0.0, 1.0)},
            window_size=config.window_size,
            hidden_size=2,
            num_layers=1,
            state_dict=model.state_dict(),
        ),
    )
    StateStore(model_dir / f"{config.equipment_id}.state").write(CalibrationState.MONITORING)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=lambda samples: None,
    )

    assert pipeline.calibration_manager.state == CalibrationState.CALIBRATING
    assert pipeline.inference_engine_holder.get() is None


def test_pipeline_stays_calibrating_after_recalibrate_even_though_model_file_remains(tmp_path):
    """recalibrate는 의도적으로 모델 파일을 지우지 않고 상태 마커만 되돌린다.
    그 직후 재시작하면 남아있는 모델로 MONITORING을 재개하는 게 아니라
    CALIBRATING에 머물러야 한다."""
    config = _e2e_config("e2e_recal_restart")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)

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


# ---- 그룹 설정 ----

from jetson_app.config import AlarmConfig, GroupConfig


def _grouped_config(equipment_id, window_size=3, min_samples=10):
    # _send는 첫 태그에 float(i), 나머지에 i % 2를 넣는다 -> s(상태 태그)는 0/1로 번갈아 변한다.
    base = _e2e_config(equipment_id, tags=("a", "s", "b"), window_size=window_size, min_samples=min_samples)
    return replace(
        base,
        groups=(
            GroupConfig(name="proc", state_tag="s", tags=("a", "s")),
            GroupConfig(name="general", state_tag=None, tags=("b",)),
        ),
    )


def test_group_specs_helper_describes_resolved_groups():
    flat = _e2e_config("flat", tags=("x", "y"))
    grouped = _grouped_config("grouped")

    assert flat.group_specs() == {"all": (None, ("x", "y"))}
    assert grouped.group_specs() == {"proc": ("s", ("a", "s")), "general": (None, ("b",))}


def test_pipeline_builds_one_debouncer_per_group_from_alarm_config(tmp_path):
    config = replace(_grouped_config("e2e_deb"), alarm=AlarmConfig(threshold=9.5, confirm_steps=7))

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=tmp_path / "model_data",
        train_fn=lambda samples: None,
    )

    debouncers = pipeline.snapshotter._debouncers
    assert set(debouncers) == {"proc", "general"}
    assert debouncers["proc"] is not debouncers["general"]
    assert debouncers["proc"]._threshold == 9.5
    assert debouncers["proc"]._confirm_ticks == 7


def test_grouped_pipeline_resumes_monitoring_after_restart(tmp_path):
    config = _grouped_config("e2e_grp_resume")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed_and_train(first, config.tags)
    assert first.calibration_manager.state == CalibrationState.MONITORING
    assert load_artifact(model_dir / f"{config.equipment_id}.pt").groups == config.group_specs()

    second = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )

    assert second.calibration_manager.state == CalibrationState.MONITORING
    assert second.inference_engine_holder.get() is not None


def test_pipeline_falls_back_to_calibrating_when_group_layout_changed(tmp_path):
    config = _grouped_config("e2e_grp_changed")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed_and_train(first, config.tags)
    assert first.calibration_manager.state == CalibrationState.MONITORING

    renamed = replace(
        config,
        groups=(
            GroupConfig(name="process_0", state_tag="s", tags=("a", "s")),
            GroupConfig(name="general", state_tag=None, tags=("b",)),
        ),
    )
    other_state = replace(
        config,
        groups=(
            GroupConfig(name="proc", state_tag=None, tags=("a", "s")),
            GroupConfig(name="general", state_tag=None, tags=("b",)),
        ),
    )
    for changed in (renamed, other_state):
        second = build_pipeline(
            config=changed,
            calibration_dir=tmp_path / "calibration_data",
            model_dir=model_dir,
            train_fn=train_fn,
        )
        assert second.calibration_manager.state == CalibrationState.CALIBRATING
        assert second.inference_engine_holder.get() is None


def test_grouped_pipeline_publishes_group_fields(tmp_path):
    config = _grouped_config("e2e_grp_publish")
    model_dir = tmp_path / "model_data"
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=_train_fn_for(config, model_dir),
    )
    _feed(pipeline, config.tags, 0, 20)
    _send_train(pipeline)
    assert pipeline.calibration_manager.state == CalibrationState.MONITORING

    published = []

    def _fake_publish(topic, payload):
        published.append((topic, payload))
        return SimpleNamespace(rc=0)

    pipeline.mqtt_subscriber.client.publish = _fake_publish
    _feed(pipeline, config.tags, 20, 30)

    assert published, "MONITORING 진입 후에도 이상 점수가 발행되지 않았다"
    record = json.loads(published[0][1])["records"][0]
    assert record["jetson:top_deviant_tag"] in config.tags
    for group in ("proc", "general"):
        assert f"jetson:{group}:score" in record
        assert isinstance(record[f"jetson:{group}:alarm"], bool)
        assert record[f"jetson:{group}:top_tag"] in config.group_specs()[group][1]


def test_flat_pipeline_publishes_the_original_three_fields(tmp_path):
    config = _e2e_config("e2e_flat_publish")
    model_dir = tmp_path / "model_data"
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=_train_fn_for(config, model_dir),
    )
    _feed(pipeline, config.tags, 0, 20)
    _send_train(pipeline)
    published = []

    def _fake_publish(topic, payload):
        published.append(payload)
        return SimpleNamespace(rc=0)

    pipeline.mqtt_subscriber.client.publish = _fake_publish
    _feed(pipeline, config.tags, 20, 30)

    record = json.loads(published[0])["records"][0]
    assert set(record) == {
        "timestamp",
        "jetson:anomaly_score",
        "jetson:alarm",
        "jetson:top_deviant_tag",
    }
