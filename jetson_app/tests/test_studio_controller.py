import json
from types import SimpleNamespace

import pytest

from jetson_app.studio.controller import Stage, StudioController
from jetson_app.studio.errors import StudioError
from studio_fakes import FakeIO, SourceFactory, noise, payload

CONNECTION = {"equipment_id": "nx5", "host": "localhost", "port": 1883, "subscribe_topics": ["dx1/data"]}


def _controller(tmp_path, fail_source=False, io=None, **kwargs):
    factory = SourceFactory(fail=fail_source)
    io = io or FakeIO()
    controller = StudioController(tmp_path, io=io, source_factory=factory, run_worker=False, **kwargs)
    return controller, factory, io


def _collect(controller, factory, steps=400, value=None, stop=True):
    """수집을 시작하고 pos(와 사이클 신호 cyc)를 흘려 넣는다."""
    controller.collect_start()
    for i in range(steps):
        pos = value if value is not None else noise(i)
        factory.last.emit(payload(i, pos=pos, cyc=1 if i % 100 >= 90 else 0))
    if stop:
        controller.collect_stop()


def _train(controller, tags=("pos",)):
    controller.train(list(tags))
    controller.join_training(timeout=30)


def _feed_pipeline(controller, start, stop, spike_at=None):
    for i in range(start, stop):
        value = noise(i) + (0.01 if i == spike_at else 0.0)
        message = SimpleNamespace(payload=payload(i, pos=value))
        controller.pipeline.mqtt_subscriber._handle_message(None, None, message)
    controller.pump()


def test_a_fresh_workspace_starts_at_the_connection_step(tmp_path):
    controller, _, _ = _controller(tmp_path)

    state = controller.state()

    assert state["stage"] == Stage.CONNECT.value and state["step"] == 1
    assert controller.connection() == {}
    assert state["can"]["save_connection"] is True and state["can"]["probe_start"] is False


def test_saving_the_connection_moves_to_probe_and_survives_a_restart(tmp_path):
    controller, _, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)

    assert controller.state()["stage"] == "PROBE" and controller.state()["step"] == 2
    again, _, _ = _controller(tmp_path)
    assert again.state()["stage"] == "PROBE" and again.connection()["host"] == "localhost"


def test_bad_connection_input_is_a_studio_error(tmp_path):
    controller, _, _ = _controller(tmp_path)

    with pytest.raises(StudioError, match="포트"):
        controller.save_connection(dict(CONNECTION, host="localhost:1883"))


def test_probe_shows_incoming_data_and_collecting_stops_it(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)

    controller.probe_start()
    factory.last.emit(payload(0, pos=1.0, cyc=0))
    probe = controller.state()["probe"]
    assert probe["active"] is True and probe["messages"] == 1 and probe["tags"] == ["cyc", "pos"]

    controller.collect_start()                              # 시험은 저절로 끝난다
    assert controller.state()["probe"]["active"] is False and factory.created[0].connected is False
    assert controller.state()["stage"] == "COLLECT"


def test_probe_without_a_saved_connection_is_refused(tmp_path):
    controller, _, _ = _controller(tmp_path)

    with pytest.raises(StudioError, match="연결 정보"):
        controller.probe_start()


def test_a_broker_that_refuses_the_connection_keeps_the_stage_and_the_old_data(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory, steps=50)
    factory.fail = True

    with pytest.raises(StudioError, match="localhost:1883"):
        controller.collect_start()
    with pytest.raises(StudioError, match="localhost:1883"):
        controller.probe_start()

    state = controller.state()
    assert state["stage"] == "TAGS" and state["collect"]["has_data"] is True


def test_collect_then_analyze_gives_tags_and_the_cycle_count(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory)

    state = controller.state()
    assert state["stage"] == "TAGS" and state["step"] == 4 and state["collect"]["messages"] == 400

    result = controller.analyze()
    assert {t["tag"] for t in result["tags"]} == {"pos", "cyc"}
    assert result["cycles"]["method"] == "edge" and result["cycles"]["cycles"] == 4
    assert "적습니다" in result["cycles"]["warning"]              # 10사이클 미만은 경고만 한다
    assert "tags" not in controller.analyze(with_tags=False)


def test_analyze_without_data_is_refused(tmp_path):
    controller, _, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)

    with pytest.raises(StudioError, match="수집"):
        controller.analyze()


def test_train_needs_tags_and_data(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    with pytest.raises(StudioError, match="수집"):
        controller.train(["pos"])
    _collect(controller, factory)

    with pytest.raises(StudioError, match="태그"):
        controller.train([])


def test_collect_train_monitor_and_alarm_on_a_shock(tmp_path):
    controller, factory, io = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory)

    _train(controller)

    state = controller.state()
    assert state["stage"] == "MONITOR" and state["step"] == 6 and state["error"] is None
    assert state["model_ready"] is True and state["running"] is False
    assert [s["name"] for s in state["signals"]] == ["pos"] and state["signals"][0]["sigma"] > 0

    controller.start()
    assert io.calls == [("localhost", 1883)] and controller.state()["running"] is True
    _feed_pipeline(controller, 400, 700, spike_at=500)

    points = controller.scores(seconds=3600, max_points=5000)["points"]
    alarms = [p for p in points if p["g"]["pos"][1]]
    assert len(alarms) == 1 and alarms[0]["g"]["pos"][2] == "pos"
    assert controller.meta()["groups"] == ["pos"] and controller.meta()["state"] == "MONITORING"


def test_stop_keeps_the_model_and_a_restart_can_monitor_again(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory)
    _train(controller)
    controller.start()

    controller.stop()
    assert controller.state()["stage"] == "MONITOR" and controller.state()["running"] is False

    again, _, _ = _controller(tmp_path)                       # 앱을 껐다 켠 것과 같다
    assert again.state()["stage"] == "MONITOR" and again.state()["model_ready"] is True
    again.start()
    assert again.state()["running"] is True


def test_training_on_data_that_never_changes_reports_why_and_keeps_the_data(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory, value=1.0)

    _train(controller)

    state = controller.state()
    assert state["stage"] == "TAGS" and "변하지" in state["error"]
    assert state["collect"]["has_data"] is True and state["model_ready"] is False
    assert state["can"]["train"] is True                        # 태그를 바꿔 다시 시도할 수 있다


def test_retrain_goes_back_to_the_tag_step_with_the_same_data(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory)
    _train(controller)
    controller.start()

    controller.retrain()

    state = controller.state()
    assert state["stage"] == "TAGS" and state["running"] is False and state["model_ready"] is False
    assert state["collect"]["has_data"] is True


def test_a_new_collection_makes_the_old_model_unusable(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory)
    _train(controller)
    assert controller.state()["stage"] == "MONITOR"

    _collect(controller, factory, steps=50)

    state = controller.state()
    assert state["stage"] == "TAGS" and state["model_ready"] is False
    with pytest.raises(StudioError, match="모델"):
        controller.start()


def test_busy_states_refuse_conflicting_requests(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory)
    _train(controller)
    controller.start()

    with pytest.raises(StudioError, match="이미"):
        controller.start()
    with pytest.raises(StudioError, match="중지"):
        controller.save_connection(CONNECTION)
    with pytest.raises(StudioError, match="중지"):
        controller.collect_start()
    controller._training = True                                # 학습 스레드가 도는 중인 상태를 흉내 낸다
    with pytest.raises(StudioError, match="이미 학습"):
        controller.train(["pos"])
    controller._training = False


def test_saving_the_connection_while_collecting_is_refused(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory, steps=10, stop=False)

    with pytest.raises(StudioError, match="중지|종료"):
        controller.save_connection(CONNECTION)
    assert controller.state()["can"]["collect_stop"] is True and controller.state()["can"]["collect_start"] is False


def test_import_folder_replaces_the_collection(tmp_path):
    saved = tmp_path / "saved"
    saved.mkdir()
    (saved / "DX_0.json").write_bytes(b"\n".join(payload(i, pos=noise(i)) for i in range(400)) + b"\n")
    controller, _, _ = _controller(tmp_path / "ws")
    controller.save_connection(CONNECTION)

    controller.import_folder(str(saved))

    assert controller.state()["stage"] == "TAGS" and controller.state()["collect"]["origin"] == str(saved)


def test_discard_returns_to_probe(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory, steps=20)

    controller.discard()

    assert controller.state()["stage"] == "PROBE" and controller.state()["collect"]["has_data"] is False


def test_shutdown_closes_everything_and_tolerates_being_called_twice(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    controller.collect_start()
    factory.last.emit(payload(0, pos=1.0))

    controller.shutdown()
    controller.shutdown()

    assert factory.last.connected is False and controller.state()["collect"]["active"] is False
    json.dumps(controller.state())                              # 상태는 항상 JSON으로 보낼 수 있어야 한다


# --- 리뷰 반영 -----------------------------------------------------------------------------------


def test_analysis_in_progress_blocks_collect_import_discard_and_train(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory, steps=50)
    controller._analyzing = 1                                  # 분석이 도는 중인 상태를 흉내 낸다

    state = controller.state()
    assert state["analyzing"] is True
    assert not any(state["can"][name] for name in ("collect_start", "discard", "import", "train", "save_connection"))
    for call in (controller.collect_start, controller.discard, lambda: controller.import_folder("x"),
                 lambda: controller.train(["pos"])):
        with pytest.raises(StudioError, match="분석"):
            call()


def test_the_analysis_flag_is_cleared_even_when_the_analysis_fails(tmp_path, monkeypatch):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory, steps=50)
    monkeypatch.setattr("jetson_app.studio.wizard.analyze", lambda path: (_ for _ in ()).throw(StudioError("boom")))

    with pytest.raises(StudioError, match="boom"):
        controller.analyze()

    assert controller.state()["analyzing"] is False and controller.state()["can"]["collect_start"] is True
