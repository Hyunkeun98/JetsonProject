import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jetson_app.studio.controller import Stage, StudioController
from jetson_app.studio.errors import StudioError


class FakeIO:
    def __init__(self, fail=False):
        self.fail = fail
        self.up = False
        self.calls = []

    def connect(self, pipeline, host, port):
        if self.fail:
            raise ConnectionRefusedError("refused")
        self.up = True
        self.calls.append((host, port))

    def disconnect(self, pipeline):
        self.up = False

    def is_connected(self, pipeline):
        return self.up


def _form(**overrides):
    form = {
        "equipment_id": "nx5",
        "host": "localhost",
        "port": 1883,
        "subscribe_topics": ["dx1/data"],
        "tags": ["pos"],
        "min_samples": 200,
    }
    form.update(overrides)
    return form


def _controller(tmp_path, io=None, clock=None):
    kwargs = {"io": io or FakeIO(), "run_worker": False}
    if clock is not None:
        kwargs["clock"] = clock
    return StudioController(tmp_path, **kwargs)


def _ts(i):
    moment = datetime(2026, 8, 4, tzinfo=timezone.utc) + timedelta(milliseconds=i * 100)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000+0000"


def _noise(i):
    return 0.001 * (((i * 7919) % 13) - 6) / 6.0


def _feed(controller, start, stop, spike_at=None, constant=None):
    for i in range(start, stop):
        value = constant if constant is not None else _noise(i) + (0.01 if i == spike_at else 0.0)
        payload = json.dumps({"records": [{"timestamp": _ts(i), "pos": value}]}).encode("utf-8")
        controller.pipeline.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))
    controller.pump()


def test_a_fresh_workspace_starts_in_setup(tmp_path):
    controller = _controller(tmp_path)

    assert controller.state()["stage"] == Stage.SETUP.value
    assert controller.form() == {}


def test_saving_a_config_moves_to_ready_and_start_moves_to_collecting(tmp_path):
    io = FakeIO()
    controller = _controller(tmp_path, io)
    controller.save_config(_form())
    assert controller.state()["stage"] == "READY"

    controller.start()

    state = controller.state()
    assert state["stage"] == "COLLECTING" and state["connected"] is True
    assert state["collected"] == 0 and state["required"] == 200
    assert io.calls == [("localhost", 1883)]


def test_start_without_a_saved_config_is_refused(tmp_path):
    with pytest.raises(StudioError, match="설정"):
        _controller(tmp_path).start()


def test_start_twice_is_refused(tmp_path):
    controller = _controller(tmp_path)
    controller.save_config(_form())
    controller.start()

    with pytest.raises(StudioError, match="이미"):
        controller.start()


def test_a_broker_that_refuses_the_connection_keeps_the_stage_ready(tmp_path):
    controller = _controller(tmp_path, FakeIO(fail=True))
    controller.save_config(_form())

    with pytest.raises(StudioError, match="localhost:1883"):
        controller.start()

    assert controller.state()["stage"] == "READY"


def test_saving_a_config_while_running_is_refused(tmp_path):
    controller = _controller(tmp_path)
    controller.save_config(_form())
    controller.start()

    with pytest.raises(StudioError, match="중지"):
        controller.save_config(_form(tags=["pos", "pos2"]))


def test_training_before_enough_samples_is_refused_with_the_counts(tmp_path):
    controller = _controller(tmp_path)
    controller.save_config(_form())
    controller.start()
    _feed(controller, 0, 50)

    with pytest.raises(StudioError, match=r"부족.*/ 200"):
        controller.train()


def test_collect_train_monitor_and_alarm_on_a_shock(tmp_path):
    controller = _controller(tmp_path)
    controller.save_config(_form())
    controller.start()
    _feed(controller, 0, 400)
    assert controller.state()["collected"] >= 200

    controller.train()
    controller.join_training(timeout=30)

    state = controller.state()
    assert state["stage"] == "MONITORING" and state["error"] is None
    assert [s["name"] for s in state["signals"]] == ["pos"]
    assert state["signals"][0]["sigma"] > 0 and state["signals"][0]["limit"] > 0

    _feed(controller, 400, 700, spike_at=500)
    points = controller.scores(seconds=3600, max_points=5000)["points"]
    alarms = [p for p in points if p["g"]["pos"][1]]
    assert len(alarms) == 1 and alarms[0]["g"]["pos"][2] == "pos"
    assert controller.meta()["groups"] == ["pos"] and controller.meta()["state"] == "MONITORING"


def test_training_on_data_that_never_changes_reports_the_reason_and_can_be_retried(tmp_path):
    controller = _controller(tmp_path)
    controller.save_config(_form())
    controller.start()
    _feed(controller, 0, 400, constant=1.0)

    controller.train()
    controller.join_training(timeout=30)

    state = controller.state()
    assert state["stage"] == "COLLECTING"
    assert "변하지 않" in state["error"]
    assert state["collected"] >= 200                    # 수집한 데이터는 그대로 남아 있다


def test_recalibrate_goes_back_to_collecting_from_zero(tmp_path):
    controller = _controller(tmp_path)
    controller.save_config(_form())
    controller.start()
    _feed(controller, 0, 400)
    controller.train()
    controller.join_training(timeout=30)

    controller.recalibrate()

    state = controller.state()
    assert state["stage"] == "COLLECTING" and state["collected"] == 0 and state["signals"] == []


def test_stop_returns_to_ready_and_a_restart_resumes_monitoring(tmp_path):
    controller = _controller(tmp_path)
    controller.save_config(_form())
    controller.start()
    _feed(controller, 0, 400)
    controller.train()
    controller.join_training(timeout=30)

    controller.stop()
    assert controller.state()["stage"] == "READY"

    again = _controller(tmp_path)                       # 앱을 껐다 켠 것과 같다
    assert again.state()["stage"] == "READY"
    again.start()
    assert again.state()["stage"] == "MONITORING"


def test_a_second_train_request_while_training_is_refused(tmp_path):
    controller = _controller(tmp_path)
    controller.save_config(_form())
    controller.start()
    _feed(controller, 0, 400)
    controller._training = True                          # 학습 스레드가 도는 중인 상태를 흉내 낸다

    with pytest.raises(StudioError, match="이미 학습"):
        controller.train()


def test_no_data_warning_appears_after_thirty_silent_seconds(tmp_path):
    now = [0.0]
    controller = _controller(tmp_path, clock=lambda: now[0])
    controller.save_config(_form())
    controller.start()
    assert controller.state()["no_data"] is False

    now[0] = 31.0

    assert controller.state()["no_data"] is True
