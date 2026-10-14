"""화면이 쓰는 HTTP API만으로 설정 → 수집 → 학습 → 감시 → 충격 알람까지 끝낸다(MQTT는 가짜)."""
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from jetson_app.studio.controller import StudioController
from jetson_app.studio.server import StudioServer


class FakeIO:
    def connect(self, pipeline, host, port):
        pass

    def disconnect(self, pipeline):
        pass

    def is_connected(self, pipeline):
        return True


def _call(server, method, path, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=data, method=method)
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _ts(i):
    moment = datetime(2026, 8, 4, tzinfo=timezone.utc) + timedelta(milliseconds=i * 100)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000+0000"


def _noise(i):
    return 0.001 * (((i * 7919) % 13) - 6) / 6.0


def _feed(controller, start, stop, spike_at=None):
    for i in range(start, stop):
        value = _noise(i) + (0.01 if i == spike_at else 0.0)
        payload = json.dumps({"records": [{"timestamp": _ts(i), "pos": value}]}).encode("utf-8")
        controller.pipeline.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))
    controller.pump()


def test_whole_workflow_over_http(tmp_path):
    controller = StudioController(tmp_path, io=FakeIO(), run_worker=False)
    server = StudioServer(controller, port=0)
    server.start()
    try:
        assert _call(server, "GET", "/api/state")["stage"] == "SETUP"
        _call(server, "POST", "/api/config", {
            "equipment_id": "demo", "host": "localhost", "port": 1883,
            "subscribe_topics": ["dx1/data"], "tags": ["pos"], "min_samples": 200,
        })
        _call(server, "POST", "/api/start", {})
        _feed(controller, 0, 400)
        assert _call(server, "GET", "/api/state")["collected"] >= 200

        _call(server, "POST", "/api/train", {})
        controller.join_training(timeout=30)
        state = _call(server, "GET", "/api/state")
        assert state["stage"] == "MONITORING" and state["signals"][0]["name"] == "pos"

        _feed(controller, 400, 700, spike_at=500)
        points = _call(server, "GET", "/api/scores?seconds=3600&max_points=5000")["points"]
        assert sum(1 for p in points if p["g"]["pos"][1]) == 1

        _call(server, "POST", "/api/recalibrate", {})
        assert _call(server, "GET", "/api/state")["stage"] == "COLLECTING"
        _call(server, "POST", "/api/stop", {})
        assert _call(server, "GET", "/api/state")["stage"] == "READY"
    finally:
        server.stop()
