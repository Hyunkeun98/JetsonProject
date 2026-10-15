"""화면이 쓰는 HTTP API만으로 연결 → 살펴보기 → 수집 → 태그 추천 → 학습 → 감시 → 충격 알람까지 끝낸다(MQTT는 가짜)."""
import json
import urllib.request
from types import SimpleNamespace

from jetson_app.studio.controller import StudioController
from jetson_app.studio.server import StudioServer
from studio_fakes import FakeIO, SourceFactory, noise, payload


def _call(server, method, path, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=data, method=method)
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def test_whole_workflow_over_http(tmp_path):
    factory = SourceFactory()
    controller = StudioController(tmp_path, io=FakeIO(), source_factory=factory, run_worker=False)
    server = StudioServer(controller, port=0)
    server.start()
    try:
        assert _call(server, "GET", "/api/state")["stage"] == "CONNECT"
        _call(server, "POST", "/api/connection", {
            "equipment_id": "demo", "host": "localhost", "port": 1883, "subscribe_topics": ["dx1/data"],
        })
        assert _call(server, "GET", "/api/state")["stage"] == "PROBE"

        _call(server, "POST", "/api/probe/start", {})
        factory.last.emit(payload(0, pos=1.0, cyc=0))
        probe = _call(server, "GET", "/api/state")["probe"]
        assert probe["messages"] == 1 and probe["tags"] == ["cyc", "pos"]
        _call(server, "POST", "/api/probe/stop", {})

        _call(server, "POST", "/api/collect/start", {})
        for i in range(400):
            factory.last.emit(payload(i, pos=noise(i), cyc=1 if i % 100 >= 90 else 0))
        assert _call(server, "GET", "/api/state")["collect"]["messages"] == 400
        _call(server, "POST", "/api/collect/stop", {})
        assert _call(server, "GET", "/api/state")["stage"] == "TAGS"

        analysis = _call(server, "POST", "/api/analyze", {})
        assert {t["tag"] for t in analysis["tags"]} == {"pos", "cyc"}
        assert analysis["cycles"]["method"] == "edge" and analysis["cycles"]["warning"]

        _call(server, "POST", "/api/train", {"tags": ["pos"]})
        controller.join_training(timeout=30)
        state = _call(server, "GET", "/api/state")
        assert state["stage"] == "MONITOR" and state["signals"][0]["name"] == "pos"

        _call(server, "POST", "/api/start", {})
        for i in range(400, 700):
            value = noise(i) + (0.01 if i == 500 else 0.0)
            message = SimpleNamespace(payload=payload(i, pos=value))
            controller.pipeline.mqtt_subscriber._handle_message(None, None, message)
        controller.pump()
        points = _call(server, "GET", "/api/scores?seconds=3600&max_points=5000")["points"]
        assert sum(1 for p in points if p["g"]["pos"][1]) == 1

        _call(server, "POST", "/api/retrain", {})
        state = _call(server, "GET", "/api/state")
        assert state["stage"] == "TAGS" and state["running"] is False
    finally:
        server.stop()
