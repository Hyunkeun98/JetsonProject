import http.client
import json
import urllib.error
import urllib.request

import pytest

from jetson_app.studio.controller import StudioController
from jetson_app.studio.server import StudioServer
from studio_fakes import FakeIO, SourceFactory

CONNECTION = {"equipment_id": "nx5", "host": "localhost", "port": 1883, "subscribe_topics": ["dx1/data"]}


def _start(tmp_path, **kwargs):
    controller = StudioController(tmp_path, io=FakeIO(), source_factory=SourceFactory(), run_worker=False)
    server = StudioServer(controller, port=0, **kwargs)
    server.start()
    return controller, server


@pytest.fixture
def open_server(tmp_path):
    controller, server = _start(tmp_path)
    yield controller, server
    server.stop()


def _request(server, method, path, body=None, headers=None):
    url = f"http://127.0.0.1:{server.port}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8") or "null"), response
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        return e.code, (json.loads(raw) if raw.startswith("{") else raw), e


def test_pages_and_health(open_server):
    _, server = open_server

    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/") as response:
        assert response.status == 200 and "text/html" in response.headers["Content-Type"]
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/dashboard") as response:
        assert response.status == 200
    assert _request(server, "GET", "/healthz")[1] == {"ok": True}
    assert _request(server, "GET", "/nope")[0] == 404


def test_state_reports_stage_and_control_flags(open_server):
    _, server = open_server

    status, body, _ = _request(server, "GET", "/api/state")

    assert status == 200
    assert body["stage"] == "CONNECT" and body["step"] == 1
    assert body["control_enabled"] is True and body["token_required"] is False
    assert body["can"]["save_connection"] is True


def test_the_connection_round_trips_and_moves_to_the_probe_step(open_server):
    _, server = open_server

    assert _request(server, "GET", "/api/connection")[1] == {}
    assert _request(server, "POST", "/api/connection", CONNECTION)[0] == 200

    assert _request(server, "GET", "/api/connection")[1]["subscribe_topics"] == ["dx1/data"]
    assert _request(server, "GET", "/api/state")[1]["stage"] == "PROBE"
    assert _request(server, "GET", "/api/meta")[1]["state"] == "IDLE"
    assert _request(server, "GET", "/api/scores?seconds=60")[1]["points"] == []


def test_user_errors_come_back_as_400_with_the_message(open_server):
    _, server = open_server

    status, body, _ = _request(server, "POST", "/api/start", {})
    assert status == 400 and "연결 정보" in body["error"]

    status, body, _ = _request(server, "POST", "/api/collect/import", {"data_dir": "C:/definitely/not/here"})
    assert status == 400 and "연결 정보" in body["error"]

    _request(server, "POST", "/api/connection", CONNECTION)
    status, body, _ = _request(server, "POST", "/api/collect/import", {"data_dir": "C:/definitely/not/here"})
    assert status == 400 and "찾을 수 없습니다" in body["error"]

    status, body, _ = _request(server, "POST", "/api/analyze", {})
    assert status == 400 and "수집" in body["error"]

    status, body, _ = _request(server, "POST", "/api/connection", dict(CONNECTION, host="localhost:1883"))
    assert status == 400 and "포트" in body["error"]


def test_wrong_body_types_are_400_not_500(open_server):
    _, server = open_server
    _request(server, "POST", "/api/connection", CONNECTION)

    assert _request(server, "POST", "/api/train", {"tags": "pos"})[0] == 400
    assert _request(server, "POST", "/api/analyze", {"cycle_tag": 5})[0] == 400
    assert _request(server, "POST", "/api/collect/import", {"data_dir": ["x"]})[0] == 400


def test_bad_json_and_non_object_bodies_are_400(open_server):
    _, server = open_server
    url = f"http://127.0.0.1:{server.port}/api/start"

    for raw in (b"{not json", b"[1, 2]"):
        request = urllib.request.Request(url, data=raw, method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code == 400


def test_oversized_body_is_rejected_with_413(open_server):
    _, server = open_server
    conn = http.client.HTTPConnection("127.0.0.1", server.port)
    conn.putrequest("POST", "/api/analyze")
    conn.putheader("Content-Length", str(70_000))
    conn.endheaders()

    response = conn.getresponse()

    assert response.status == 413
    conn.close()


def test_post_with_a_foreign_origin_is_rejected(open_server):
    _, server = open_server

    status, body, _ = _request(server, "POST", "/api/start", {}, headers={"Origin": "http://evil.example"})

    assert status == 403


def test_post_with_a_foreign_host_header_is_rejected_on_loopback(open_server):
    _, server = open_server
    conn = http.client.HTTPConnection("127.0.0.1", server.port)

    conn.request("POST", "/api/start", body=b"{}", headers={"Host": "evil.example"})

    assert conn.getresponse().status == 403
    conn.close()


def test_token_is_required_for_post_when_configured(tmp_path):
    _, server = _start(tmp_path, token="s3cret")
    try:
        assert _request(server, "GET", "/api/state")[1]["token_required"] is True
        assert _request(server, "POST", "/api/connection", CONNECTION)[0] == 403
        assert _request(server, "POST", "/api/connection", CONNECTION, headers={"X-Studio-Token": "wrong"})[0] == 403
        assert _request(server, "POST", "/api/connection", CONNECTION, headers={"X-Studio-Token": "s3cret"})[0] == 200
    finally:
        server.stop()


def test_control_can_be_disabled_entirely(tmp_path):
    _, server = _start(tmp_path, control_enabled=False)
    try:
        status, body, _ = _request(server, "POST", "/api/connection", CONNECTION)
        assert status == 403 and "꺼져" in body["error"]
        assert _request(server, "GET", "/api/state")[1]["control_enabled"] is False
    finally:
        server.stop()


def test_a_second_server_cannot_bind_a_port_that_is_already_serving(open_server, tmp_path):
    _, server = open_server

    with pytest.raises(OSError):
        StudioServer(
            StudioController(tmp_path / "other", io=FakeIO(), source_factory=SourceFactory(), run_worker=False),
            port=server.port,
        )


def test_negative_content_length_is_rejected(open_server):
    _, server = open_server
    conn = http.client.HTTPConnection("127.0.0.1", server.port)
    conn.putrequest("POST", "/api/start")
    conn.putheader("Content-Length", "-5")
    conn.endheaders()

    assert conn.getresponse().status == 400
    conn.close()


def test_a_failing_get_returns_a_generic_500_instead_of_dropping_the_connection(open_server, monkeypatch):
    controller, server = open_server
    monkeypatch.setattr(controller, "state", lambda: (_ for _ in ()).throw(OverflowError("boom")))

    status, body, _ = _request(server, "GET", "/api/state")

    assert status == 500 and "내부 오류" in body["error"] and "boom" not in body["error"]
