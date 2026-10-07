import json
import socket
import urllib.error
import urllib.request

import pytest

from jetson_app.publisher import GroupOutput
from jetson_app.score_history import ScoreHistory
from jetson_app.web_server import ScoreWebServer


@pytest.fixture
def server():
    history = ScoreHistory(1000, ("axis_x",))
    for i in range(300):
        history.record(i * 100, {"axis_x": GroupOutput(score=float(i), alarm=i > 290, top_tag="tag")})
    srv = ScoreWebServer(history, lambda: {"equipment_id": "t", "groups": ["axis_x"], "state": "MONITORING"}, port=0)
    srv.start()
    yield srv
    srv.stop()


def _get(server, path):
    url = f"http://127.0.0.1:{server.port}{path}"
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.headers, response.read()


def test_serves_the_dashboard_page_in_live_mode(server):
    status, headers, body = _get(server, "/")
    text = body.decode("utf-8")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert headers["Cache-Control"] == "no-store" and headers["X-Content-Type-Options"] == "nosniff"
    assert 'const BOOT = {"mode":"live"};' in text


def test_healthz_and_meta(server):
    assert json.loads(_get(server, "/healthz")[2]) == {"ok": True}
    meta = json.loads(_get(server, "/api/meta")[2])
    assert meta["equipment_id"] == "t" and meta["state"] == "MONITORING"


def test_scores_endpoint_respects_seconds_and_max_points(server):
    data = json.loads(_get(server, "/api/scores?seconds=10&max_points=1000")[2])
    assert data["now_ms"] == 29900
    assert data["points"][0]["t"] >= 19900 and data["points"][-1]["g"]["axis_x"] == [299.0, True, "tag"]
    small = json.loads(_get(server, "/api/scores?seconds=30&max_points=60")[2])
    assert 50 <= len(small["points"]) <= 61
    assert max(p["g"]["axis_x"][0] for p in small["points"]) == 299.0


def test_bad_query_values_fall_back_to_defaults(server):
    for query in ("seconds=abc", "seconds=nan&max_points=inf", "max_points=-5", "seconds=-1"):
        status, _headers, body = _get(server, "/api/scores?" + query)
        assert status == 200 and "points" in json.loads(body)


def test_unknown_path_is_404_and_writes_are_405(server):
    with pytest.raises(urllib.error.HTTPError) as not_found:
        _get(server, "/nope")
    assert not_found.value.code == 404
    request = urllib.request.Request(f"http://127.0.0.1:{server.port}/api/scores", data=b"x", method="POST")
    with pytest.raises(urllib.error.HTTPError) as not_allowed:
        urllib.request.urlopen(request, timeout=5)
    assert not_allowed.value.code == 405
    assert not_allowed.value.headers["Allow"] == "GET"


def test_port_in_use_raises_oserror():
    history = ScoreHistory(10, ("a",))
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    try:
        with pytest.raises(OSError):
            ScoreWebServer(history, dict, port=blocker.getsockname()[1])
    finally:
        blocker.close()
