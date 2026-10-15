from types import SimpleNamespace

import pytest

from jetson_app.studio.probe import ProbeSession
from jetson_app.studio.raw_source import RawSource
from studio_fakes import FakeSource, payload


class FakeClient:
    def __init__(self):
        self.subscribed = []
        self.events = []
        self.on_connect = self.on_disconnect = self.on_message = None

    def connect(self, host, port, keepalive=60):
        self.events.append(("connect", host, port))

    def loop_start(self):
        self.events.append("loop_start")

    def loop_stop(self):
        self.events.append("loop_stop")

    def disconnect(self):
        self.events.append("disconnect")

    def subscribe(self, topic, qos=0):
        self.subscribed.append((topic, qos))


def test_raw_source_subscribes_on_connect_and_forwards_payloads(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr("paho.mqtt.client.Client", lambda: client)
    received = []
    source = RawSource()

    source.connect("broker", 1883, ["dx1/AxisData", "dx1/ProcStart"], lambda topic, data: received.append((topic, data)))
    assert source.is_connected() is False                        # 접속 응답(on_connect)을 받기 전

    client.on_connect(client, None, {}, 0)
    client.on_message(client, None, SimpleNamespace(topic="dx1/AxisData", payload=b"{}"))

    assert source.is_connected() is True
    assert client.subscribed == [("dx1/AxisData", 1), ("dx1/ProcStart", 1)]
    assert received == [("dx1/AxisData", b"{}")]
    client.on_disconnect(client, None, 1)
    assert source.is_connected() is False
    source.disconnect()
    assert client.events[-2:] == ["loop_stop", "disconnect"]


def test_raw_source_does_not_subscribe_when_the_broker_refuses(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr("paho.mqtt.client.Client", lambda: client)
    source = RawSource()
    source.connect("broker", 1883, ["t"], lambda topic, data: None)

    client.on_connect(client, None, {}, 5)

    assert client.subscribed == [] and source.is_connected() is False


def _probe(clock=None, **kwargs):
    source = FakeSource()
    now = clock if clock is not None else [0.0]
    session = ProbeSession(source, "broker", 1883, ["dx1/AxisData"], clock=lambda: now[0], **kwargs)
    return session, source, now


def test_probe_reports_counts_rate_last_message_and_numeric_tags():
    session, source, now = _probe()
    session.start()

    for i in range(10):
        now[0] = i * 0.5
        source.emit(payload(i, pos=1.5, state="RUN", on=True), topic="dx1/AxisData")
    now[0] = 5.0
    info = session.snapshot()

    assert info["active"] is True and info["connected"] is True
    assert info["messages"] == 10 and info["rate"] == pytest.approx(2.0)
    assert info["tags"] == ["on", "pos"] and info["tag_count"] == 2          # 문자열 값은 숫자 태그가 아니다
    assert '"pos":1.5' in info["last_message"].replace(" ", "")
    assert info["topics"] == ["dx1/AxisData"] and info["last_rx_age"] == pytest.approx(0.5)


def test_probe_stop_disconnects_and_ignores_later_payloads():
    session, source, _ = _probe()
    session.start()
    source.emit(payload(0, pos=1.0))

    session.stop()
    source.emit(payload(1, pos=2.0))

    assert source.connected is False
    assert session.snapshot()["messages"] == 1 and session.snapshot()["active"] is False


def test_probe_stops_itself_after_five_minutes():
    session, source, now = _probe()
    session.start()

    now[0] = 301.0
    info = session.snapshot()

    assert info["active"] is False and info["expired"] is True and source.connected is False


def test_probe_start_passes_a_refused_connection_on():
    session = ProbeSession(FakeSource(fail=True), "broker", 1883, ["t"])

    with pytest.raises(OSError):
        session.start()

    assert session.snapshot()["active"] is False


def test_probe_keeps_only_the_last_message_text_bounded():
    session, source, _ = _probe()
    session.start()
    source.emit(b'{"records":[{"timestamp":"x","pos":1}]}' + b" " * 5000)

    assert len(session.snapshot()["last_message"]) <= 2000


# --- 리뷰 반영 -----------------------------------------------------------------------------------


def test_probe_ignores_payloads_it_cannot_parse():
    session, source, _ = _probe()
    session.start()

    source.emit(b'{"records":[{"timestamp":"x","big":' + b"9" * 400 + b"}]}")
    source.emit(b"[" * 100000)
    source.emit(payload(0, pos=1.0))

    info = session.snapshot()
    assert info["messages"] == 3 and info["tags"] == ["pos"]


def test_probe_keeps_its_arrival_list_short_even_when_nobody_polls():
    session, source, now = _probe()
    session.start()

    for i in range(1000):
        now[0] = i * 0.1
        source.emit(payload(i, pos=1.0))

    assert len(session._arrivals) <= 55


def test_raw_source_survives_a_failing_callback_and_a_bad_topic_filter(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr("paho.mqtt.client.Client", lambda: client)

    def explode(topic, data):
        raise RuntimeError("boom")

    source = RawSource()
    source.connect("broker", 1883, ["dx1/#/Axis"], explode)
    client.subscribe = lambda topic, qos=0: (_ for _ in ()).throw(ValueError("bad topic"))

    client.on_connect(client, None, {}, 0)                        # 던지지 않는다
    client.on_message(client, None, SimpleNamespace(topic="t", payload=b"{}"))   # 던지지 않는다
