"""studio 시험용 가짜 부품과 DX1 payload 만들기."""
import json
from datetime import datetime, timedelta, timezone


def ts(i):
    moment = datetime(2026, 8, 4, tzinfo=timezone.utc) + timedelta(milliseconds=i * 100)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000+0000"


def noise(i):
    return 0.001 * (((i * 7919) % 13) - 6) / 6.0


def payload(i, **values):
    """100ms 간격 i번째 DX1 payload(레코드 하나)."""
    return json.dumps({"records": [dict(timestamp=ts(i), **values)]}).encode("utf-8")


class FakeSource:
    """RawSource와 같은 모양의 가짜 구독. emit()으로 payload를 흘려 넣는다."""

    def __init__(self, fail=False):
        self.fail = fail
        self.connected = False
        self.calls = []
        self._on_payload = None

    def connect(self, host, port, topics, on_payload):
        if self.fail:
            raise ConnectionRefusedError("refused")
        self.connected = True
        self.calls.append((host, port, tuple(topics)))
        self._on_payload = on_payload

    def disconnect(self):
        self.connected = False

    def is_connected(self):
        return self.connected

    def emit(self, payload_bytes, topic="dx1/data"):
        self._on_payload(topic, payload_bytes)


class SourceFactory:
    def __init__(self, fail=False):
        self.fail = fail
        self.created = []

    def __call__(self):
        source = FakeSource(self.fail)
        self.created.append(source)
        return source

    @property
    def last(self):
        return self.created[-1]


class FakeIO:
    """감시 파이프라인의 MQTT 연결을 흉내 낸다."""

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
