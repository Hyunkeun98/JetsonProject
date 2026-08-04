import types

from jetson_app.config import EquipmentConfig
from jetson_app.mqtt_subscriber import MqttRecordSubscriber, Record, parse_and_filter_records


def _make_config(**overrides):
    defaults = dict(
        equipment_id="test_dx1",
        subscribe_topic="dx1/test_dx1/telemetry",
        publish_topic="jetson/test_dx1/anomaly",
        tags=("servo1:torque", "sensor:A_L_01"),
    )
    defaults.update(overrides)
    return EquipmentConfig(**defaults)


class _FakeClient:
    def __init__(self):
        self.subscribed_topics = []

    def subscribe(self, topic):
        self.subscribed_topics.append(topic)


def test_parse_and_filter_records_keeps_only_configured_tags():
    payload = (
        b'{"records": [{"timestamp": "2026-08-03T00:00:00Z", '
        b'"servo1:torque": 12.3, "servo1:unused": 99.9, "sensor:A_L_01": 110.2}]}'
    )

    records = parse_and_filter_records(payload, tags=("servo1:torque", "sensor:A_L_01"))

    assert records == [
        Record(
            timestamp="2026-08-03T00:00:00Z",
            values={"servo1:torque": 12.3, "sensor:A_L_01": 110.2},
        )
    ]


def test_parse_and_filter_records_skips_record_with_no_matching_tags():
    payload = b'{"records": [{"timestamp": "2026-08-03T00:00:00Z", "unrelated:tag": 1.0}]}'

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_handles_multiple_records():
    payload = (
        b'{"records": ['
        b'{"timestamp": "t1", "servo1:torque": 1.0}, '
        b'{"timestamp": "t2", "servo1:torque": 2.0}'
        b']}'
    )

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert [r.timestamp for r in records] == ["t1", "t2"]


def test_parse_and_filter_records_handles_malformed_json():
    payload = b"not json"

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_handles_records_not_a_list():
    payload = b'{"records": null}'

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_skips_non_dict_items_in_records():
    payload = b'{"records": ["not a dict", {"timestamp": "t1", "servo1:torque": 1.0}]}'

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == [Record(timestamp="t1", values={"servo1:torque": 1.0})]


def test_parse_and_filter_records_handles_non_object_top_level_json():
    payload = b'["not", "an", "object"]'

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_handles_non_utf8_bytes():
    payload = b"\x80\x81\x82"  # invalid UTF-8 start byte -> UnicodeDecodeError

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_handle_connect_subscribes_on_success():
    config = _make_config()
    subscriber = MqttRecordSubscriber(config, on_record=lambda record: None)
    fake_client = _FakeClient()

    subscriber._handle_connect(fake_client, None, None, 0)

    assert fake_client.subscribed_topics == [config.subscribe_topic]


def test_handle_connect_does_not_subscribe_on_failure():
    config = _make_config()
    subscriber = MqttRecordSubscriber(config, on_record=lambda record: None)
    fake_client = _FakeClient()

    subscriber._handle_connect(fake_client, None, None, 5)

    assert fake_client.subscribed_topics == []


def test_handle_message_delivers_parsed_records_to_on_record():
    config = _make_config()
    received: list[Record] = []
    subscriber = MqttRecordSubscriber(config, on_record=received.append)
    fake_client = _FakeClient()
    msg = types.SimpleNamespace(
        payload=(
            b'{"records": [{"timestamp": "2026-08-03T00:00:00Z", '
            b'"servo1:torque": 12.3, "sensor:A_L_01": 110.2}]}'
        )
    )

    subscriber._handle_message(fake_client, None, msg)

    assert received == [
        Record(
            timestamp="2026-08-03T00:00:00Z",
            values={"servo1:torque": 12.3, "sensor:A_L_01": 110.2},
        )
    ]
