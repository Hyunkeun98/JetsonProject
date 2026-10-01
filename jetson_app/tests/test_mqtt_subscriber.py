import types
from datetime import timedelta
from unittest.mock import MagicMock

from jetson_app.config import CalibrationConfig, EquipmentConfig
from jetson_app.mqtt_subscriber import MqttRecordSubscriber, Record, parse_and_filter_records
from jetson_app.timeparse import parse_dx1_timestamp


_EPOCH_2026_08_03_NS = parse_dx1_timestamp("2026-08-03T00:00:00Z")


def _make_config(subscribe_topics, command_topic="jetson/x/cmd"):
    return EquipmentConfig(
        equipment_id="x",
        subscribe_topics=subscribe_topics,
        publish_topic="jetson/x/anomaly",
        command_topic=command_topic,
        tags=("a",),
        resample_interval_ms=50,
        window_size=100,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=10),
    )


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
            epoch_ns=_EPOCH_2026_08_03_NS,
        )
    ]


def test_parse_and_filter_records_skips_record_with_no_matching_tags():
    payload = b'{"records": [{"timestamp": "2026-08-03T00:00:00Z", "unrelated:tag": 1.0}]}'

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_handles_multiple_records():
    payload = (
        b'{"records": ['
        b'{"timestamp": "2026-08-03T00:00:01Z", "servo1:torque": 1.0}, '
        b'{"timestamp": "2026-08-03T00:00:02Z", "servo1:torque": 2.0}'
        b']}'
    )

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert [r.timestamp for r in records] == ["2026-08-03T00:00:01Z", "2026-08-03T00:00:02Z"]


def test_parse_and_filter_records_handles_malformed_json():
    payload = b"not json"

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_handles_records_not_a_list():
    payload = b'{"records": null}'

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_skips_non_dict_items_in_records():
    payload = (
        b'{"records": ["not a dict", '
        b'{"timestamp": "2026-08-03T00:00:00Z", "servo1:torque": 1.0}]}'
    )

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == [
        Record(
            timestamp="2026-08-03T00:00:00Z",
            values={"servo1:torque": 1.0},
            epoch_ns=_EPOCH_2026_08_03_NS,
        )
    ]


def test_parse_and_filter_records_handles_non_object_top_level_json():
    payload = b'["not", "an", "object"]'

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_handles_non_utf8_bytes():
    payload = b"\x80\x81\x82"  # invalid UTF-8 start byte -> UnicodeDecodeError

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_handle_connect_subscribes_on_success():
    config = _make_config(subscribe_topics=("topic/a",))
    subscriber = MqttRecordSubscriber(config, on_record=lambda record: None)
    fake_client = MagicMock()

    subscriber._handle_connect(fake_client, None, None, 0)

    subscribed_topics = {call.args[0] for call in fake_client.subscribe.call_args_list}
    assert "topic/a" in subscribed_topics


def test_handle_connect_does_not_subscribe_on_failure():
    config = _make_config(subscribe_topics=("topic/a",))
    subscriber = MqttRecordSubscriber(config, on_record=lambda record: None)
    fake_client = MagicMock()

    subscriber._handle_connect(fake_client, None, None, 5)

    fake_client.subscribe.assert_not_called()


def test_handle_message_delivers_parsed_records_to_on_record():
    config = _make_config(subscribe_topics=("topic/a",))
    received: list[Record] = []
    subscriber = MqttRecordSubscriber(config, on_record=received.append)
    fake_client = _FakeClient()
    msg = types.SimpleNamespace(
        payload=b'{"records": [{"timestamp": "2026-08-03T00:00:00Z", "a": 12.3}]}'
    )

    subscriber._handle_message(fake_client, None, msg)

    assert received == [
        Record(timestamp="2026-08-03T00:00:00Z", values={"a": 12.3}, epoch_ns=_EPOCH_2026_08_03_NS)
    ]


def test_handle_connect_subscribes_to_all_configured_topics():
    config = _make_config(subscribe_topics=("topic/a", "topic/b"))
    subscriber = MqttRecordSubscriber(config, on_record=lambda r: None)
    fake_client = MagicMock()

    subscriber._handle_connect(fake_client, None, None, 0)

    subscribed_topics = {call.args[0] for call in fake_client.subscribe.call_args_list}
    assert subscribed_topics == {"topic/a", "topic/b", "jetson/x/cmd"}


def test_client_property_exposes_underlying_paho_client():
    config = _make_config(subscribe_topics=("topic/a",))
    subscriber = MqttRecordSubscriber(config, on_record=lambda r: None)

    assert subscriber.client is subscriber._client


def test_parse_and_filter_records_extracts_event_time_from_dx1_timestamp():
    payload = (
        b'{"records": [{"timestamp": "2026-10-01T00:47:01.718651520+0000", '
        b'"NX5_Axis_X_Data:AxX_Act_Pos": 123.4}]}'
    )

    records = parse_and_filter_records(payload, tags=("NX5_Axis_X_Data:AxX_Act_Pos",))

    assert [r.epoch_ns for r in records] == [parse_dx1_timestamp("2026-10-01T00:47:01.718651520+0000")]


def test_parse_and_filter_records_drops_records_with_unparseable_timestamp_and_reports_them():
    payload = (
        b'{"records": ['
        b'{"timestamp": "not-a-time", "a": 1.0}, '
        b'{"timestamp": "2026-08-03T00:00:00Z", "a": 2.0}]}'
    )
    dropped = []

    records = parse_and_filter_records(payload, tags=("a",), on_invalid_timestamp=lambda: dropped.append(1))

    assert [r.values for r in records] == [{"a": 2.0}]
    assert dropped == [1]


def test_parse_and_filter_records_keeps_every_record_of_a_batch_message():
    batch = ", ".join(
        '{"timestamp": "2026-10-01T00:47:0%d.%d00000000+0000", "a": %d}' % (s, d, s * 10 + d)
        for s in (1, 2)
        for d in range(5)
    )
    payload = ('{"records": [%s]}' % batch).encode("utf-8")

    records = parse_and_filter_records(payload, tags=("a",))

    assert [r.values["a"] for r in records] == [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    assert [r.epoch_ns for r in records] == sorted(r.epoch_ns for r in records)
