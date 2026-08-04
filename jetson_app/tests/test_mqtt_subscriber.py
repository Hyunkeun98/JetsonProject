from jetson_app.mqtt_subscriber import Record, parse_and_filter_records


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
