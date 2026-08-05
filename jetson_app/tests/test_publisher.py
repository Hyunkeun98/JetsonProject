import json

from jetson_app.publisher import ResultPublisher


class _FakeClient:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload):
        self.published.append((topic, payload))


def test_publish_sends_expected_schema_to_configured_topic():
    client = _FakeClient()
    publisher = ResultPublisher(client=client, publish_topic="jetson/line_A/anomaly")
    publisher.publish(
        timestamp="2026-08-06T00:00:00+00:00",
        anomaly_score=4.2,
        alarm=True,
        top_deviant_tag="PLC_Collector_Actuator_1:AirBlower.Cmd[0]",
    )
    assert len(client.published) == 1
    topic, payload = client.published[0]
    assert topic == "jetson/line_A/anomaly"
    data = json.loads(payload)
    assert data == {
        "records": [
            {
                "timestamp": "2026-08-06T00:00:00+00:00",
                "jetson:anomaly_score": 4.2,
                "jetson:alarm": True,
                "jetson:top_deviant_tag": "PLC_Collector_Actuator_1:AirBlower.Cmd[0]",
            }
        ]
    }


def test_publish_multiple_calls_each_send_one_message():
    client = _FakeClient()
    publisher = ResultPublisher(client=client, publish_topic="t")
    publisher.publish("ts1", 1.0, False, "tag1")
    publisher.publish("ts2", 2.0, True, "tag2")
    assert len(client.published) == 2
    assert client.published[0][0] == "t"
    assert client.published[1][0] == "t"
