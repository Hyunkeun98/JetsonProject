from __future__ import annotations

import json

import paho.mqtt.client as mqtt


class ResultPublisher:
    """이상 점수/알람/최대 기여 태그를 상위 아키텍처 문서 3절의 발행 스키마로
    설비 config의 publish_topic에 MQTT 발행한다."""

    def __init__(self, client, publish_topic: str) -> None:
        self._client = client
        self._publish_topic = publish_topic

    def publish(
        self, timestamp: str, anomaly_score: float, alarm: bool, top_deviant_tag: str
    ) -> None:
        payload = json.dumps(
            {
                "records": [
                    {
                        "timestamp": timestamp,
                        "jetson:anomaly_score": anomaly_score,
                        "jetson:alarm": alarm,
                        "jetson:top_deviant_tag": top_deviant_tag,
                    }
                ]
            }
        )
        # QoS 0(paho 기본)에서는 재연결 중 발행이 조용히 버려진다(rc=MQTT_ERR_NO_CONN).
        # 알람 경로가 흔적 없이 유실되지 않도록 rc를 확인해 로그를 남긴다.
        result = self._client.publish(self._publish_topic, payload)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[ResultPublisher] 발행 실패 (rc={result.rc}): topic={self._publish_topic}")
