from __future__ import annotations

import json


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
        self._client.publish(self._publish_topic, payload)
