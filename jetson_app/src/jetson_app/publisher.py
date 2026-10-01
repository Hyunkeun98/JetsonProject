from __future__ import annotations

import json
from dataclasses import dataclass

import paho.mqtt.client as mqtt


@dataclass(frozen=True)
class GroupOutput:
    """한 그룹의 발행 값(점수, 디바운스를 거친 알람, 원인 태그)."""

    score: float
    alarm: bool
    top_tag: str


class ResultPublisher:
    """이상 점수/알람/최대 기여 태그를 상위 아키텍처 문서 3절의 발행 스키마로
    설비 config의 publish_topic에 MQTT 발행한다."""

    def __init__(self, client, publish_topic: str) -> None:
        self._client = client
        self._publish_topic = publish_topic

    def publish(
        self,
        timestamp: str,
        anomaly_score: float,
        alarm: bool,
        top_deviant_tag: str,
        groups: dict[str, GroupOutput] | None = None,
    ) -> None:
        record = {
            "timestamp": timestamp,
            "jetson:anomaly_score": anomaly_score,
            "jetson:alarm": alarm,
            "jetson:top_deviant_tag": top_deviant_tag,
        }
        # 그룹이 하나뿐이면(평면 tags 설정) 기존 발행 형식을 그대로 유지한다.
        if groups and len(groups) > 1:
            for name, group in groups.items():
                record[f"jetson:{name}:score"] = group.score
                record[f"jetson:{name}:alarm"] = group.alarm
                record[f"jetson:{name}:top_tag"] = group.top_tag
        payload = json.dumps({"records": [record]})
        # QoS 0(paho 기본)에서는 재연결 중 발행이 조용히 버려진다(rc=MQTT_ERR_NO_CONN).
        # 알람 경로가 흔적 없이 유실되지 않도록 rc를 확인해 로그를 남긴다.
        result = self._client.publish(self._publish_topic, payload)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[ResultPublisher] 발행 실패 (rc={result.rc}): topic={self._publish_topic}")
