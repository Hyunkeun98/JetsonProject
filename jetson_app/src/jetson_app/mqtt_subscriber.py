from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import paho.mqtt.client as mqtt

from .config import EquipmentConfig


@dataclass(frozen=True)
class Record:
    timestamp: str
    values: dict[str, float]


def parse_and_filter_records(payload: bytes, tags: tuple[str, ...]) -> list[Record]:
    tag_set = set(tags)
    try:
        data = json.loads(payload)
    except ValueError:
        # json.JSONDecodeError (malformed JSON) and UnicodeDecodeError
        # (non-UTF8 bytes, raised while json.loads decodes payload) are
        # both ValueError subclasses.
        return []

    if not isinstance(data, dict):
        return []

    records = data.get("records", [])
    if not isinstance(records, list):
        return []

    result = []
    for raw in records:
        if not isinstance(raw, dict):
            continue
        timestamp = raw.get("timestamp")
        values = {k: v for k, v in raw.items() if k in tag_set}
        if timestamp is not None and values:
            result.append(Record(timestamp=timestamp, values=values))
    return result


class MqttRecordSubscriber:
    def __init__(
        self,
        config: EquipmentConfig,
        on_record: Callable[[Record], None],
    ) -> None:
        self._config = config
        self._on_record = on_record
        self._client = mqtt.Client()
        self._client.on_connect = self._handle_connect
        self._client.on_message = self._handle_message

    def connect(self, host: str, port: int = 1883) -> None:
        self._client.connect(host, port)

    def loop_forever(self) -> None:
        self._client.loop_forever()

    def _handle_connect(self, client, userdata, flags, rc):
        if rc != 0:
            print(
                f"[{self._config.equipment_id}] MQTT 연결 실패 (rc={rc}) "
                "— 브로커 인증/ACL 설정을 확인하세요"
            )
            return
        client.subscribe(self._config.subscribe_topic)

    def _handle_message(self, client, userdata, msg):
        for record in parse_and_filter_records(msg.payload, self._config.tags):
            self._on_record(record)
