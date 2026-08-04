from __future__ import annotations

import argparse

from .config import load_equipment_config
from .mqtt_subscriber import MqttRecordSubscriber, Record


def _print_record(record: Record) -> None:
    print(f"[{record.timestamp}] {record.values}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DX1 -> Jetson MQTT 수신 확인용 구독자")
    parser.add_argument("--config", required=True, help="설비 config YAML 경로")
    parser.add_argument("--host", required=True, help="MQTT 브로커 호스트 (Jetson 자신의 IP 또는 localhost)")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    config = load_equipment_config(args.config)
    subscriber = MqttRecordSubscriber(config, on_record=_print_record)
    subscriber.connect(args.host, args.port)

    print(f"[{config.equipment_id}] '{config.subscribe_topic}' 구독 시작 ({args.host}:{args.port})")
    subscriber.loop_forever()


if __name__ == "__main__":
    main()
