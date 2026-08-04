from __future__ import annotations

import argparse
import sys

from .calibration import CalibrationSample
from .config import ConfigError, load_equipment_config
from .pipeline import build_pipeline


def _placeholder_train_fn(samples: list[CalibrationSample]) -> None:
    print(
        f"[학습] {len(samples)}개 샘플로 학습을 실행합니다 "
        "(모델 미연결 — 다음 계획에서 실제 GRU 학습으로 교체 예정)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DX1 -> Jetson MQTT 데이터 파이프라인")
    parser.add_argument("--config", required=True, help="설비 config YAML 경로")
    parser.add_argument("--host", required=True, help="MQTT 브로커 호스트 (Jetson 자신의 IP 또는 localhost)")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument(
        "--calibration-dir",
        default="calibration_data",
        help="캘리브레이션 버퍼 파일을 저장할 디렉터리 (기본: calibration_data)",
    )
    args = parser.parse_args()

    try:
        config = load_equipment_config(args.config)
        pipeline = build_pipeline(
            config=config,
            calibration_dir=args.calibration_dir,
            train_fn=_placeholder_train_fn,
        )
        pipeline.mqtt_subscriber.connect(args.host, args.port)
    except (ConfigError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)

    pipeline.snapshotter.start()
    print(
        f"[{config.equipment_id}] {len(config.subscribe_topics)}개 토픽 구독 시작 "
        f"({args.host}:{args.port}), 캘리브레이션 데이터: {args.calibration_dir}"
    )
    try:
        pipeline.mqtt_subscriber.loop_forever()
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        pipeline.snapshotter.stop()


if __name__ == "__main__":
    main()
