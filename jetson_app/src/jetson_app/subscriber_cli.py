from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_equipment_config
from .pipeline import build_pipeline
from .score_history import DEFAULT_HISTORY_SECONDS
from .training import make_train_fn, model_artifact_path
from .web_server import ScoreWebServer


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
    parser.add_argument(
        "--model-dir",
        default="model_data",
        help="학습된 모델 아티팩트를 저장할 디렉터리 (기본: model_data)",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=None,
        help="이 포트로 점수 그래프 웹 화면을 연다 (예: 8080). 주지 않으면 웹 화면을 켜지 않는다",
    )
    parser.add_argument(
        "--web-host",
        default="127.0.0.1",
        help="웹 화면을 열 주소 (기본: 127.0.0.1, 이 장비에서만). 같은 네트워크의 다른 PC에서 보려면 0.0.0.0",
    )
    args = parser.parse_args()

    try:
        config = load_equipment_config(args.config)
        model_path = model_artifact_path(args.model_dir, config.equipment_id)
        # 잘못된 --model-dir이 학습 완료 후 save_artifact 안(예외를 삼키는 MQTT 명령
        # 콜백)에서야 드러나지 않도록, --calibration-dir과 동일하게 기동 시점에
        # 디렉터리를 만들어 OSError를 아래 핸들러로 흘린다.
        model_path.parent.mkdir(parents=True, exist_ok=True)
        train_fn = make_train_fn(
            tags=config.tags,
            window_size=config.window_size,
            model_path=model_path,
            resample_interval_ms=config.resample_interval_ms,
            groups=config.group_specs(),
            epochs=config.training.epochs,
            max_training_samples=config.training.max_samples,
        )
        pipeline = build_pipeline(
            config=config,
            calibration_dir=args.calibration_dir,
            model_dir=args.model_dir,
            train_fn=train_fn,
            history_seconds=DEFAULT_HISTORY_SECONDS if args.web_port is not None else 0,
        )
        pipeline.mqtt_subscriber.connect(args.host, args.port)
    except (ConfigError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)

    web_server = _start_web(args, config, pipeline)
    pipeline.snapshotter.start()
    print(
        f"[{config.equipment_id}] {len(config.subscribe_topics)}개 토픽 구독 시작 "
        f"({args.host}:{args.port}), 캘리브레이션 데이터: {args.calibration_dir}, "
        f"모델 저장 위치: {args.model_dir}"
    )
    try:
        pipeline.mqtt_subscriber.loop_forever()
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        pipeline.snapshotter.stop()
        if web_server is not None:
            web_server.stop()


def _start_web(args, config, pipeline):
    """--web-port가 있으면 점수 그래프 웹 서버를 연다. 열지 못해도 앱은 웹 없이 계속 돈다."""
    if args.web_port is None or pipeline.history is None:
        return None
    manager = pipeline.calibration_manager

    def meta() -> dict:
        return {
            "equipment_id": config.equipment_id,
            "groups": list(pipeline.history.group_names),
            "threshold": config.alarm.threshold,
            "confirm_steps": config.alarm.confirm_steps,
            "interval_ms": config.resample_interval_ms,
            "history_seconds": DEFAULT_HISTORY_SECONDS,
            "state": manager.state.value,
        }

    try:
        server = ScoreWebServer(pipeline.history, meta, host=args.web_host, port=args.web_port)
    except OSError as e:
        print(f"[web] 웹 화면을 열지 못했습니다(앱은 계속 실행됩니다): {e}", file=sys.stderr)
        return None
    server.start()
    print(f"[web] 점수 그래프: http://{args.web_host}:{server.port}/")
    if args.web_host not in ("127.0.0.1", "localhost", "::1"):
        print("[web] 경고: 이 화면에는 로그인이 없습니다. 같은 네트워크의 누구나 볼 수 있습니다", file=sys.stderr)
    return server


if __name__ == "__main__":
    main()
