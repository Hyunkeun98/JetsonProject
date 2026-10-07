from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_equipment_config
from .replay import ReplayError, score_from_files, train_from_files
from .scores import write_scores_csv
from .segments import SegmentsError, load_segments


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="저장해 둔 DX1 JSON 파일로 학습하고 구간별 점수를 낸다(MQTT 브로커 불필요)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("train", "학습 구간으로 모델을 학습해 저장한다"),
        ("score", "정상 검증/이상 구간의 점수를 CSV로 저장한다"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, help="설비 config YAML (앱과 같은 파일)")
        p.add_argument("--data-dir", required=True, help="DX1 JSON(.json/.jsonl) 파일이 있는 폴더")
        p.add_argument("--segments", required=True, help="구간 파일 YAML")
        p.add_argument("--model-dir", default="model_data", help="모델 폴더 (기본: model_data)")
        if name == "score":
            p.add_argument("--out", default="scores.csv", help="점수 CSV 경로 (기본: scores.csv)")
    args = parser.parse_args(argv)

    try:
        config = load_equipment_config(args.config)
        segment_set = load_segments(args.segments)
        if args.command == "train":
            print(f"[replay] 학습 중입니다. 데이터가 많으면 수십 분 걸릴 수 있습니다 ({len(segment_set.train)}개 구간)")
            model_path, steps = train_from_files(config, args.data_dir, segment_set, args.model_dir)
            if steps < config.calibration.min_samples:
                print(
                    f"경고: 학습 스텝 {steps}개가 설정의 calibration.min_samples "
                    f"({config.calibration.min_samples})보다 적습니다",
                    file=sys.stderr,
                )
            print(f"[replay] 학습 완료: 스텝 {steps}개 -> {model_path} (상태 MONITORING 기록)")
        else:
            rows, group_names = score_from_files(config, args.data_dir, segment_set, args.model_dir)
            write_scores_csv(args.out, rows, group_names)
            print(f"[replay] 점수 {len(rows)}행 -> {args.out}")
    except (ConfigError, SegmentsError, ReplayError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
