from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .config import ConfigError, load_equipment_config
from .scores import ScoresError, read_scores_csv
from .threshold import DEFAULT_GRID, DEFAULT_MARGIN, ThresholdError, analyze, format_report


def _number_list(text: str, cast):
    try:
        return tuple(cast(part) for part in text.split(",") if part.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"쉼표로 구분한 숫자여야 합니다: {text}") from e


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="jetson-replay score가 만든 점수 CSV로 알람 임계값 후보 표와 추천값을 만든다"
    )
    parser.add_argument("--scores", required=True, help="점수 CSV (jetson-replay score의 출력)")
    parser.add_argument("--config", help="설정 YAML. 있으면 alarm.confirm_steps와 현재 임계값을 참고한다")
    parser.add_argument("--confirm", type=lambda t: _number_list(t, int), help="연속 확정 횟수 목록 (예: 3,5,10)")
    parser.add_argument("--grid", type=lambda t: _number_list(t, float), default=DEFAULT_GRID, help="후보 임계값 목록")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="정상만 있을 때 곱할 여유 (기본 1.2)")
    parser.add_argument("--max-alarms-per-day", type=float, help="정상만 있을 때 허용할 하루 알람 건수")
    parser.add_argument("--report", default="threshold_report.json", help="JSON 보고서 경로")
    args = parser.parse_args(argv)

    try:
        current = None
        confirms = args.confirm
        if args.config:
            config = load_equipment_config(args.config)
            current = (config.alarm.threshold, config.alarm.confirm_steps)
            if not confirms:
                confirms = (config.alarm.confirm_steps,)
        confirms = confirms or (3,)
        if any(c <= 0 for c in confirms):
            raise ThresholdError("--confirm 값은 1 이상이어야 합니다")
        rows, group_names = read_scores_csv(args.scores)
        analyses = [
            analyze(rows, group_names, c, grid=args.grid, margin=args.margin, max_alarms_per_day=args.max_alarms_per_day)
            for c in confirms
        ]
    except (ConfigError, ScoresError, ThresholdError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)

    if current is not None:
        print(f"현재 설정: threshold={current[0]}, confirm_steps={current[1]}")
    for analysis in analyses:
        print(format_report(analysis))
        print()
    print("추천값은 제안입니다. 표를 보고 직접 정해 설정의 alarm.threshold에 적고 앱을 재시작하세요.")
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in analyses], f, ensure_ascii=False, indent=2)
    print(f"보고서: {args.report}")


if __name__ == "__main__":
    main()
