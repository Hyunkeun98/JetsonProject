from __future__ import annotations

import argparse
import sys

from .deviation_suggest import analyze_tags, format_report
from .replay import ReplayError
from .segments import SegmentsError, load_segments


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="정상 데이터를 훑어 기준값 이탈 점수(deviation)에 맞는 태그를 추천한다"
    )
    parser.add_argument("--data-dir", required=True, help="정상 운전을 저장한 DX1 JSON(.json/.jsonl) 폴더")
    parser.add_argument(
        "--segments",
        help="구간 파일 YAML. 있으면 그 train 구간만 쓴다(이상 구간이 섞인 데이터일 때). 없으면 폴더 전체를 쓴다",
    )
    parser.add_argument("--interval-ms", type=int, default=100, help="리샘플 격자 간격(ms, 기본 100)")
    parser.add_argument("--baseline-seconds", type=float, default=2.0, help="기준값을 잡는 직전 구간(초, 기본 2)")
    parser.add_argument("--settle-seconds", type=float, default=1.0, help="판정을 늦추는 시간(초, 기본 1)")
    args = parser.parse_args(argv)
    try:
        segments = None
        if args.segments:
            segments = load_segments(args.segments).train or None
        reports = analyze_tags(
            args.data_dir,
            interval_ms=args.interval_ms,
            baseline_seconds=args.baseline_seconds,
            settle_seconds=args.settle_seconds,
            segments=segments,
        )
    except (SegmentsError, ReplayError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)
    print(format_report(reports))


if __name__ == "__main__":
    main()
