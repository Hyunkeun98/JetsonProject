from __future__ import annotations

import argparse
import sys

from .process_scores import (
    DEFAULT_PROC_PATTERN,
    DEFAULT_PROCESSES,
    ProcessScoreError,
    format_pivot,
    process_windows,
    score_windows,
    write_process_csv,
)
from .scores import ScoresError, read_scores_csv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="점수 CSV와 원본 JSON의 공정 신호(ProcStart)로, 사이클마다 공정별 점수를 표와 CSV로 만든다"
    )
    parser.add_argument("--scores", required=True, help="점수 CSV (jetson-replay score의 출력)")
    parser.add_argument("--data-dir", required=True, help="공정 신호가 든 DX1 JSON 폴더")
    parser.add_argument("--confirm", type=int, default=3, help="연속 확정 횟수 (기본 3)")
    parser.add_argument("--pad-s", type=float, default=0.0, help="공정 구간 앞뒤로 넓힐 초 (기본 0)")
    parser.add_argument("--processes", type=int, default=DEFAULT_PROCESSES, help="공정 수 (기본 11: 0~10)")
    parser.add_argument("--proc-pattern", default=DEFAULT_PROC_PATTERN, help="공정 신호 태그 패턴, {}에 번호가 들어간다")
    parser.add_argument("--out", default="process_scores.csv", help="출력 CSV")
    args = parser.parse_args(argv)

    try:
        if args.confirm <= 0 or args.processes <= 0 or args.pad_s < 0:
            raise ProcessScoreError("--confirm, --processes는 1 이상, --pad-s는 0 이상이어야 합니다")
        rows, group_names = read_scores_csv(args.scores)
        windows = process_windows(args.data_dir, args.proc_pattern, args.processes)
        entries = score_windows(rows, windows, group_names, args.confirm, int(args.pad_s * 1e9))
        write_process_csv(args.out, entries, group_names, args.confirm)
    except (ScoresError, ProcessScoreError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)
    print(format_pivot(entries, group_names, args.processes))
    scored = sum(1 for e in entries if e["rows"])
    print(f"\n공정 구간 {len(entries)}개 중 점수가 있는 구간 {scored}개. CSV: {args.out}")


if __name__ == "__main__":
    main()
