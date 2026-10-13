from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_equipment_config
from .scores import ScoresError, read_scores_csv
from .segments import SegmentsError, load_segments
from .tag_select import TagSelectError, format_selection, select_tags


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="태그별 점수가 든 점수 CSV로, 이상을 가장 잘 갈라 내는 점수 태그 조합을 고른다 "
        "(모델 재학습 없이 설정의 score_exclude_tags로 적용)"
    )
    parser.add_argument("--scores", required=True, help="점수 CSV (jetson-replay score의 출력)")
    parser.add_argument("--config", required=True, help="설정 YAML (그룹과 태그를 가져온다)")
    parser.add_argument(
        "--segments",
        help="구간 YAML. 이상 구간에 groups: [그룹]을 적으면 그 그룹의 점수로만 평가한다",
    )
    parser.add_argument("--confirm", type=int, help="연속 확정 횟수 (기본: 설정의 alarm.confirm_steps)")
    parser.add_argument("--top", type=int, default=4, help="그룹마다 보여 줄 상위 조합 수")
    args = parser.parse_args(argv)

    try:
        config = load_equipment_config(args.config)
        confirm = args.confirm or config.alarm.confirm_steps
        anomaly_groups = {}
        if args.segments:
            anomaly_groups = {s.name: s.groups for s in load_segments(args.segments).anomaly}
        rows, _names = read_scores_csv(args.scores)
        group_tags = {g.name: g.tags for g in config.resolved_groups()}
        if not group_tags:
            raise TagSelectError(
                "이 설정에는 GRU 그룹(groups/tags)이 없습니다. 기준값 이탈(deviation) 신호는 태그 조합을 고르지 않습니다"
            )
        selections = select_tags(rows, group_tags, anomaly_groups, confirm)
    except (ConfigError, ScoresError, SegmentsError, TagSelectError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)
    print(format_selection(selections, confirm, top=args.top))


if __name__ == "__main__":
    main()
