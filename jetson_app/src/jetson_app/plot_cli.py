from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_equipment_config
from .dashboard_page import render_dashboard
from .debounce import DEFAULT_CONFIRM_TICKS, DEFAULT_THRESHOLD
from .scores import ScoresError, ScoreRow, read_scores_csv
from .threshold import ThresholdError, analyze


def build_file_boot(
    rows: list[ScoreRow],
    group_names: tuple[str, ...],
    equipment_id: str,
    threshold: float,
    confirm: int,
) -> dict:
    """점수 행을 화면(파일 모드)이 읽는 JSON 구조로 바꾼다."""
    tag_index: dict[str, int] = {}
    tag_table: list[str] = []
    scores: dict[str, list] = {g: [] for g in group_names}
    tops: dict[str, list] = {g: [] for g in group_names}
    segments: list[dict] = []
    for i, row in enumerate(rows):
        if segments and segments[-1]["kind"] == row.kind and segments[-1]["name"] == row.segment:
            segments[-1]["i1"] = i + 1
        else:
            segments.append({"kind": row.kind, "name": row.segment, "i0": i, "i1": i + 1})
        for g in group_names:
            entry = row.groups.get(g)
            if entry is None:
                scores[g].append(None)
                tops[g].append(-1)
                continue
            score, top = entry
            scores[g].append(round(score, 4))
            if top not in tag_index:
                tag_index[top] = len(tag_table)
                tag_table.append(top)
            tops[g].append(tag_index[top])

    recommended = None
    try:
        analysis = analyze(rows, group_names, confirm)
        if analysis.recommended is not None:
            recommended = {"confirm": confirm, "value": analysis.recommended}
    except ThresholdError:
        pass  # 정상 구간이 없으면 추천 없이 그래프만 보여 준다

    return {
        "mode": "file",
        "equipment_id": equipment_id,
        "groups": list(group_names),
        "threshold": threshold,
        "confirm": confirm,
        "recommended": recommended,
        "t": [row.epoch_ns // 1_000_000 for row in rows],
        "reset": [1 if row.reset else 0 for row in rows],
        "scores": scores,
        "tops": tops,
        "tagTable": tag_table,
        "segments": segments,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="점수 CSV(jetson-replay score의 출력)를 서버 없이 열 수 있는 그래프 HTML 한 장으로 만든다"
    )
    parser.add_argument("--scores", required=True, help="점수 CSV")
    parser.add_argument("--config", help="설정 YAML (설비 이름과 현재 alarm 설정을 가져온다)")
    parser.add_argument("--threshold", type=float, help="초기 임계값 (기본: 설정의 alarm.threshold, 없으면 3.0)")
    parser.add_argument("--confirm", type=int, help="초기 연속 확정 횟수 (기본: 설정, 없으면 3)")
    parser.add_argument("--out", default="scores_plot.html", help="출력 HTML (기본: scores_plot.html)")
    args = parser.parse_args(argv)

    try:
        equipment_id = Path(args.scores).stem
        threshold, confirm = DEFAULT_THRESHOLD, DEFAULT_CONFIRM_TICKS
        if args.config:
            config = load_equipment_config(args.config)
            equipment_id = config.equipment_id
            threshold, confirm = config.alarm.threshold, config.alarm.confirm_steps
        if args.threshold is not None:
            threshold = args.threshold
        if args.confirm is not None:
            confirm = args.confirm
        if threshold <= 0 or confirm <= 0:
            raise ThresholdError("--threshold와 --confirm은 0보다 커야 합니다")
        rows, group_names = read_scores_csv(args.scores)
        if not rows:
            raise ScoresError("점수 파일에 행이 없습니다")
        boot = build_file_boot(rows, group_names, equipment_id, threshold, confirm)
        Path(args.out).write_text(render_dashboard(boot), encoding="utf-8")
    except (ConfigError, ScoresError, ThresholdError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)
    print(f"그래프 {args.out} ({len(rows)}행). 브라우저로 열어 보세요")


if __name__ == "__main__":
    main()
