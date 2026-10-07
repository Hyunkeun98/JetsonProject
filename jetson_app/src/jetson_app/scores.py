from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .timeparse import format_epoch_ns

_FIXED_COLUMNS = ("epoch_ns", "time", "kind", "segment", "reset")


class ScoresError(ValueError):
    pass


@dataclass(frozen=True)
class ScoreRow:
    """검증/이상 구간의 스텝 하나. reset이 True면 이 행부터 새 윈도우(디바운서 리셋)다.
    groups: 그룹 이름 -> (점수, 원인 태그). 점수가 없는 그룹은 들어 있지 않다."""

    epoch_ns: int
    kind: str
    segment: str
    reset: bool
    groups: dict


def write_scores_csv(path: str | Path, rows: list[ScoreRow], group_names: tuple[str, ...]) -> None:
    header = list(_FIXED_COLUMNS)
    for name in group_names:
        header += [f"{name}_score", f"{name}_top"]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            line = [row.epoch_ns, format_epoch_ns(row.epoch_ns), row.kind, row.segment, int(row.reset)]
            for name in group_names:
                if name in row.groups:
                    score, top = row.groups[name]
                    line += [repr(float(score)), top]
                else:
                    line += ["", ""]
            writer.writerow(line)


def read_scores_csv(path: str | Path) -> tuple[list[ScoreRow], tuple[str, ...]]:
    try:
        f = Path(path).open("r", encoding="utf-8", newline="")
    except OSError as e:
        raise ScoresError(f"점수 파일을 읽을 수 없습니다: {e}") from e
    with f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or tuple(header[: len(_FIXED_COLUMNS)]) != _FIXED_COLUMNS:
            raise ScoresError("점수 파일의 머리글이 jetson-replay score가 만든 형식이 아닙니다")
        group_columns = header[len(_FIXED_COLUMNS) :]
        if not group_columns or len(group_columns) % 2 or not all(
            group_columns[i].endswith("_score") and group_columns[i + 1].endswith("_top")
            for i in range(0, len(group_columns), 2)
        ):
            raise ScoresError("점수 파일의 그룹 열(<그룹>_score, <그룹>_top)이 올바르지 않습니다")
        group_names = tuple(group_columns[i][: -len("_score")] for i in range(0, len(group_columns), 2))
        rows = []
        for line_no, line in enumerate(reader, start=2):
            try:
                groups = {}
                for index, name in enumerate(group_names):
                    score_text = line[len(_FIXED_COLUMNS) + 2 * index]
                    if score_text != "":
                        groups[name] = (float(score_text), line[len(_FIXED_COLUMNS) + 2 * index + 1])
                rows.append(
                    ScoreRow(
                        epoch_ns=int(line[0]),
                        kind=line[2],
                        segment=line[3],
                        reset=line[4] == "1",
                        groups=groups,
                    )
                )
            except (ValueError, IndexError) as e:
                raise ScoresError(f"점수 파일 {line_no}번째 줄을 읽을 수 없습니다: {e}") from e
    return rows, group_names
