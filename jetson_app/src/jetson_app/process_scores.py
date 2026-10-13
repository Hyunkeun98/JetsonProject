from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from pathlib import Path

from .mqtt_subscriber import parse_and_filter_records
from .scores import ScoreRow
from .timeparse import format_epoch_ns

DEFAULT_PROC_PATTERN = "NX5_ProcStart:U{}_ProcStart"
DEFAULT_PROCESSES = 11


class ProcessScoreError(ValueError):
    pass


@dataclass(frozen=True)
class ProcessWindow:
    cycle: int  # 1부터
    process: int  # 0부터
    start_ns: int
    end_ns: int


def _read_edges(data_dir: str | Path, tags: tuple) -> dict:
    """태그 -> [(켜진 시각 ns, 꺼진 시각 ns)]. 꺼지지 않고 끝난 마지막 구간은 버린다."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise ProcessScoreError(f"데이터 폴더를 찾을 수 없습니다: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix in (".json", ".jsonl"))
    samples: dict = {t: [] for t in tags}
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for record in parse_and_filter_records(line.encode("utf-8"), tags):
                    for tag, value in record.values.items():
                        samples[tag].append((record.epoch_ns, value))
    windows: dict = {}
    for tag, items in samples.items():
        items.sort()
        out, start = [], None
        for ns, value in items:
            on = value is not None and value >= 0.5
            if on and start is None:
                start = ns
            elif not on and start is not None:
                out.append((start, ns))
                start = None
        windows[tag] = out
    return windows


def process_windows(
    data_dir: str | Path, pattern: str = DEFAULT_PROC_PATTERN, processes: int = DEFAULT_PROCESSES
) -> list:
    """공정 0의 시작 신호가 켜질 때마다 새 사이클로 보고, 각 공정 신호가 켜진 구간을 그 사이클에 배정한다."""
    tags = tuple(pattern.format(n) for n in range(processes))
    edges = _read_edges(data_dir, tags)
    cycle_starts = [start for start, _ in edges[tags[0]]]
    if not cycle_starts:
        raise ProcessScoreError("공정 0의 시작 신호가 한 번도 켜지지 않았습니다. 태그 이름(--proc-pattern)을 확인하세요")
    result = []
    for n, tag in enumerate(tags):
        for start, end in edges[tag]:
            cycle = bisect.bisect_right(cycle_starts, start)
            if cycle == 0:
                continue  # 첫 사이클 시작 전에 켜진 신호
            result.append(ProcessWindow(cycle=cycle, process=n, start_ns=start, end_ns=end))
    result.sort(key=lambda w: (w.cycle, w.process, w.start_ns))
    return result


def _critical(scores: list, resets: list, confirm: int):
    best, run = None, []
    for score, reset in zip(scores, resets):
        if reset:
            run = []
        run.append(score)
        if len(run) >= confirm:
            value = min(run[-confirm:])
            if best is None or value > best:
                best = value
    return best


def score_windows(
    rows: list,
    windows: list,
    group_names: tuple,
    confirm: int,
    pad_ns: int = 0,
) -> list:
    ordered = sorted(rows, key=lambda r: r.epoch_ns)
    epochs = [r.epoch_ns for r in ordered]
    out = []
    for w in windows:
        lo = bisect.bisect_left(epochs, w.start_ns - pad_ns)
        hi = bisect.bisect_left(epochs, w.end_ns + pad_ns)
        part = ordered[lo:hi]
        entry = {
            "cycle": w.cycle,
            "process": w.process,
            "start_ns": w.start_ns,
            "end_ns": w.end_ns,
            "rows": len(part),
            "anomaly": sorted({r.segment for r in part if r.kind == "anomaly"}),
            "groups": {},
        }
        for g in group_names:
            items = [(r.groups[g], r.reset) for r in part if g in r.groups]
            if not items:
                continue
            best = max(items, key=lambda it: it[0][0])
            entry["groups"][g] = {
                "max": best[0][0],
                "top": best[0][1],
                "crit": _critical([it[0][0] for it in items], [it[1] for it in items], confirm),
            }
        entry["overall"] = max((v["max"] for v in entry["groups"].values()), default=None)
        out.append(entry)
    return out


def write_process_csv(path: str | Path, entries: list, group_names: tuple, confirm: int) -> None:
    header = ["cycle", "process", "start", "end", "rows", "anomaly", "overall_max"]
    for g in group_names:
        header += [f"{g}_max", f"{g}_c{confirm}", f"{g}_top"]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for e in entries:
            line = [
                e["cycle"], e["process"], format_epoch_ns(e["start_ns"]), format_epoch_ns(e["end_ns"]),
                e["rows"], "+".join(e["anomaly"]), "" if e["overall"] is None else f"{e['overall']:.4f}",
            ]
            for g in group_names:
                v = e["groups"].get(g)
                if v is None:
                    line += ["", "", ""]
                else:
                    line += [f"{v['max']:.4f}", "" if v["crit"] is None else f"{v['crit']:.4f}", v["top"]]
            writer.writerow(line)


def format_pivot(entries: list, group_names: tuple, processes: int) -> str:
    """사이클(행) x 공정(열) 표. 칸은 그 공정 구간의 최대 점수이고, 이상 구간과 겹치면 *를 붙인다."""
    cycles = sorted({e["cycle"] for e in entries})
    lines = []
    for g in (None,) + tuple(group_names):
        title = "전체(그룹 중 최대)" if g is None else f"그룹 {g}"
        lines.append(f"\n[{title}]  공정별 최대 점수 (사이클 x 공정)")
        lines.append("사이클 " + "".join(f"{'P' + str(n):>8}" for n in range(processes)))
        for c in cycles:
            cells: dict = {}
            for e in entries:
                if e["cycle"] != c or e["rows"] == 0 or e["process"] in cells:
                    continue
                value = e["overall"] if g is None else (e["groups"].get(g) or {}).get("max")
                if value is None:
                    continue
                cells[e["process"]] = f"{value:.1f}" + ("*" if e["anomaly"] else "")
            lines.append(f"{c:>6} " + "".join(f"{cells.get(n, '-'):>8}" for n in range(processes)))
    lines.append("\n'-'는 그 공정 구간에 점수가 없는 경우, '*'는 이상 구간과 겹친 경우입니다.")
    return "\n".join(lines)
