import json

import pytest

from jetson_app.process_scores import (
    ProcessScoreError,
    format_pivot,
    process_windows,
    score_windows,
    write_process_csv,
)
from jetson_app.process_scores_cli import main as procscores_main
from jetson_app.scores import ScoreRow, write_scores_csv

SEC = 1_000_000_000
BASE = 1_790_000_000  # 임의의 epoch 초


def _stamp(sec):
    from datetime import datetime, timezone

    return datetime.fromtimestamp(BASE + sec, timezone.utc).isoformat()


def _write_proc_file(tmp_path, schedule, processes=3, cycle_len=100):
    """schedule: (사이클 번호 1.., 공정 번호, 켜진 초, 꺼진 초) 목록. 나머지는 0."""
    events = []
    for _c, n, on, off in schedule:
        events.append((on, n, 1))
        events.append((off, n, 0))
    records = []
    state = {n: 0 for n in range(processes)}
    for t, n, v in sorted(events):
        state[n] = v
        record = {"timestamp": _stamp(t)}
        record.update({f"NX5_ProcStart:U{k}_ProcStart": state[k] for k in range(processes)})
        records.append(record)
    (tmp_path / "DX_ProcStart.json").write_text(json.dumps({"records": records}) + "\n", encoding="utf-8")


def _rows(score_by_second, kind="normal_eval", name="n"):
    return [
        ScoreRow((BASE + s) * SEC, kind, name, i == 0, {"g": (v, "tag")})
        for i, (s, v) in enumerate(sorted(score_by_second.items()))
    ]


def test_windows_follow_process_zero_to_split_cycles(tmp_path):
    schedule = [(1, 0, 10, 12), (1, 1, 20, 23), (2, 0, 110, 112), (2, 1, 120, 123), (2, 2, 130, 131)]
    _write_proc_file(tmp_path, schedule)
    windows = process_windows(tmp_path, processes=3)
    assert [(w.cycle, w.process) for w in windows] == [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
    assert windows[1].start_ns == (BASE + 20) * SEC and windows[1].end_ns == (BASE + 23) * SEC


def test_signal_before_the_first_cycle_start_is_ignored(tmp_path):
    _write_proc_file(tmp_path, [(0, 1, 5, 6), (1, 0, 10, 12), (1, 1, 20, 23)])
    windows = process_windows(tmp_path, processes=3)
    assert [(w.cycle, w.process) for w in windows] == [(1, 0), (1, 1)]


def test_missing_process_zero_signal_is_an_error(tmp_path):
    _write_proc_file(tmp_path, [(1, 1, 20, 23)])
    with pytest.raises(ProcessScoreError, match="공정 0"):
        process_windows(tmp_path, processes=3)
    with pytest.raises(ProcessScoreError, match="폴더"):
        process_windows(tmp_path / "nope", processes=3)


def test_window_scores_use_rows_inside_each_window(tmp_path):
    _write_proc_file(tmp_path, [(1, 0, 10, 12), (1, 1, 20, 23)])
    windows = process_windows(tmp_path, processes=3)
    rows = _rows({s: 1.0 for s in range(0, 40)})
    rows[21 - 0] = ScoreRow(rows[21].epoch_ns, "normal_eval", "n", False, {"g": (9.0, "peak_tag")})
    entries = score_windows(rows, windows, ("g",), confirm=2)
    p0, p1 = entries
    assert p0["rows"] == 2 and p0["groups"]["g"]["max"] == 1.0
    assert p1["rows"] == 3 and p1["groups"]["g"]["max"] == 9.0 and p1["groups"]["g"]["top"] == "peak_tag"
    assert p1["groups"]["g"]["crit"] == 1.0  # 9 하나만 높아 연속 2번은 안 된다
    assert p1["overall"] == 9.0


def test_anomaly_rows_flag_the_window_and_empty_windows_have_no_score(tmp_path):
    _write_proc_file(tmp_path, [(1, 0, 10, 12), (1, 1, 20, 23)])
    windows = process_windows(tmp_path, processes=3)
    rows = _rows({10: 1.0, 11: 1.0}) + _rows({20: 5.0, 21: 5.0, 22: 5.0}, kind="anomaly", name="touch")
    p0, p1 = score_windows(rows, windows, ("g",), 2)
    assert p0["anomaly"] == [] and p1["anomaly"] == ["touch"]
    empty = score_windows(_rows({50: 1.0}), windows, ("g",), 2)
    assert all(e["rows"] == 0 and e["overall"] is None for e in empty)
    assert "-" in format_pivot(empty, ("g",), 3)


def test_pad_widens_the_window(tmp_path):
    _write_proc_file(tmp_path, [(1, 0, 10, 12)])
    windows = process_windows(tmp_path, processes=3)
    rows = _rows({s: float(s) for s in range(0, 30)})
    narrow = score_windows(rows, windows, ("g",), 1)[0]
    wide = score_windows(rows, windows, ("g",), 1, pad_ns=3 * SEC)[0]
    assert wide["rows"] == narrow["rows"] + 6 and wide["groups"]["g"]["max"] > narrow["groups"]["g"]["max"]


def test_cli_writes_csv_and_prints_a_cycle_by_process_table(tmp_path, capsys):
    _write_proc_file(tmp_path, [(1, 0, 10, 12), (1, 1, 20, 23), (2, 0, 110, 112), (2, 1, 120, 123)])
    scores = tmp_path / "scores.csv"
    write_scores_csv(scores, _rows({s: 2.0 for s in range(0, 140)}), ("g",))
    out = tmp_path / "ps.csv"
    procscores_main(["--scores", str(scores), "--data-dir", str(tmp_path), "--processes", "3", "--out", str(out)])
    text = capsys.readouterr().out
    assert "P0" in text and "P2" in text and "2.0" in text
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("cycle,process") and len(lines) == 1 + 4


def test_cli_errors_exit_with_code_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        procscores_main(["--scores", str(tmp_path / "x.csv"), "--data-dir", str(tmp_path)])
    assert info.value.code == 2 and "error:" in capsys.readouterr().err


def test_write_process_csv_leaves_blank_cells_for_missing_groups(tmp_path):
    entries = [{"cycle": 1, "process": 0, "start_ns": BASE * SEC, "end_ns": (BASE + 1) * SEC, "rows": 0,
                "anomaly": [], "groups": {}, "overall": None}]
    path = tmp_path / "o.csv"
    write_process_csv(path, entries, ("g",), 3)
    assert path.read_text(encoding="utf-8").splitlines()[1].endswith(",,,,")
