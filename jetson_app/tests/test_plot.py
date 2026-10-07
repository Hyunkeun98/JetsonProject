import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from jetson_app.dashboard_page import render_dashboard
from jetson_app.debounce import Debouncer
from jetson_app.plot_cli import build_file_boot, main as plot_main
from jetson_app.scores import ScoreRow, write_scores_csv
from jetson_app.threshold import _simulate_segment

STEP_NS = 100_000_000
PAGE = Path(__file__).parent.parent / "src" / "jetson_app" / "web" / "dashboard.html"
NODE = shutil.which("node")


def _rows():
    rows = []
    for i in range(30):
        rows.append(ScoreRow(i * STEP_NS, "normal_eval", "n", i == 0, {"a": (1.0 + (i == 20) * 9, "ta"), "b": (0.5, "tb")}))
    for i in range(10):
        rows.append(ScoreRow(10**12 + i * STEP_NS, "anomaly", "x", i == 0, {"a": (20.0, "ta"), "b": (30.0, "tb")}))
    return rows


def test_boot_data_has_scores_segments_and_tag_table():
    boot = build_file_boot(_rows(), ("a", "b"), "eq", threshold=3.0, confirm=3)
    assert boot["mode"] == "file" and boot["groups"] == ["a", "b"]
    assert len(boot["t"]) == 40 and boot["reset"][0] == 1 and boot["reset"][30] == 1 and sum(boot["reset"]) == 2
    assert boot["segments"] == [
        {"kind": "normal_eval", "name": "n", "i0": 0, "i1": 30},
        {"kind": "anomaly", "name": "x", "i0": 30, "i1": 40},
    ]
    assert boot["tagTable"][boot["tops"]["a"][0]] == "ta"
    assert boot["scores"]["a"][20] == 10.0
    assert boot["recommended"] is not None and boot["recommended"]["confirm"] == 3


def test_missing_group_scores_become_null():
    rows = [ScoreRow(0, "normal_eval", "n", True, {"a": (1.0, "ta")}), ScoreRow(STEP_NS, "normal_eval", "n", False, {"a": (1.0, "ta"), "b": (2.0, "tb")})]
    boot = build_file_boot(rows, ("a", "b"), "eq", 3.0, 3)
    assert boot["scores"]["b"] == [None, 2.0] and boot["tops"]["b"][0] == -1


def test_render_escapes_script_breaking_text():
    html = render_dashboard({"mode": "file", "equipment_id": "</script><script>alert(1)</script>"})
    assert "</script><script>alert(1)" not in html
    start = html.index("const BOOT = ") + len("const BOOT = ")
    payload = html[start : html.index(";\n", start)].replace("<\\/", "</")
    assert json.loads(payload)["equipment_id"] == "</script><script>alert(1)</script>"


def test_cli_writes_a_standalone_html(tmp_path, capsys):
    csv_path = tmp_path / "scores.csv"
    write_scores_csv(csv_path, _rows(), ("a", "b"))
    out = tmp_path / "plot.html"
    plot_main(["--scores", str(csv_path), "--out", str(out), "--threshold", "5", "--confirm", "2"])
    text = out.read_text(encoding="utf-8")
    assert '"mode":"file"' in text and '"threshold":5.0' in text and '"confirm":2' in text
    assert "http://" not in text.replace("http://www.w3.org", "") and "cdn" not in text.lower()
    assert "그래프" in capsys.readouterr().out


def test_cli_errors_exit_with_code_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        plot_main(["--scores", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "o.html")])
    assert info.value.code == 2 and "error:" in capsys.readouterr().err


def _script_text():
    return re.search(r"<script>(.*)</script>", PAGE.read_text(encoding="utf-8"), re.S).group(1)


@pytest.mark.skipif(NODE is None, reason="node가 없어 화면 JS 검사를 건너뜀")
def test_dashboard_javascript_is_syntactically_valid(tmp_path):
    script = tmp_path / "page.js"
    script.write_text(_script_text(), encoding="utf-8")
    result = subprocess.run([NODE, "--check", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _core_js():
    text = _script_text()
    return text[text.index("// BEGIN-CORE") : text.index("// END-CORE")]


@pytest.mark.skipif(NODE is None, reason="node가 없어 화면 JS 검사를 건너뜀")
def test_js_alarm_logic_matches_the_python_debouncer(tmp_path):
    rng = random.Random(7)
    cases, expected = [], []
    for _ in range(60):
        n = rng.randint(1, 80)
        resets = [1 if i == 0 or rng.random() < 0.06 else 0 for i in range(n)]
        scores = {
            g: [None if rng.random() < 0.08 else round(rng.uniform(0, 10), 3) for _ in range(n)]
            for g in ("a", "b")
        }
        threshold = round(rng.uniform(1, 9), 1)
        confirm = rng.choice([1, 2, 3, 5])
        rows = [
            ScoreRow(
                i * STEP_NS, "normal_eval", "n", bool(resets[i]),
                {g: (scores[g][i], "t") for g in scores if scores[g][i] is not None},
            )
            for i in range(n)
        ]
        # 앱 규칙을 그대로 따라 한 그룹별 알람 배열(디바운서 직접 사용)
        per_group = {}
        for g in scores:
            deb = Debouncer(threshold=threshold, confirm_ticks=confirm)
            arr = []
            for i in range(n):
                if resets[i]:
                    deb.reset()
                arr.append(0 if scores[g][i] is None else int(deb.update(scores[g][i])))
            per_group[g] = arr
        episodes, _first = _simulate_segment(rows, ("a", "b"), threshold, confirm)
        cases.append({"scores": scores, "resets": resets, "threshold": threshold, "confirm": confirm})
        expected.append({"per_group": per_group, "episodes": episodes})

    driver = _core_js() + """
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = input.map((c) => {
  const per = {};
  for (const g of Object.keys(c.scores)) per[g] = runAlarms(c.scores[g], c.resets, c.threshold, c.confirm);
  const overall = orAlarms(Object.values(per));
  return { per_group: per, episodes: countEpisodes(overall, c.resets, 0, c.resets.length) };
});
process.stdout.write(JSON.stringify(out));
"""
    script = tmp_path / "core.js"
    script.write_text(driver, encoding="utf-8")
    result = subprocess.run([NODE, str(script)], input=json.dumps(cases), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    got = json.loads(result.stdout)
    assert got == expected
    assert any(e["episodes"] > 0 for e in expected) and any(e["episodes"] == 0 for e in expected)
