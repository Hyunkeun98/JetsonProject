import random

import pytest

from jetson_app.scores import ScoreRow, ScoresError, read_scores_csv, write_scores_csv
from jetson_app.threshold import (
    ThresholdError,
    _simulate_segment,
    analyze,
    critical_score,
    format_report,
)

STEP_NS = 100_000_000


def _rows(kind, name, scores, start_ns=0, resets=(), group="g"):
    return [
        ScoreRow(
            epoch_ns=start_ns + i * STEP_NS,
            kind=kind,
            segment=name,
            reset=(i == 0 or i in resets),
            groups={group: (score, "tag")},
        )
        for i, score in enumerate(scores)
    ]


def test_critical_score_is_max_of_window_minimums():
    rows = _rows("normal_eval", "n", [1, 5, 4, 6, 2, 9, 9, 9, 1])
    assert critical_score(rows, ("g",), 3) == 9
    assert critical_score(rows, ("g",), 2) == 9
    assert critical_score(rows, ("g",), 1) == 9
    assert critical_score(_rows("normal_eval", "n", [1, 5, 4, 6]), ("g",), 3) == 4


def test_critical_score_without_enough_rows_is_minus_infinity():
    rows = _rows("normal_eval", "n", [9, 9])
    assert critical_score(rows, ("g",), 3) == float("-inf")


def test_critical_score_does_not_bridge_a_reset():
    rows = _rows("normal_eval", "n", [9, 9, 9, 9], resets=(2,))
    assert critical_score(rows, ("g",), 3) == float("-inf")
    assert critical_score(rows, ("g",), 2) == 9


def test_critical_score_matches_simulation_for_random_scores():
    rng = random.Random(1)
    for _ in range(40):
        scores = [rng.uniform(0, 10) for _ in range(rng.randint(1, 60))]
        resets = tuple(i for i in range(1, len(scores)) if rng.random() < 0.08)
        rows = _rows("normal_eval", "n", scores, resets=resets)
        for confirm in (1, 2, 3, 5):
            c = critical_score(rows, ("g",), confirm)
            for threshold in (0.5, 2.0, 4.0, 6.0, 8.0, 9.5):
                alarms, _first = _simulate_segment(rows, ("g",), threshold, confirm)
                assert (alarms > 0) == (threshold <= c)


def test_critical_score_takes_the_max_over_groups():
    rows = [
        ScoreRow(i * STEP_NS, "normal_eval", "n", i == 0, {"a": (1.0, "x"), "b": (7.0, "y")})
        for i in range(5)
    ]
    assert critical_score(rows, ("a", "b"), 3) == 7.0
    assert critical_score(rows, ("a",), 3) == 1.0


def test_simulation_counts_a_continuous_alarm_once():
    rows = _rows("normal_eval", "n", [5, 5, 5, 5, 5, 0, 5, 5, 5])
    alarms, first = _simulate_segment(rows, ("g",), 3.0, 3)
    assert alarms == 2
    assert first == 2 * STEP_NS


def test_normal_only_recommends_t0_with_margin_and_flags_unverified_sensitivity():
    rows = _rows("normal_eval", "n", [1.0, 2.5, 2.5, 2.5, 1.0] * 20)
    result = analyze(rows, ("g",), confirm=3)
    assert result.t0 == 2.6  # 정상 임계 점수 2.5보다 큰 0.1 단위 값
    assert result.recommended == 3.2  # 2.6 * 1.2 = 3.12 -> 0.1 단위로 올림
    assert result.t1 is None
    assert any("민감도" in note for note in result.notes)


def test_t0_is_never_below_the_minimum_threshold():
    rows = _rows("normal_eval", "n", [0.1] * 50)
    assert analyze(rows, ("g",), confirm=3).t0 == 2.0


def test_anomaly_recommendation_is_the_middle_of_the_workable_range():
    normal = _rows("normal_eval", "n", [1.0] * 40 + [3.0] * 5 + [1.0] * 40)
    anomalies = []
    for number, level in enumerate((20.0, 16.0, 18.0), start=1):
        anomalies += _rows("anomaly", f"a{number}", [1.0] * 10 + [level] * 10, start_ns=number * 10**12)
    result = analyze(normal + anomalies, ("g",), confirm=3)
    assert result.t0 == 3.1
    assert result.t1 == 16.0
    assert result.recommended == 9.6  # (3.1 + 16.0) / 2 = 9.55 -> 9.6
    assert result.missed_at_t0 == ()
    row_at_10 = next(r for r in result.table if r.threshold == 10.0)
    assert (row_at_10.normal_alarms, row_at_10.anomalies_detected, row_at_10.anomalies_total) == (0, 3, 3)
    assert row_at_10.mean_latency_s == pytest.approx(1.2)  # 알람은 11번째 행(인덱스 12)에서 확정


def test_conflict_gives_no_recommendation_and_names_missed_segments():
    normal = _rows("normal_eval", "n", [1.0] * 20 + [8.0] * 5 + [1.0] * 20)
    weak = _rows("anomaly", "weak", [1.0] * 10 + [5.0] * 10, start_ns=10**12)
    strong = _rows("anomaly", "strong", [1.0] * 10 + [30.0] * 10, start_ns=2 * 10**12)
    result = analyze(normal + weak + strong, ("g",), confirm=3)
    assert result.t0 == 8.1
    assert result.recommended is None
    assert result.missed_at_t0 == ("weak",)
    assert any("동시에 만족" in note for note in result.notes)


def test_few_anomaly_segments_produce_an_overfitting_warning():
    normal = _rows("normal_eval", "n", [1.0] * 30)
    anomaly = _rows("anomaly", "a", [1.0] * 5 + [20.0] * 10, start_ns=10**12)
    result = analyze(normal + anomaly, ("g",), confirm=3)
    assert any("과적합" in note for note in result.notes)


def test_max_alarms_per_day_picks_the_lowest_grid_value_that_fits():
    # 정상 구간 약 100초 동안 점수 4.5짜리 알람이 2건 -> 임계값 5 이상이면 0건
    scores = ([1.0] * 100 + [4.5] * 5) * 2 + [1.0] * 100
    rows = _rows("normal_eval", "n", scores)
    result = analyze(rows, ("g",), confirm=3, max_alarms_per_day=1.0)
    assert result.recommended == 5.0


def test_analyze_requires_normal_rows():
    with pytest.raises(ThresholdError, match="정상 검증"):
        analyze(_rows("anomaly", "a", [1.0] * 10), ("g",), confirm=3)


def test_format_report_mentions_recommendation_and_table():
    rows = _rows("normal_eval", "n", [1.0] * 30)
    text = format_report(analyze(rows, ("g",), confirm=3))
    assert "추천 임계값" in text and "임계값" in text


def test_scores_csv_round_trip(tmp_path):
    rows = [
        ScoreRow(10**18, "normal_eval", "n1", True, {"a": (1.5, "t1"), "b": (-0.25, "t2")}),
        ScoreRow(10**18 + STEP_NS, "anomaly", "touch", False, {"a": (3.0, "t1")}),
    ]
    path = tmp_path / "scores.csv"
    write_scores_csv(path, rows, ("a", "b"))
    loaded, groups = read_scores_csv(path)
    assert groups == ("a", "b")
    assert loaded == rows


def test_scores_csv_rejects_foreign_files(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ScoresError, match="머리글"):
        read_scores_csv(path)
    with pytest.raises(ScoresError, match="읽을 수 없습니다"):
        read_scores_csv(tmp_path / "missing.csv")
