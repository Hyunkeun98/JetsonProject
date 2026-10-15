import json
import random

from jetson_app.studio.cycles import CYCLE_TARGET, CycleCounter, cycle_warning, scan_directory
from studio_fakes import payload


def _feed(counter, steps, **series):
    for i in range(steps):
        counter.feed_payload(payload(i, **{name: fn(i) for name, fn in series.items()}))
    return counter


def _ramp(i):
    k = i % 300                       # 30초 주기: 10초 올라가고 10초 내려가고 10초 쉰다
    return float(k if k < 100 else 200 - k if k < 200 else 0)


def test_rising_edges_of_a_binary_signal_are_counted_and_the_smallest_is_the_default():
    counter = _feed(
        CycleCounter(), 1200,
        U0=lambda i: 1 if i % 100 >= 90 else 0,
        slow=lambda i: 1 if i % 200 >= 190 else 0,
    )

    report = counter.report()

    assert report.method == "edge" and report.exact is True
    assert report.tag == "slow" and report.cycles == 6
    assert {(c.tag, c.cycles) for c in report.candidates} == {("U0", 12), ("slow", 6)}
    assert counter.report("U0").cycles == 12 and counter.report("U0").tag == "U0"
    json.dumps(report.to_dict())                          # 그대로 JSON 응답으로 보낼 수 있어야 한다


def test_a_boolean_signal_counts_like_0_and_1():
    counter = _feed(CycleCounter(), 500, run=lambda i: (i % 100) >= 50)

    assert counter.report().cycles == 5


def test_an_increasing_integer_counter_counts_its_increments():
    counter = _feed(CycleCounter(), 1200, parts=lambda i: i // 100)

    report = counter.report()

    assert report.method == "counter" and report.cycles == 11 and report.exact is True


def test_a_timer_that_changes_every_sample_is_not_a_cycle_signal():
    counter = _feed(CycleCounter(), 600, timer=lambda i: i)

    assert counter.report().method == "none"


def test_a_single_edge_is_not_a_cycle():
    counter = _feed(CycleCounter(), 300, started=lambda i: 1 if i > 100 else 0)

    assert counter.report().method == "none"


def test_period_is_estimated_from_repeating_motion_when_there_is_no_cycle_signal():
    rng = random.Random(7)
    counter = _feed(CycleCounter(), 3000, pos=_ramp, vel=lambda i: rng.uniform(-1, 1))

    report = counter.report()

    assert report.method == "period" and report.exact is False and report.tag is None
    assert 28 <= report.period_seconds <= 32
    assert 9 <= report.cycles <= 11


def test_noise_without_any_repetition_gives_no_cycle_count():
    rng = random.Random(3)
    counter = _feed(CycleCounter(), 3000, vel=lambda i: rng.uniform(-1, 1))

    report = counter.report()

    assert report.method == "none" and report.cycles is None
    assert report.duration_seconds > 290 and "수집 시간" in report.message


def test_a_recording_shorter_than_three_periods_is_not_estimated():
    counter = _feed(CycleCounter(), 400, pos=_ramp)          # 40초: 주기 30초의 3배가 안 된다

    assert counter.report().method == "none"


def test_warning_only_below_ten_cycles_and_it_says_when_the_number_is_an_estimate():
    enough = _feed(CycleCounter(), 1200, U0=lambda i: 1 if i % 100 >= 90 else 0).report()
    few = _feed(CycleCounter(), 600, U0=lambda i: 1 if i % 100 >= 90 else 0).report()
    rng = random.Random(5)
    unknown = _feed(CycleCounter(), 100, vel=lambda i: rng.uniform(-1, 1)).report()
    estimated = _feed(CycleCounter(), 1800, pos=_ramp).report()

    assert CYCLE_TARGET == 10
    assert cycle_warning(enough) is None
    assert "6" in cycle_warning(few) and "추정" not in cycle_warning(few)
    assert "알 수 없습니다" in cycle_warning(unknown)
    assert estimated.method == "period" and "추정" in cycle_warning(estimated)


def test_scan_directory_reads_data_files_only(tmp_path):
    lines = [payload(i, U0=1 if i % 100 >= 90 else 0) for i in range(1200)]
    (tmp_path / "a.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    (tmp_path / "capture.meta").write_text('{"x": 1}', encoding="utf-8")

    assert scan_directory(tmp_path).report().cycles == 12
