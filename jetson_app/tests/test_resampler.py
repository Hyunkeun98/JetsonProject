from jetson_app.mqtt_subscriber import Record
from jetson_app.resampler import EventTimeResampler


def rec(ts_ms, **values):
    return Record(timestamp="", values=values, epoch_ns=ts_ms * 1_000_000)


def feed(resampler, records):
    steps = []
    for r in records:
        steps.extend(resampler.add(r))
    return steps


def make(interval_ms=100, lateness_ms=0, window_size=5, tags=("a",)):
    return EventTimeResampler(tags=tags, interval_ms=interval_ms, max_lateness_ms=lateness_ms, window_size=window_size)


def values_of(steps, tag="a"):
    return [s.snapshot.values[tag] for s in steps]


def test_batch_of_ten_records_yields_ten_distinct_steps():
    r = make()
    batch = [rec(i * 100, a=i) for i in range(10)] + [rec(1000, a=10)]
    steps = feed(r, batch)
    assert values_of(steps) == list(range(10))
    assert [s.epoch_ns for s in steps] == [i * 100_000_000 for i in range(10)]


def test_faster_than_grid_keeps_last_value_in_bucket():
    r = make()
    steps = feed(r, [rec(0, a=1), rec(20, a=2), rec(40, a=3), rec(60, a=4), rec(80, a=5), rec(100, a=6)])
    assert values_of(steps) == [5]


def test_slower_than_grid_forward_fills():
    r = make()
    steps = feed(r, [rec(0, a=1), rec(450, a=2)])
    assert values_of(steps) == [1, 1, 1, 1]
    assert [s.epoch_ns for s in steps] == [0, 100_000_000, 200_000_000, 300_000_000]


def test_fine_grid_20ms_with_100ms_source():
    r = make(interval_ms=20)
    steps = feed(r, [rec(0, a=1), rec(100, a=2), rec(200, a=3)])
    assert len(steps) == 10
    assert values_of(steps) == [1] * 5 + [2] * 5


def test_coarse_grid_1000ms_with_100ms_source():
    r = make(interval_ms=1000)
    steps = feed(r, [rec(i * 100, a=i) for i in range(21)])
    assert values_of(steps) == [9, 19]
    assert [s.epoch_ns for s in steps] == [0, 1_000_000_000]


def test_max_lateness_delays_emission_and_late_record_is_dropped():
    r = make(lateness_ms=300)
    assert feed(r, [rec(0, a=1), rec(200, a=2)]) == []
    steps = feed(r, [rec(500, a=3)])
    assert values_of(steps) == [1, 1]  # 칸 0, 1 확정 (칸 1은 ffill)
    assert r.add(rec(150, a=99)) == []  # 이미 확정된 칸 1에 속하는 늦은 record
    assert r.late_dropped == 1


def test_multi_topic_values_align_within_lateness():
    r = make(lateness_ms=200, tags=("a", "b"))
    steps = feed(
        r,
        [rec(0, a=1), rec(100, a=2), rec(200, a=3), rec(0, b=10), rec(100, b=20), rec(300, a=4)],
    )
    assert [s.snapshot.values for s in steps] == [{"a": 1, "b": 10}]
    steps = feed(r, [rec(400, a=5)])
    assert [s.snapshot.values for s in steps] == [{"a": 2, "b": 20}]


def test_partial_snapshot_keeps_none_for_unseen_tag_and_all_none_is_skipped():
    r = make(tags=("a", "b"))
    assert r.add(rec(0, zzz=1)) == []  # 추적하지 않는 태그뿐인 record
    steps = feed(r, [rec(0, a=1), rec(100, a=2)])
    assert [s.snapshot.values for s in steps] == [{"a": 1, "b": None}]


def test_records_inside_one_bucket_use_latest_event_time_regardless_of_arrival_order():
    r = make()
    steps = feed(r, [rec(80, a=2), rec(20, a=1), rec(100, a=3)])
    assert values_of(steps) == [2]


def test_duplicate_delivery_is_idempotent():
    r = make()
    steps = feed(r, [rec(0, a=1), rec(0, a=1), rec(100, a=2)])
    assert values_of(steps) == [1]
    assert r.add(rec(0, a=1)) == []  # 확정 뒤 재전송은 늦은 record로 폐기
    assert r.late_dropped == 1


def test_gap_longer_than_window_resets_window_without_flooding_steps():
    r = make(window_size=3)
    steps = feed(r, [rec(0, a=1), rec(100, a=2), rec(200, a=3), rec(10_000, a=4)])
    assert values_of(steps) == [1, 2, 3]
    assert [s.reset_window for s in steps] == [False, False, False]
    steps = feed(r, [rec(10_100, a=5)])
    assert values_of(steps) == [4]
    assert steps[0].reset_window is True
    assert steps[0].epoch_ns == 10_000 * 1_000_000


def test_one_day_forward_jump_stays_bounded():
    r = make(window_size=3)
    feed(r, [rec(0, a=1), rec(100, a=2)])
    steps = feed(r, [rec(86_400_000, a=3), rec(86_400_100, a=4)])
    assert len(steps) <= 3
    assert steps[-1].reset_window is True


def test_gap_equal_to_window_size_is_forward_filled_not_reset():
    r = make(window_size=3)
    steps = feed(r, [rec(0, a=1), rec(400, a=2), rec(500, a=3)])
    assert values_of(steps) == [1, 1, 1, 1, 2]
    assert all(s.reset_window is False for s in steps)


def test_record_far_in_the_past_after_emission_is_dropped():
    r = make()
    feed(r, [rec(0, a=1), rec(100, a=2), rec(200, a=3)])
    assert r.add(rec(-86_400_000, a=9)) == []
    assert r.late_dropped == 1
