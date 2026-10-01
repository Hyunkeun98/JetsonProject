import pytest

from jetson_app.timeparse import format_epoch_ns, parse_dx1_timestamp


def test_parses_dx1_nanosecond_timestamp_with_compact_offset():
    # 2026-10-01T00:47:01Z = 1790815621 초
    assert parse_dx1_timestamp("2026-10-01T00:47:01.718651520+0000") == 1790815621_718651520


def test_parses_z_suffix_and_colon_offset():
    base = parse_dx1_timestamp("2026-10-01T00:00:00Z")
    assert parse_dx1_timestamp("2026-10-01T00:00:00+00:00") == base
    # +09:00 이면 같은 순간이 9시간 이른 UTC 시각이다
    assert parse_dx1_timestamp("2026-10-01T09:00:00+09:00") == base
    assert parse_dx1_timestamp("2026-09-30T19:00:00-0500") == base


@pytest.mark.parametrize("fraction,expected_ns", [("", 0), (".5", 500_000_000), (".000001", 1000), (".123456789", 123456789)])
def test_fraction_digits_zero_to_nine(fraction, expected_ns):
    base = parse_dx1_timestamp("2026-10-01T00:00:00Z")
    assert parse_dx1_timestamp("2026-10-01T00:00:00%sZ" % fraction) == base + expected_ns


@pytest.mark.parametrize(
    "bad",
    ["t1", "", "2026-10-01", "2026-10-01T00:00:00", "2026-13-01T00:00:00Z", "2026-10-01T00:00:60Z", None, 123, "2026-10-01T00:00:00.1234567891Z"],
)
def test_invalid_timestamps_return_none(bad):
    assert parse_dx1_timestamp(bad) is None


def test_format_epoch_ns_round_trips_through_fromisoformat():
    from datetime import datetime

    epoch_ns = parse_dx1_timestamp("2026-10-01T00:47:01.718651520+0000")
    text = format_epoch_ns(epoch_ns)
    assert text == "2026-10-01T00:47:01.718651+00:00"
    assert datetime.fromisoformat(text).tzinfo is not None
    assert format_epoch_ns(parse_dx1_timestamp("2026-10-01T00:00:00Z")) == "2026-10-01T00:00:00+00:00"
