from jetson_app.buffer import Snapshot, SlidingWindow, TagBuffer


def test_tag_buffer_update_and_snapshot_returns_latest_values():
    buf = TagBuffer(tags=("a", "b"))

    buf.update({"a": 1, "b": 2})

    assert buf.snapshot() == Snapshot(values={"a": 1, "b": 2})


def test_tag_buffer_ignores_untracked_tags():
    buf = TagBuffer(tags=("a",))

    buf.update({"a": 1, "unrelated": 99})

    assert buf.snapshot() == Snapshot(values={"a": 1})


def test_tag_buffer_snapshot_returns_none_for_unseen_tags():
    buf = TagBuffer(tags=("a", "b"))

    buf.update({"a": 1})

    assert buf.snapshot() == Snapshot(values={"a": 1, "b": None})


def test_tag_buffer_snapshot_keeps_last_value_until_next_update():
    buf = TagBuffer(tags=("a",))

    buf.update({"a": 1})
    first = buf.snapshot()
    second = buf.snapshot()

    assert first == second == Snapshot(values={"a": 1})


def test_sliding_window_push_and_to_list_preserves_order():
    window = SlidingWindow(window_size=3)
    s1, s2 = Snapshot(values={"a": 1}), Snapshot(values={"a": 2})

    window.push(s1)
    window.push(s2)

    assert window.to_list() == [s1, s2]


def test_sliding_window_is_full_only_when_window_size_reached():
    window = SlidingWindow(window_size=2)

    assert window.is_full() is False
    window.push(Snapshot(values={"a": 1}))
    assert window.is_full() is False
    window.push(Snapshot(values={"a": 2}))
    assert window.is_full() is True


def test_sliding_window_drops_oldest_when_over_capacity():
    window = SlidingWindow(window_size=2)
    s1, s2, s3 = (
        Snapshot(values={"a": 1}),
        Snapshot(values={"a": 2}),
        Snapshot(values={"a": 3}),
    )

    window.push(s1)
    window.push(s2)
    window.push(s3)

    assert window.to_list() == [s2, s3]
