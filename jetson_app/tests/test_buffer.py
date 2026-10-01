from jetson_app.buffer import Snapshot, SlidingWindow


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


def test_sliding_window_clear_empties_the_window():
    window = SlidingWindow(window_size=2)
    window.push(Snapshot(values={"a": 1}))
    window.push(Snapshot(values={"a": 2}))

    window.clear()

    assert window.to_list() == []
    assert window.is_full() is False
