from jetson_app.debounce import DEFAULT_CONFIRM_TICKS, DEFAULT_THRESHOLD, Debouncer


def test_debouncer_confirms_after_consecutive_ticks_over_threshold():
    d = Debouncer(threshold=3.0, confirm_ticks=3)
    assert d.update(5.0) is False
    assert d.update(5.0) is False
    assert d.update(5.0) is True


def test_debouncer_resets_on_dip_below_threshold():
    d = Debouncer(threshold=3.0, confirm_ticks=3)
    assert d.update(5.0) is False
    assert d.update(5.0) is False
    assert d.update(1.0) is False  # 밑으로 내려가면 카운터 리셋
    assert d.update(5.0) is False
    assert d.update(5.0) is False
    assert d.update(5.0) is True


def test_debouncer_score_exactly_at_threshold_counts_as_over():
    d = Debouncer(threshold=3.0, confirm_ticks=1)
    assert d.update(3.0) is True


def test_debouncer_stays_confirmed_while_scores_remain_over():
    d = Debouncer(threshold=3.0, confirm_ticks=2)
    assert d.update(5.0) is False
    assert d.update(5.0) is True
    assert d.update(5.0) is True  # 계속 알람 유지


def test_debouncer_default_constructor_uses_module_defaults():
    d = Debouncer()
    for _ in range(DEFAULT_CONFIRM_TICKS - 1):
        assert d.update(DEFAULT_THRESHOLD) is False
    assert d.update(DEFAULT_THRESHOLD) is True


def test_debouncer_rejects_non_positive_confirm_ticks():
    try:
        Debouncer(threshold=3.0, confirm_ticks=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
