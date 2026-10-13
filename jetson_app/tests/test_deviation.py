import random

import pytest

from jetson_app.calibration import CalibrationSample
from jetson_app.config import DeviationConfig, DeviationSignalConfig
from jetson_app.deviation import (
    DeviationArtifact,
    DeviationEngine,
    DeviationSignal,
    DeviationTracker,
    derive_steps,
    fit_deviation,
    load_deviation_artifact,
    save_deviation_artifact,
)


def _tracker(baseline=5, settle=2, window=10, hold=5, move_range=100.0):
    return DeviationTracker(
        baseline_steps=baseline,
        settle_steps=settle,
        motion_window_steps=window,
        hold_steps=hold,
        move_range=move_range,
    )


def _run(tracker, values):
    return [tracker.push(v) for v in values]


def test_derive_steps_converts_seconds_to_grid_steps():
    steps = derive_steps(interval_ms=100, baseline_seconds=2.0, settle_seconds=1.0)
    assert steps == {
        "baseline_steps": 20,
        "settle_steps": 10,
        "motion_window_steps": 30,
        "hold_steps": 20,
    }


def test_tracker_reports_deviation_from_recent_median_after_settle_delay():
    results = _run(_tracker(), [0.0] * 20 + [1.0] + [0.0] * 10)
    # 스파이크는 20번째 값이고, settle_steps(2)만큼 늦은 22번째 push에서 판정이 나온다.
    assert results[22] == (1.0, False)
    assert all(r is None or r == (0.0, False) for i, r in enumerate(results) if i != 22)


def test_tracker_has_no_result_until_baseline_is_filled():
    results = _run(_tracker(baseline=5, settle=2), [0.0] * 6)
    assert all(r is None for r in results)


def test_tracker_marks_samples_as_moving_when_motion_follows_within_settle_time():
    # 20번째 값부터 빠르게 움직이기 시작한다. 시작 직후 샘플은 충격이 아니라 움직임의 시작이다.
    values = [0.0] * 20 + [60.0 * k for k in range(1, 15)]
    results = _run(_tracker(), values)
    first_ramp_result = results[22]  # 20번째 값(60.0)의 판정
    assert first_ramp_result is not None and first_ramp_result[1] is True


def test_tracker_excludes_baseline_recovery_after_motion():
    values = [0.0] * 20 + [60.0 * k for k in range(1, 5)] + [240.0] * 40
    results = _run(_tracker(), values)
    # 움직임이 끝난 뒤에도 hold_steps 동안은 기준값이 오염되어 있으므로 제외한다.
    after = [r for r in results[24:30] if r is not None]
    assert after and all(moving for _, moving in after)
    assert results[-1] == (0.0, False)


def test_tracker_reset_forgets_history():
    tracker = _tracker()
    _run(tracker, [0.0] * 20)
    tracker.reset()
    assert _run(tracker, [5.0] * 6) == [None] * 6


def _noise(n, sigma=0.01, seed=0, offset=0.0):
    rng = random.Random(seed)
    return [offset + rng.gauss(0.0, sigma) for _ in range(n)]


def _samples(values, tag="pos"):
    return [CalibrationSample(timestamp=f"2026-10-02T00:00:{i:02d}+00:00", values={tag: v}) for i, v in enumerate(values)]


def _config(**overrides):
    kwargs = dict(
        baseline_seconds=1.0,
        settle_seconds=0.5,
        margin=1.2,
        confirm_steps=1,
        signals=(DeviationSignalConfig(name="axis_x", tag="pos", move_range=None),),
    )
    kwargs.update(overrides)
    return DeviationConfig(**kwargs)


def test_fit_estimates_noise_scale_and_limit_from_normal_data():
    artifact = fit_deviation(_samples(_noise(3000)), _config(), interval_ms=100)
    signal = artifact.signals["axis_x"]
    # 이탈량 = 값 - 직전 중앙값이므로 잡음 sigma의 약 1.1배(중앙값 추정 오차 포함)다.
    assert 0.008 < signal.sigma < 0.020
    assert signal.limit > 3.0
    assert artifact.baseline_steps == 10 and artifact.settle_steps == 5


def test_fit_limit_is_margin_times_largest_normal_z():
    config = _config(margin=1.5)
    artifact = fit_deviation(_samples(_noise(3000, seed=1)), config, interval_ms=100)
    low = fit_deviation(_samples(_noise(3000, seed=1)), _config(margin=1.0), interval_ms=100)
    assert artifact.signals["axis_x"].limit == pytest.approx(1.5 * low.signals["axis_x"].limit)


def test_fit_auto_move_range_scales_with_signal_range_and_noise():
    # 신호가 0~50을 오가는 움직임이 있으면 범위의 1e-4(0.005)보다 잡음 100배(1.0)가 더 크다.
    values = _noise(1500) + [i * 1.0 for i in range(50)] + [50.0 + n for n in _noise(1500, seed=2)]
    signal = fit_deviation(_samples(values), _config(), interval_ms=100).signals["axis_x"]
    assert signal.move_range == pytest.approx(100 * 0.01, rel=0.4)


def test_fit_uses_configured_move_range_when_given():
    config = _config(signals=(DeviationSignalConfig(name="axis_x", tag="pos", move_range=0.5),))
    signal = fit_deviation(_samples(_noise(2000)), config, interval_ms=100).signals["axis_x"]
    assert signal.move_range == 0.5


def test_fit_ignores_motion_when_estimating_noise_and_limit():
    quiet = fit_deviation(_samples(_noise(3000)), _config(), interval_ms=100).signals["axis_x"]
    moving = _noise(1500) + [200.0 * k for k in range(1, 6)] + [1000.0 + n for n in _noise(1500, seed=3)]
    signal = fit_deviation(_samples(moving), _config(), interval_ms=100).signals["axis_x"]
    assert signal.limit == pytest.approx(quiet.limit, rel=0.5)


def test_fit_rejects_too_few_samples():
    with pytest.raises(ValueError, match="샘플"):
        fit_deviation(_samples(_noise(10)), _config(), interval_ms=100)


def test_fit_rejects_tag_without_observed_values():
    samples = [CalibrationSample(timestamp="2026-10-02T00:00:00+00:00", values={"other": 1.0}) for _ in range(500)]
    with pytest.raises(ValueError, match="pos"):
        fit_deviation(samples, _config(), interval_ms=100)


def _artifact(limit=5.0, sigma=0.01, move_range=100.0):
    return DeviationArtifact(
        resample_interval_ms=100,
        baseline_steps=5,
        settle_steps=2,
        motion_window_steps=10,
        hold_steps=5,
        signals={"axis_x": DeviationSignal(tag="pos", sigma=sigma, limit=limit, move_range=move_range)},
    )


def test_artifact_roundtrip(tmp_path):
    path = tmp_path / "dev.json"
    save_deviation_artifact(path, _artifact())
    assert load_deviation_artifact(path) == _artifact()


def test_engine_score_equals_alarm_threshold_when_deviation_reaches_limit():
    engine = DeviationEngine(_artifact(limit=5.0, sigma=0.01), alarm_threshold=3.0)
    outputs = []
    for v in [0.0] * 20 + [0.05] + [0.0] * 10:
        outputs.append(engine.update({"pos": v}))
    hit = [o for o in outputs if o and o["axis_x"].score > 0.0]
    assert len(hit) == 1
    result = hit[0]["axis_x"]
    assert result.score == pytest.approx(3.0)   # z=5(=limit) -> 임계값
    assert result.top_tag == "pos"


def test_engine_reports_zero_while_moving():
    engine = DeviationEngine(_artifact(), alarm_threshold=3.0)
    outputs = [engine.update({"pos": v}) for v in [0.0] * 20 + [60.0 * k for k in range(1, 15)]]
    scores = [o["axis_x"].score for o in outputs if o]
    assert scores and max(scores) == 0.0


def test_engine_skips_steps_where_tag_is_missing_and_resets():
    engine = DeviationEngine(_artifact(), alarm_threshold=3.0)
    assert engine.update({"pos": None}) is None
    for _ in range(20):
        engine.update({"pos": 0.0})
    engine.reset()
    assert all(engine.update({"pos": 9.0}) is None for _ in range(6))


def test_fit_falls_back_to_std_when_most_deviations_are_exactly_zero():
    # 거의 계단처럼 바뀌는 신호: 중앙값 절대편차가 0이어도 잡음 크기는 0이 되면 안 된다.
    values = [float(n // 100) + (0.001 if n % 50 == 0 else 0.0) for n in range(3000)]
    config = _config(signals=(DeviationSignalConfig(name="axis_x", tag="pos", move_range=10.0),))

    signal = fit_deviation(_samples(values), config, interval_ms=100).signals["axis_x"]

    assert signal.sigma > 1e-6 and signal.limit > 0


def test_fit_rejects_signal_that_never_deviates_while_idle():
    config = _config(signals=(DeviationSignalConfig(name="axis_x", tag="pos", move_range=10.0),))

    with pytest.raises(ValueError, match="전혀 변하지"):
        fit_deviation(_samples([5.0] * 1000), config, interval_ms=100)
