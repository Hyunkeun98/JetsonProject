from jetson_app.calibration import CalibrationSample
from jetson_app.tag_stats import (
    NormalizationStats,
    TagType,
    compute_normalization_stats,
    detect_tag_types,
)


def test_detect_tag_types_binary_tag():
    samples = [
        CalibrationSample(timestamp="t1", values={"a": 0, "b": 5.0}),
        CalibrationSample(timestamp="t2", values={"a": 1, "b": 5.5}),
        CalibrationSample(timestamp="t3", values={"a": 0, "b": 6.0}),
    ]
    result = detect_tag_types(samples, ("a", "b"))
    assert result["a"] == TagType.BINARY
    assert result["b"] == TagType.CONTINUOUS


def test_detect_tag_types_treats_float_zero_one_as_binary():
    samples = [
        CalibrationSample(timestamp="t1", values={"a": 0.0}),
        CalibrationSample(timestamp="t2", values={"a": 1.0}),
    ]
    result = detect_tag_types(samples, ("a",))
    assert result["a"] == TagType.BINARY


def test_detect_tag_types_never_observed_defaults_continuous():
    samples = [CalibrationSample(timestamp="t1", values={"a": None})]
    result = detect_tag_types(samples, ("a",))
    assert result["a"] == TagType.CONTINUOUS


def test_compute_normalization_stats_mean_and_std():
    samples = [
        CalibrationSample(timestamp="t1", values={"b": 2.0}),
        CalibrationSample(timestamp="t2", values={"b": 4.0}),
        CalibrationSample(timestamp="t3", values={"b": 6.0}),
    ]
    tag_types = {"b": TagType.CONTINUOUS}
    result = compute_normalization_stats(samples, ("b",), tag_types)
    assert result["b"].mean == 4.0
    assert abs(result["b"].std - 1.632993) < 1e-4


def test_compute_normalization_stats_constant_value_std_fallback():
    samples = [
        CalibrationSample(timestamp="t1", values={"b": 5.0}),
        CalibrationSample(timestamp="t2", values={"b": 5.0}),
    ]
    tag_types = {"b": TagType.CONTINUOUS}
    result = compute_normalization_stats(samples, ("b",), tag_types)
    assert result["b"].mean == 5.0
    assert result["b"].std == 1.0


def test_compute_normalization_stats_unobserved_tag_fallback():
    samples = [CalibrationSample(timestamp="t1", values={"b": None})]
    tag_types = {"b": TagType.CONTINUOUS}
    result = compute_normalization_stats(samples, ("b",), tag_types)
    assert result["b"].mean == 0.0
    assert result["b"].std == 1.0


def test_compute_normalization_stats_excludes_binary_tags():
    samples = [
        CalibrationSample(timestamp="t1", values={"a": 0, "b": 2.0}),
        CalibrationSample(timestamp="t2", values={"a": 1, "b": 4.0}),
    ]
    tag_types = {"a": TagType.BINARY, "b": TagType.CONTINUOUS}
    result = compute_normalization_stats(samples, ("a", "b"), tag_types)
    assert "a" not in result
    assert "b" in result
