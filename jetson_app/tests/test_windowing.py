import torch

from jetson_app.calibration import CalibrationSample
from jetson_app.windowing import build_windows


def test_build_windows_basic_shapes_and_values():
    samples = [
        CalibrationSample(timestamp=f"t{i}", values={"a": float(i), "b": float(i * 10)})
        for i in range(5)
    ]
    X, y = build_windows(samples, ("a", "b"), window_size=2)
    assert X.shape == (3, 2, 2)
    assert y.shape == (3, 2)
    assert torch.equal(X[0], torch.tensor([[0.0, 0.0], [1.0, 10.0]]))
    assert torch.equal(y[0], torch.tensor([2.0, 20.0]))
    assert torch.equal(X[2], torch.tensor([[2.0, 20.0], [3.0, 30.0]]))
    assert torch.equal(y[2], torch.tensor([4.0, 40.0]))


def test_build_windows_skips_windows_with_none_in_window():
    samples = [
        CalibrationSample(timestamp="t0", values={"a": 1.0}),
        CalibrationSample(timestamp="t1", values={"a": None}),
        CalibrationSample(timestamp="t2", values={"a": 3.0}),
        CalibrationSample(timestamp="t3", values={"a": 4.0}),
    ]
    X, y = build_windows(samples, ("a",), window_size=2)
    assert X.shape == (0, 2, 1)
    assert y.shape == (0, 1)


def test_build_windows_skips_when_target_is_none():
    samples = [
        CalibrationSample(timestamp="t0", values={"a": 1.0}),
        CalibrationSample(timestamp="t1", values={"a": 2.0}),
        CalibrationSample(timestamp="t2", values={"a": None}),
    ]
    X, y = build_windows(samples, ("a",), window_size=2)
    assert X.shape == (0, 2, 1)
    assert y.shape == (0, 1)


def test_build_windows_insufficient_samples_returns_empty():
    samples = [CalibrationSample(timestamp="t0", values={"a": 1.0})]
    X, y = build_windows(samples, ("a",), window_size=5)
    assert X.shape == (0, 5, 1)
    assert y.shape == (0, 1)
