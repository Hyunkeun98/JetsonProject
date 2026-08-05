from __future__ import annotations

import math

from jetson_app.buffer import Snapshot
from jetson_app.calibration import CalibrationSample
from jetson_app.inference import ActiveModelHolder, AnomalyResult, InferenceEngine
from jetson_app.training import train_model


def _make_samples(n: int) -> list[CalibrationSample]:
    return [
        CalibrationSample(timestamp=f"t{i}", values={"a": float(i % 2), "b": float(i)})
        for i in range(n)
    ]


def _train_tiny_artifact():
    samples = _make_samples(30)
    return train_model(
        samples, tags=("a", "b"), window_size=3, epochs=2, hidden_size=4, num_layers=1
    )


def test_score_returns_result_for_full_window_and_present_values():
    artifact = _train_tiny_artifact()
    engine = InferenceEngine(artifact)
    window = [Snapshot(values={"a": 1.0, "b": float(i)}) for i in range(3)]
    actual = Snapshot(values={"a": 0.0, "b": 3.0})
    result = engine.score(window, actual)
    assert isinstance(result, AnomalyResult)
    assert result.top_deviant_tag in ("a", "b")
    assert not math.isnan(result.anomaly_score)


def test_score_returns_none_for_wrong_window_length():
    artifact = _train_tiny_artifact()
    engine = InferenceEngine(artifact)
    window = [Snapshot(values={"a": 1.0, "b": 1.0})]  # window_size는 3인데 1개뿐
    actual = Snapshot(values={"a": 0.0, "b": 3.0})
    assert engine.score(window, actual) is None


def test_score_returns_none_when_window_has_none_value():
    artifact = _train_tiny_artifact()
    engine = InferenceEngine(artifact)
    window = [Snapshot(values={"a": 1.0, "b": float(i)}) for i in range(2)] + [
        Snapshot(values={"a": None, "b": 2.0})
    ]
    actual = Snapshot(values={"a": 0.0, "b": 3.0})
    assert engine.score(window, actual) is None


def test_score_returns_none_when_actual_has_none_value():
    artifact = _train_tiny_artifact()
    engine = InferenceEngine(artifact)
    window = [Snapshot(values={"a": 1.0, "b": float(i)}) for i in range(3)]
    actual = Snapshot(values={"a": None, "b": 3.0})
    assert engine.score(window, actual) is None


def test_active_model_holder_starts_empty_and_can_be_set():
    holder = ActiveModelHolder()
    assert holder.get() is None
    artifact = _train_tiny_artifact()
    engine = InferenceEngine(artifact)
    holder.set(engine)
    assert holder.get() is engine
