from __future__ import annotations

from pathlib import Path

from jetson_app.calibration import CalibrationSample
from jetson_app.tag_stats import TagType
from jetson_app.training import load_artifact, make_train_fn, save_artifact, train_model


def _make_samples(n: int) -> list[CalibrationSample]:
    return [
        CalibrationSample(timestamp=f"t{i}", values={"a": float(i % 2), "b": float(i)})
        for i in range(n)
    ]


def test_train_model_produces_consistent_artifact():
    samples = _make_samples(30)
    artifact = train_model(
        samples,
        tags=("a", "b"),
        window_size=3,
        epochs=2,
        hidden_size=4,
        num_layers=1,
    )
    assert artifact.tags == ("a", "b")
    assert artifact.tag_types["a"] == TagType.BINARY.value
    assert artifact.tag_types["b"] == TagType.CONTINUOUS.value
    assert "b" in artifact.norm_stats
    assert "a" not in artifact.norm_stats
    assert set(artifact.error_stats.keys()) == {"a", "b"}
    assert artifact.window_size == 3
    assert artifact.hidden_size == 4
    assert artifact.num_layers == 1
    assert "gru.weight_ih_l0" in artifact.state_dict


def test_train_model_raises_when_not_enough_windows():
    samples = _make_samples(3)
    try:
        train_model(samples, tags=("a", "b"), window_size=5, epochs=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_save_and_load_artifact_round_trip(tmp_path: Path):
    samples = _make_samples(30)
    artifact = train_model(
        samples,
        tags=("a", "b"),
        window_size=3,
        epochs=2,
        hidden_size=4,
        num_layers=1,
    )
    path = tmp_path / "model.pt"
    save_artifact(path, artifact)
    assert path.exists()

    loaded = load_artifact(path)
    assert loaded.tags == artifact.tags
    assert loaded.tag_types == artifact.tag_types
    assert loaded.norm_stats == artifact.norm_stats
    assert loaded.error_stats == artifact.error_stats
    assert loaded.window_size == artifact.window_size
    assert loaded.hidden_size == artifact.hidden_size
    assert loaded.num_layers == artifact.num_layers
    assert loaded.state_dict.keys() == artifact.state_dict.keys()


def test_make_train_fn_trains_and_saves(tmp_path: Path):
    samples = _make_samples(30)
    model_path = tmp_path / "line_A.pt"
    train_fn = make_train_fn(
        tags=("a", "b"),
        window_size=3,
        model_path=model_path,
        epochs=2,
        hidden_size=4,
        num_layers=1,
    )
    train_fn(samples)
    assert model_path.exists()
    loaded = load_artifact(model_path)
    assert loaded.tags == ("a", "b")
