from __future__ import annotations

from pathlib import Path

from jetson_app.calibration import CalibrationSample
from jetson_app.tag_stats import TagType
import pytest

from jetson_app.training import (
    _floor_std,
    load_artifact,
    make_train_fn,
    save_artifact,
    train_model,
)


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
    with pytest.raises(ValueError) as excinfo:
        train_model(samples, tags=("a", "b"), window_size=5, epochs=1)
    # 모든 태그가 최소 한 번은 관측된 경우에만 '샘플 부족' 메시지가 나와야 한다
    assert "not enough calibration samples" in str(excinfo.value)


def test_train_model_names_never_observed_tags():
    """DX1이 한 번도 발행하지 않는 태그(config 오탈자)가 원인일 때
    '샘플이 모자란다'가 아니라 문제 태그 이름을 알려줘야 한다."""
    samples = [
        CalibrationSample(
            timestamp=f"t{i}", values={"a": float(i % 2), "b": float(i), "ghost": None}
        )
        for i in range(50)
    ]
    with pytest.raises(ValueError) as excinfo:
        train_model(samples, tags=("a", "b", "ghost"), window_size=3, epochs=1)
    message = str(excinfo.value)
    assert "ghost" in message
    assert "never observed" in message
    assert "not enough calibration samples" not in message


def test_train_model_truncates_to_max_training_samples(tmp_path: Path):
    samples = _make_samples(50)
    artifact = train_model(
        samples,
        tags=("a", "b"),
        window_size=3,
        epochs=1,
        hidden_size=4,
        num_layers=1,
        max_training_samples=20,
    )
    assert artifact.tags == ("a", "b")
    assert artifact.window_size == 3
    assert set(artifact.error_stats.keys()) == {"a", "b"}
    assert "gru.weight_ih_l0" in artifact.state_dict


def test_floor_std_leaves_large_std_untouched():
    assert _floor_std(mean=1.0, std=0.5) == 0.5


def test_floor_std_replaces_zero_std():
    assert _floor_std(mean=0.0, std=0.0) == 1e-3


def test_floor_std_applies_relative_floor_to_tiny_std():
    # 0.05 * 2.0 = 0.1 > 1e-3 이므로 상대 하한이 적용된다
    assert _floor_std(mean=2.0, std=0.0193) == pytest.approx(0.1)
    # 음수 평균에도 절댓값 기준으로 동작해야 한다
    assert _floor_std(mean=-2.0, std=0.0193) == pytest.approx(0.1)


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


import torch

from jetson_app.model import AnomalyGRU
from jetson_app.training import (
    compute_raw_errors,
    model_artifact_path,
    normalize_continuous_columns,
    state_marker_path,
)


def test_normalize_continuous_columns_only_touches_continuous_indices():
    # tags=(cont, binary): index 0만 정규화 대상
    X = torch.tensor([[[10.0, 0.0], [20.0, 1.0]]])  # shape (1, 2, 2)
    y = torch.tensor([[30.0, 1.0]])
    norm_stats = {"cont": (10.0, 5.0)}
    normalize_continuous_columns(X, y, ("cont", "binary"), [0], norm_stats)
    assert torch.allclose(X[:, :, 0], torch.tensor([[0.0, 2.0]]))
    assert torch.allclose(X[:, :, 1], torch.tensor([[0.0, 1.0]]))  # binary 열은 그대로
    assert torch.allclose(y[:, 0], torch.tensor([4.0]))
    assert torch.allclose(y[:, 1], torch.tensor([1.0]))  # binary 열은 그대로


def test_compute_raw_errors_shapes_and_non_negative():
    model = AnomalyGRU(
        num_tags=2, continuous_indices=[0], binary_indices=[1], hidden_size=4, num_layers=1
    )
    X = torch.randn(3, 2, 2)
    y = torch.randn(3, 2)
    errors = compute_raw_errors(model, X, y, ("cont", "binary"), [0], [1], batch_size=2)
    assert set(errors.keys()) == {"cont", "binary"}
    assert errors["cont"].shape == (3,)
    assert errors["binary"].shape == (3,)
    assert torch.all(errors["cont"] >= 0)
    assert torch.all(errors["binary"] >= 0)


def test_model_artifact_path_builds_expected_path():
    path = model_artifact_path("model_data", "line_A")
    assert path == Path("model_data") / "line_A.pt"


def test_state_marker_path_builds_expected_path():
    path = state_marker_path("model_data", "line_A")
    assert path == Path("model_data") / "line_A.state"
