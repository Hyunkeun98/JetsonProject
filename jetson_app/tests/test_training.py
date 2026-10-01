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


def test_train_model_records_resample_interval_in_artifact():
    artifact = train_model(
        _make_samples(30),
        tags=("a", "b"),
        window_size=3,
        resample_interval_ms=100,
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )
    assert artifact.resample_interval_ms == 100


def test_save_and_load_artifact_preserves_resample_interval(tmp_path: Path):
    artifact = train_model(
        _make_samples(30),
        tags=("a", "b"),
        window_size=3,
        resample_interval_ms=100,
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )
    path = tmp_path / "model.pt"
    save_artifact(path, artifact)

    assert load_artifact(path).resample_interval_ms == 100


def test_load_artifact_without_grid_info_defaults_to_zero(tmp_path: Path):
    # 격자 정보가 생기기 전에 저장된 기존 artifact: pipeline이 항상 불일치로 처리하도록 0
    artifact = train_model(
        _make_samples(30),
        tags=("a", "b"),
        window_size=3,
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )
    path = tmp_path / "legacy.pt"
    save_artifact(path, artifact)
    import torch as _torch

    data = _torch.load(path, weights_only=False)
    del data["resample_interval_ms"]
    _torch.save(data, path)

    assert load_artifact(path).resample_interval_ms == 0


def test_make_train_fn_trains_and_saves(tmp_path: Path):
    samples = _make_samples(30)
    model_path = tmp_path / "line_A.pt"
    train_fn = make_train_fn(
        tags=("a", "b"),
        window_size=3,
        model_path=model_path,
        resample_interval_ms=50,
        epochs=2,
        hidden_size=4,
        num_layers=1,
    )
    train_fn(samples)
    assert model_path.exists()
    loaded = load_artifact(model_path)
    assert loaded.tags == ("a", "b")
    assert loaded.resample_interval_ms == 50


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


# ---- 그룹/상태별 정상 오차 기준 ----

from jetson_app.training import MIN_REGIME_SAMPLES, compute_regime_error_stats


def _state_samples(n, state_for, a_for=lambda i: float(i % 7)):
    return [
        CalibrationSample(
            timestamp=f"t{i}",
            values={"s": state_for(i), "a": a_for(i), "b": float(i)},
        )
        for i in range(n)
    ]


def test_regime_stats_split_errors_by_state_value():
    errors = {"a": torch.tensor([0.1] * 150 + [1.0] * 150)}
    state_values = {"s": torch.tensor([0.0] * 150 + [1.0] * 150)}
    groups = {"g": ("s", ("s", "a"))}

    stats, fallbacks = compute_regime_error_stats(errors, state_values, groups)

    assert stats["a"]["off"][0] == pytest.approx(0.1)
    assert stats["a"]["on"][0] == pytest.approx(1.0)
    assert stats["a"]["off"][2] == 150 and stats["a"]["on"][2] == 150
    # 표준편차 하한(_floor_std)은 상태별 평균 기준으로 적용된다
    assert stats["a"]["off"][1] == pytest.approx(0.005)
    assert stats["a"]["on"][1] == pytest.approx(0.05)
    assert "s" not in stats  # 상태 태그 자신은 전체 통계를 쓴다
    assert fallbacks == []


def test_regime_stats_treat_half_as_on():
    errors = {"a": torch.tensor([0.2] * MIN_REGIME_SAMPLES + [0.9] * MIN_REGIME_SAMPLES)}
    state_values = {"s": torch.tensor([0.49] * MIN_REGIME_SAMPLES + [0.5] * MIN_REGIME_SAMPLES)}

    stats, _ = compute_regime_error_stats(errors, state_values, {"g": ("s", ("s", "a"))})

    assert stats["a"]["off"][0] == pytest.approx(0.2)
    assert stats["a"]["on"][0] == pytest.approx(0.9)


def test_regime_stats_drop_states_with_too_few_samples_and_report_them():
    errors = {"a": torch.tensor([0.1] * 200 + [1.0] * 5)}
    state_values = {"s": torch.tensor([0.0] * 200 + [1.0] * 5)}

    stats, fallbacks = compute_regime_error_stats(errors, state_values, {"g": ("s", ("s", "a"))})

    assert "on" not in stats["a"]
    assert "off" in stats["a"]
    assert fallbacks == ["a(on:5)"]


def test_regime_stats_ignore_groups_without_a_state_tag():
    errors = {"a": torch.tensor([0.1] * 300)}

    stats, fallbacks = compute_regime_error_stats(errors, {}, {"general": (None, ("a",))})

    assert stats == {} and fallbacks == []


def test_train_model_fills_groups_and_regime_stats():
    groups = {"proc": ("s", ("s", "a")), "general": (None, ("b",))}
    samples = _state_samples(400, state_for=lambda i: float((i // 10) % 2))

    artifact = train_model(
        samples,
        tags=("s", "a", "b"),
        window_size=3,
        epochs=1,
        hidden_size=4,
        num_layers=1,
        groups=groups,
    )

    assert artifact.groups == groups
    assert set(artifact.regime_error_stats) == {"a"}  # s는 상태 태그, b는 상태 없는 그룹
    on_n = artifact.regime_error_stats["a"]["on"][2]
    off_n = artifact.regime_error_stats["a"]["off"][2]
    assert on_n + off_n == 400 - 3  # 윈도우 수
    assert set(artifact.error_stats) == {"s", "a", "b"}  # 전체 통계는 그대로


def test_train_model_judges_state_on_raw_values_not_normalized_ones():
    # 상태 태그가 0.0/0.2 사이를 오간다. 원본 기준(>= 0.5)이면 모두 OFF이고,
    # 정규화된 값(-1/+1)으로 판정하면 절반이 ON이 되어 버린다.
    groups = {"proc": ("s", ("s", "a"))}
    samples = _state_samples(400, state_for=lambda i: 0.2 * ((i // 10) % 2))

    artifact = train_model(
        samples,
        tags=("s", "a", "b"),
        window_size=3,
        epochs=1,
        hidden_size=4,
        num_layers=1,
        groups=groups,
    )

    assert set(artifact.regime_error_stats["a"]) == {"off"}
    assert artifact.regime_error_stats["a"]["off"][2] == 400 - 3


def test_train_model_logs_states_that_fall_back_to_pooled_stats(capsys):
    groups = {"proc": ("s", ("s", "a"))}
    samples = _state_samples(300, state_for=lambda i: 0.0)  # 한 번도 켜지지 않음

    artifact = train_model(
        samples,
        tags=("s", "a", "b"),
        window_size=3,
        epochs=1,
        hidden_size=4,
        num_layers=1,
        groups=groups,
    )

    assert "on" not in artifact.regime_error_stats["a"]
    assert "a(on:0)" in capsys.readouterr().out


def test_save_and_load_artifact_preserve_groups_and_regime_stats(tmp_path: Path):
    groups = {"proc": ("s", ("s", "a")), "general": (None, ("b",))}
    artifact = train_model(
        _state_samples(400, state_for=lambda i: float((i // 10) % 2)),
        tags=("s", "a", "b"),
        window_size=3,
        epochs=1,
        hidden_size=4,
        num_layers=1,
        groups=groups,
    )
    path = tmp_path / "model.pt"
    save_artifact(path, artifact)

    loaded = load_artifact(path)

    assert loaded.groups == artifact.groups
    assert loaded.regime_error_stats == artifact.regime_error_stats


def test_load_artifact_without_group_fields_defaults_to_empty(tmp_path: Path):
    artifact = train_model(
        _make_samples(30), tags=("a", "b"), window_size=3, epochs=1, hidden_size=4, num_layers=1
    )
    path = tmp_path / "legacy.pt"
    save_artifact(path, artifact)
    data = torch.load(path, weights_only=False)
    del data["groups"]
    del data["regime_error_stats"]
    torch.save(data, path)

    loaded = load_artifact(path)

    assert loaded.groups == {} and loaded.regime_error_stats == {}


def test_make_train_fn_passes_groups_to_the_artifact(tmp_path: Path):
    groups = {"proc": ("s", ("s", "a")), "general": (None, ("b",))}
    model_path = tmp_path / "m.pt"
    train_fn = make_train_fn(
        tags=("s", "a", "b"),
        window_size=3,
        model_path=model_path,
        resample_interval_ms=100,
        groups=groups,
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )

    train_fn(_state_samples(400, state_for=lambda i: float((i // 10) % 2)))

    assert load_artifact(model_path).groups == groups
