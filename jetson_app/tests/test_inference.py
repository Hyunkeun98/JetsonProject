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


# ---- 그룹 점수와 상태별 기준 ----

import pytest
import torch

import jetson_app.inference as inference_module
from jetson_app.inference import GroupResult
from jetson_app.model import AnomalyGRU
from jetson_app.training import ModelArtifact


def _grouped_artifact(groups, regime_error_stats, error_stats=None):
    tags = ("s", "a", "b")
    model = AnomalyGRU(
        num_tags=3, continuous_indices=[0, 1, 2], binary_indices=[], hidden_size=2, num_layers=1
    )
    return ModelArtifact(
        tags=tags,
        tag_types={"s": "continuous", "a": "continuous", "b": "continuous"},
        norm_stats={"s": (0.0, 1.0), "a": (0.0, 1.0), "b": (0.0, 1.0)},
        # 상태 태그 s의 기준을 높게 잡아(z가 음수) 그룹 점수가 a의 점수로 정해지게 한다.
        error_stats=error_stats
        or {"s": (1.0, 1.0), "a": (0.5, 0.5), "b": (0.0, 1.0)},
        window_size=2,
        hidden_size=2,
        num_layers=1,
        state_dict=model.state_dict(),
        resample_interval_ms=100,
        groups=groups,
        regime_error_stats=regime_error_stats,
    )


_GROUPS = {"proc": ("s", ("s", "a")), "general": (None, ("b",))}
_REGIME = {"a": {"on": (1.0, 1.0, 200), "off": (0.1, 0.05, 200)}}


def _score_with_injected_errors(monkeypatch, artifact, state_value, errors):
    # 모델 출력과 무관하게 원본 오차를 직접 주입해 기준 선택 규칙만 검증한다.
    injected = {tag: torch.tensor([value]) for tag, value in errors.items()}
    monkeypatch.setattr(inference_module, "compute_raw_errors", lambda *args, **kwargs: injected)
    engine = InferenceEngine(artifact)
    window = [Snapshot(values={"s": 0.0, "a": 0.0, "b": 0.0}) for _ in range(2)]
    actual = Snapshot(values={"s": state_value, "a": 0.0, "b": 0.0})
    return engine.score(window, actual)


def test_same_raw_error_scores_higher_when_group_is_off_than_on(monkeypatch):
    artifact = _grouped_artifact(_GROUPS, _REGIME)
    errors = {"s": 0.0, "a": 0.6, "b": 0.0}

    off = _score_with_injected_errors(monkeypatch, artifact, state_value=0.0, errors=errors)
    on = _score_with_injected_errors(monkeypatch, artifact, state_value=1.0, errors=errors)

    assert off.group_results["proc"].score == pytest.approx((0.6 - 0.1) / 0.05)  # 대기 기준: 좁다
    assert on.group_results["proc"].score == pytest.approx((0.6 - 1.0) / 1.0)  # 동작 기준: 넓다
    assert off.group_results["proc"].score > on.group_results["proc"].score
    assert off.group_results["proc"].top_tag == "a"


def test_overall_score_is_the_maximum_group_score(monkeypatch):
    artifact = _grouped_artifact(_GROUPS, _REGIME)
    errors = {"s": 0.0, "a": 0.6, "b": 3.0}  # 동작 중이면 a는 낮고 general 그룹의 b가 가장 크다

    result = _score_with_injected_errors(monkeypatch, artifact, state_value=1.0, errors=errors)

    assert set(result.group_results) == {"proc", "general"}
    assert result.group_results["general"] == GroupResult(score=pytest.approx(3.0), top_tag="b")
    assert result.anomaly_score == pytest.approx(
        max(g.score for g in result.group_results.values())
    )
    assert result.top_deviant_tag == "b"


def test_state_without_regime_stats_falls_back_to_pooled_stats(monkeypatch):
    regime_only_on = {"a": {"on": (1.0, 1.0, 200)}}  # off 표본 부족으로 빠진 경우
    artifact = _grouped_artifact(_GROUPS, regime_only_on)
    errors = {"s": 0.0, "a": 0.6, "b": 0.0}

    result = _score_with_injected_errors(monkeypatch, artifact, state_value=0.0, errors=errors)

    assert result.group_results["proc"].score == pytest.approx((0.6 - 0.5) / 0.5)  # error_stats


def test_artifact_without_groups_scores_as_a_single_all_group(monkeypatch):
    artifact = _grouped_artifact({}, {})
    errors = {"s": 0.0, "a": 0.6, "b": 2.0}

    result = _score_with_injected_errors(monkeypatch, artifact, state_value=0.0, errors=errors)

    assert list(result.group_results) == ["all"]
    assert result.group_results["all"].score == pytest.approx(result.anomaly_score)
    assert result.anomaly_score == pytest.approx(2.0)
    assert result.top_deviant_tag == "b"


def test_stats_for_uses_pooled_stats_when_state_value_is_unknown():
    engine = InferenceEngine(_grouped_artifact(_GROUPS, _REGIME))

    assert engine._stats_for("a", None) == (0.5, 0.5)
    assert engine._stats_for("a", 1.0) == (1.0, 1.0)
    assert engine._stats_for("a", 0.0) == (0.1, 0.05)
    assert engine._stats_for("s", 1.0) == (1.0, 1.0)  # 상태 태그 자신은 항상 전체 통계


def test_real_trained_grouped_artifact_scores_without_error():
    groups = {"proc": ("s", ("s", "a")), "general": (None, ("b",))}
    samples = [
        CalibrationSample(
            timestamp=f"t{i}",
            values={"s": float((i // 10) % 2), "a": float(i % 7), "b": float(i)},
        )
        for i in range(400)
    ]
    artifact = train_model(
        samples, tags=("s", "a", "b"), window_size=3, epochs=1, hidden_size=4, num_layers=1,
        groups=groups,
    )
    engine = InferenceEngine(artifact)
    window = [Snapshot(values={"s": 1.0, "a": 1.0, "b": float(i)}) for i in range(3)]

    result = engine.score(window, Snapshot(values={"s": 1.0, "a": 2.0, "b": 3.0}))

    assert set(result.group_results) == {"proc", "general"}
    assert not math.isnan(result.anomaly_score)


# ---- 점수 대상과 입력 전용 태그 ----


def _context_artifact():
    # a: 점수 대상, s: 점수 밖 상태 태그, c: 어느 그룹에도 없는 입력 전용 태그
    tags = ("a", "s", "c")
    model = AnomalyGRU(
        num_tags=3, continuous_indices=[0, 1, 2], binary_indices=[], hidden_size=2, num_layers=1
    )
    return ModelArtifact(
        tags=tags,
        tag_types={t: "continuous" for t in tags},
        norm_stats={t: (0.0, 1.0) for t in tags},
        # s와 c는 표준편차가 아주 작아 오차가 조금만 커도 z가 폭발한다.
        error_stats={"a": (0.5, 0.5), "s": (0.0, 0.01), "c": (0.0, 0.01)},
        window_size=2,
        hidden_size=2,
        num_layers=1,
        state_dict=model.state_dict(),
        resample_interval_ms=100,
        groups={"proc": ("s", ("a",))},
        regime_error_stats={"a": {"on": (1.0, 1.0, 200), "off": (0.1, 0.05, 200)}},
    )


def _score_context(monkeypatch, state_value, errors, window_c=0.0):
    injected = {tag: torch.tensor([value]) for tag, value in errors.items()}
    monkeypatch.setattr(inference_module, "compute_raw_errors", lambda *args, **kwargs: injected)
    engine = InferenceEngine(_context_artifact())
    window = [Snapshot(values={"a": 0.0, "s": 0.0, "c": window_c}) for _ in range(2)]
    actual = Snapshot(values={"a": 0.0, "s": state_value, "c": 0.0})
    return engine.score(window, actual)


def test_unscored_state_and_context_tags_never_affect_the_score(monkeypatch):
    errors = {"a": 0.6, "s": 9.0, "c": 9.0}  # s, c의 오차는 매우 크지만 점수 대상이 아니다

    result = _score_context(monkeypatch, state_value=1.0, errors=errors)

    assert set(result.group_results) == {"proc"}
    assert result.group_results["proc"] == GroupResult(score=pytest.approx(-0.4), top_tag="a")
    assert result.anomaly_score == pytest.approx(-0.4)  # 동작 기준 (0.6-1.0)/1.0, s/c의 z(900)는 무시
    assert result.top_deviant_tag == "a"


def test_state_tag_outside_scored_tags_still_selects_the_regime(monkeypatch):
    errors = {"a": 0.6, "s": 9.0, "c": 9.0}

    off = _score_context(monkeypatch, state_value=0.0, errors=errors)

    assert off.anomaly_score == pytest.approx((0.6 - 0.1) / 0.05)  # 대기 기준


def test_score_is_none_when_no_scored_tag_has_an_error(monkeypatch):
    result = _score_context(monkeypatch, state_value=1.0, errors={"s": 1.0, "c": 1.0})

    assert result is None


def test_score_is_none_when_a_context_tag_is_unobserved_in_the_window():
    engine = InferenceEngine(_context_artifact())
    window = [Snapshot(values={"a": 0.0, "s": 0.0, "c": None}) for _ in range(2)]

    result = engine.score(window, Snapshot(values={"a": 0.0, "s": 1.0, "c": 0.0}))

    assert result is None  # 입력 전용 태그 토픽이 오지 않으면 채점이 멈춘다(README에 안내)


def test_real_trained_artifact_with_context_tag_scores_only_the_scored_group():
    groups = {"proc": ("s", ("a",))}
    samples = [
        CalibrationSample(
            timestamp=f"t{i}",
            values={"s": float((i // 10) % 2), "a": float(i % 7), "c": float(i % 5)},
        )
        for i in range(400)
    ]
    artifact = train_model(
        samples, tags=("a", "s", "c"), window_size=3, epochs=1, hidden_size=4, num_layers=1,
        groups=groups,
    )
    engine = InferenceEngine(artifact)
    window = [Snapshot(values={"a": 1.0, "s": 1.0, "c": float(i)}) for i in range(3)]

    result = engine.score(window, Snapshot(values={"a": 2.0, "s": 1.0, "c": 3.0}))

    assert set(artifact.regime_error_stats) == {"a"}  # 점수 대상 태그만 상태별 기준이 있다
    assert set(result.group_results) == {"proc"}
    assert result.top_deviant_tag == "a"
    assert not math.isnan(result.anomaly_score)
