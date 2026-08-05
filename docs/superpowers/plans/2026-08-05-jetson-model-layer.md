# Jetson 모델 레이어 (PyTorch GRU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 캘리브레이션 버퍼(`list[CalibrationSample]`)를 입력받아 설비 통합 PyTorch GRU 모델을 학습하고, 정규화/오차 통계와 함께 디스크에 저장/로드하는 기능을 만들어, 지금까지 자리표시자였던 `train_fn`을 실제 학습 로직으로 교체한다.

**Architecture:** 태그 타입(이진/연속) 자동 판별 → 정규화 통계 계산 → 슬라이딩 윈도우 시퀀스 생성 → 태그 통합 1개 GRU 모델(타입별 손실: 연속=MSE, 이진=BCE)을 학습 → 학습 후 캘리브레이션 데이터로 태그별 "정상 오차" 평균/표준편차 계산 → 가중치+통계를 단일 파일로 저장. `jetson_app.calibration.CalibrationManager`가 이미 구현해 둔 `train`/`recalibrate` 상태머신에 `TrainFn` 콜백으로 꽂아 넣는다.

**Tech Stack:** PyTorch (CPU, PyPI), 기존 `jetson_app` 패키지(Python 3.8 타깃, uv 관리).

**범위:** 이 계획은 [`2026-08-04-jetson-anomaly-inference-pipeline-design.md`](../specs/2026-08-04-jetson-anomaly-inference-pipeline-design.md)의 5절(모델 구조)을 구현한다. 6절(실시간 이상 점수 계산/디바운스)과 7절의 추론 관련 에러 처리는 다음 계획(실시간 추론 통합, sub-project 3)에서 다룬다 — 이 계획은 "학습해서 파일로 저장"까지만 하고, 저장된 모델을 불러와 실시간 채점하는 로직은 만들지 않는다.

## Global Constraints

- **Python 3.8 호환**: 모든 신규 파일 최상단에 `from __future__ import annotations`를 두고, 애노테이션 위치(함수 인자/반환값, 데이터클래스 필드)에서는 소문자 제네릭(`dict[...]`, `list[...]`, `tuple[...]`, `X | None`)을 자유롭게 써도 된다(지연 평가). 단, 모듈 최상위에서 **실행 시점에 평가되는** 제네릭 별칭(예: `Foo = Callable[[...], ...]` 형태의 대입문)은 `typing.Callable`/`typing.List` 등 `typing` 모듈을 써야 한다 — 이전 계획에서 `TrainFn = Callable[[list[...]], None]`이 3.8 임포트를 깨뜨린 실제 버그가 있었다. 이 계획에서는 새 모듈 최상위 제네릭 별칭을 만들지 않고 `jetson_app.calibration.TrainFn`을 그대로 재사용한다.
- **uv로만 의존성 관리**: `uv add <package>`로 추가하고 `uv run pytest`로 테스트 실행. `pip install`을 직접 쓰지 않는다.
- **torch는 개발/테스트용 PyPI(CPU) 버전**: `uv add torch`로 추가하는 표준 PyPI wheel은 개발 PC와 실제 Jetson(JetPack 5.x, aarch64) 양쪽에서 CPU로는 동작하지만, Jetson의 GPU를 쓰려면 NVIDIA가 JetPack 버전별로 배포하는 전용 wheel로 나중에 교체해야 한다(Task 5에서 README에 수동 가이드만 남긴다 — 실기가 없어 여기서 검증하지 않는다).
- **수동 트리거 원칙 유지**: 모델 학습은 오직 `CalibrationManager.handle_train_command()` → `TrainFn` 콜백을 통해서만 일어난다. 이 계획에서 새로 자동/주기적 학습 트리거를 추가하지 않는다.
- **태그 타입은 캘리브레이션 데이터로 자동 판별**: config에 타입을 적지 않는다(스펙 1절 원칙 유지) — 태그의 관측값이 전부 `{0, 1}`이면 이진, 아니면 연속.
- **하이퍼파라미터는 코드에 합리적 기본값으로 하드코딩**: `hidden_size`, `num_layers`, `learning_rate`, `epochs`, `batch_size`는 함수 인자로 오버라이드 가능하게 만들되 (테스트에서 빠른 학습을 위해 필요), `config.yaml`이나 CLI 플래그로는 노출하지 않는다.
- **모델 아티팩트는 단일 파일**: `torch.save`로 가중치(`state_dict`) + 정규화 통계 + 태그 타입 + 오차 통계를 하나의 dict로 묶어 저장. `torch.load`는 항상 `weights_only=False`를 명시한다(순수 텐서가 아닌 일반 파이썬 객체를 함께 저장하므로, torch 기본값이 버전에 따라 바뀌어도 깨지지 않도록).

---

### Task 1: 태그 타입 자동 판별 + 정규화 통계

**Files:**
- Create: `jetson_app/src/jetson_app/tag_stats.py`
- Test: `jetson_app/tests/test_tag_stats.py`

**Interfaces:**
- Consumes: `jetson_app.calibration.CalibrationSample(timestamp: str, values: dict[str, float | int | None])` (기존)
- Produces: `TagType`(Enum: `BINARY`, `CONTINUOUS`), `NormalizationStats(mean: float, std: float)` frozen dataclass, `detect_tag_types(samples, tags) -> dict[str, TagType]`, `compute_normalization_stats(samples, tags, tag_types) -> dict[str, NormalizationStats]` — Task 4(`training.py`)가 그대로 가져다 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_tag_stats.py`:

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd jetson_app && uv run pytest tests/test_tag_stats.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'jetson_app.tag_stats'`)

- [ ] **Step 3: 구현**

`jetson_app/src/jetson_app/tag_stats.py`:

```python
from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum

from .calibration import CalibrationSample

_BINARY_VALUES = {0, 1, 0.0, 1.0}


class TagType(Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float


def detect_tag_types(
    samples: list[CalibrationSample], tags: tuple[str, ...]
) -> dict[str, TagType]:
    result: dict[str, TagType] = {}
    for tag in tags:
        observed = [
            s.values[tag] for s in samples if tag in s.values and s.values[tag] is not None
        ]
        if observed and all(v in _BINARY_VALUES for v in observed):
            result[tag] = TagType.BINARY
        else:
            result[tag] = TagType.CONTINUOUS
    return result


def compute_normalization_stats(
    samples: list[CalibrationSample],
    tags: tuple[str, ...],
    tag_types: dict[str, TagType],
) -> dict[str, NormalizationStats]:
    result: dict[str, NormalizationStats] = {}
    for tag in tags:
        if tag_types.get(tag) != TagType.CONTINUOUS:
            continue
        observed = [
            float(s.values[tag])
            for s in samples
            if tag in s.values and s.values[tag] is not None
        ]
        if not observed:
            result[tag] = NormalizationStats(mean=0.0, std=1.0)
            continue
        mean = statistics.fmean(observed)
        std = statistics.pstdev(observed) if len(observed) > 1 else 0.0
        result[tag] = NormalizationStats(mean=mean, std=std if std > 0 else 1.0)
    return result
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd jetson_app && uv run pytest tests/test_tag_stats.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: 커밋**

```bash
cd jetson_app
git add src/jetson_app/tag_stats.py tests/test_tag_stats.py
git commit -m "feat: add tag type auto-detection and normalization stats"
```

---

### Task 2: torch 의존성 추가 + 슬라이딩 윈도우 시퀀스 생성

**Files:**
- Modify: `jetson_app/pyproject.toml` (torch 의존성 추가, `uv add`가 자동 처리)
- Create: `jetson_app/src/jetson_app/windowing.py`
- Test: `jetson_app/tests/test_windowing.py`

**Interfaces:**
- Consumes: `jetson_app.calibration.CalibrationSample` (기존)
- Produces: `build_windows(samples, tags, window_size) -> tuple[torch.Tensor, torch.Tensor]` — Task 4(`training.py`)가 학습 입력(X)/타깃(y) 생성에 그대로 쓴다. `X.shape == (n_windows, window_size, len(tags))`, `y.shape == (n_windows, len(tags))`.

- [ ] **Step 1: torch 의존성 추가**

Run: `cd jetson_app && uv add torch`
Expected: `pyproject.toml`의 `dependencies`에 `torch`가 추가되고 `uv.lock`이 갱신됨, 명령이 exit 0로 종료.

- [ ] **Step 2: import 확인**

Run: `cd jetson_app && uv run python -c "import torch; print(torch.__version__)"`
Expected: 버전 문자열 출력, exit 0.

- [ ] **Step 3: 실패하는 테스트 작성**

`jetson_app/tests/test_windowing.py`:

```python
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
```

- [ ] **Step 4: 테스트 실패 확인**

Run: `cd jetson_app && uv run pytest tests/test_windowing.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'jetson_app.windowing'`)

- [ ] **Step 5: 구현**

`jetson_app/src/jetson_app/windowing.py`:

```python
from __future__ import annotations

import torch

from .calibration import CalibrationSample


def build_windows(
    samples: list[CalibrationSample],
    tags: tuple[str, ...],
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """samples를 (window_size, target) 슬라이딩 윈도우로 잘라 학습용 텐서로 만든다.
    윈도우 안이나 타깃에 None이 하나라도 있으면 그 윈도우는 통째로 건너뛴다."""
    n_tags = len(tags)
    window_rows: list[list[list[float]]] = []
    target_rows: list[list[float]] = []

    for start in range(len(samples) - window_size):
        window = samples[start : start + window_size]
        target = samples[start + window_size]

        window_values: list[list[float]] = []
        window_has_none = False
        for sample in window:
            row = [sample.values.get(tag) for tag in tags]
            if any(v is None for v in row):
                window_has_none = True
                break
            window_values.append([float(v) for v in row])
        if window_has_none:
            continue

        target_row = [target.values.get(tag) for tag in tags]
        if any(v is None for v in target_row):
            continue

        window_rows.append(window_values)
        target_rows.append([float(v) for v in target_row])

    if not window_rows:
        return (
            torch.empty((0, window_size, n_tags), dtype=torch.float32),
            torch.empty((0, n_tags), dtype=torch.float32),
        )

    X = torch.tensor(window_rows, dtype=torch.float32)
    y = torch.tensor(target_rows, dtype=torch.float32)
    return X, y
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `cd jetson_app && uv run pytest tests/test_windowing.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: 커밋**

```bash
cd jetson_app
git add pyproject.toml uv.lock src/jetson_app/windowing.py tests/test_windowing.py
git commit -m "feat: add torch dependency and sliding-window sequence builder"
```

---

### Task 3: GRU 모델 정의 (타입별 head)

**Files:**
- Create: `jetson_app/src/jetson_app/model.py`
- Test: `jetson_app/tests/test_model.py`

**Interfaces:**
- Consumes: `torch`(Task 2에서 추가됨)
- Produces: `AnomalyGRU(nn.Module)` — 생성자 `(num_tags, continuous_indices, binary_indices, hidden_size=64, num_layers=2)`, `forward(x) -> (continuous_out, binary_logits)` (둘 중 해당 태그가 없으면 그쪽은 `None`). Task 4(`training.py`)가 이 클래스로 모델을 만들고 학습한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_model.py`:

```python
import torch

from jetson_app.model import AnomalyGRU


def test_forward_shapes_with_mixed_tags():
    model = AnomalyGRU(
        num_tags=3,
        continuous_indices=[0, 2],
        binary_indices=[1],
        hidden_size=8,
        num_layers=1,
    )
    x = torch.randn(4, 5, 3)  # batch=4, window=5, tags=3
    continuous_out, binary_logits = model(x)
    assert continuous_out.shape == (4, 2)
    assert binary_logits.shape == (4, 1)


def test_forward_all_continuous_binary_logits_none():
    model = AnomalyGRU(
        num_tags=2,
        continuous_indices=[0, 1],
        binary_indices=[],
        hidden_size=4,
        num_layers=1,
    )
    x = torch.randn(2, 3, 2)
    continuous_out, binary_logits = model(x)
    assert continuous_out.shape == (2, 2)
    assert binary_logits is None


def test_forward_all_binary_continuous_out_none():
    model = AnomalyGRU(
        num_tags=2,
        continuous_indices=[],
        binary_indices=[0, 1],
        hidden_size=4,
        num_layers=1,
    )
    x = torch.randn(2, 3, 2)
    continuous_out, binary_logits = model(x)
    assert continuous_out is None
    assert binary_logits.shape == (2, 2)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd jetson_app && uv run pytest tests/test_model.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'jetson_app.model'`)

- [ ] **Step 3: 구현**

`jetson_app/src/jetson_app/model.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class AnomalyGRU(nn.Module):
    """설비 통합 1개 GRU. 연속 태그는 회귀 head(MSE용), 이진 태그는
    시그모이드 이전 logit head(BCEWithLogitsLoss용)로 다음 시점 값을 예측한다."""

    def __init__(
        self,
        num_tags: int,
        continuous_indices: list[int],
        binary_indices: list[int],
        hidden_size: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.continuous_indices = continuous_indices
        self.binary_indices = binary_indices
        self.gru = nn.GRU(
            input_size=num_tags,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.continuous_head = (
            nn.Linear(hidden_size, len(continuous_indices)) if continuous_indices else None
        )
        self.binary_head = (
            nn.Linear(hidden_size, len(binary_indices)) if binary_indices else None
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]  # 마지막 레이어의 마지막 시점 hidden state, shape (batch, hidden_size)
        continuous_out = (
            self.continuous_head(last_hidden) if self.continuous_head is not None else None
        )
        binary_logits = (
            self.binary_head(last_hidden) if self.binary_head is not None else None
        )
        return continuous_out, binary_logits
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd jetson_app && uv run pytest tests/test_model.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
cd jetson_app
git add src/jetson_app/model.py tests/test_model.py
git commit -m "feat: add AnomalyGRU model with type-aware output heads"
```

---

### Task 4: 학습 루프 + 모델 아티팩트 저장/로드

**Files:**
- Create: `jetson_app/src/jetson_app/training.py`
- Test: `jetson_app/tests/test_training.py`

**Interfaces:**
- Consumes: `CalibrationSample`, `TrainFn`(from `jetson_app.calibration`), `detect_tag_types`/`compute_normalization_stats`/`TagType`(from `jetson_app.tag_stats`, Task 1), `build_windows`(from `jetson_app.windowing`, Task 2), `AnomalyGRU`(from `jetson_app.model`, Task 3)
- Produces: `ModelArtifact` frozen dataclass, `train_model(samples, tags, window_size, epochs=20, hidden_size=64, num_layers=2, learning_rate=1e-3, batch_size=64) -> ModelArtifact`, `save_artifact(path, artifact) -> None`, `load_artifact(path) -> ModelArtifact`, `make_train_fn(tags, window_size, model_path, epochs=20, hidden_size=64, num_layers=2, learning_rate=1e-3, batch_size=64) -> TrainFn` — Task 5(`subscriber_cli.py`)가 `make_train_fn`을 호출해 실제 `train_fn`으로 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_training.py`:

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd jetson_app && uv run pytest tests/test_training.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'jetson_app.training'`)

- [ ] **Step 3: 구현**

`jetson_app/src/jetson_app/training.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .calibration import CalibrationSample, TrainFn
from .model import AnomalyGRU
from .tag_stats import TagType, compute_normalization_stats, detect_tag_types
from .windowing import build_windows

DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64


@dataclass(frozen=True)
class ModelArtifact:
    tags: tuple[str, ...]
    tag_types: dict[str, str]
    norm_stats: dict[str, tuple[float, float]]
    error_stats: dict[str, tuple[float, float]]
    window_size: int
    hidden_size: int
    num_layers: int
    state_dict: dict


def train_model(
    samples: list[CalibrationSample],
    tags: tuple[str, ...],
    window_size: int,
    epochs: int = DEFAULT_EPOCHS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ModelArtifact:
    tag_types = detect_tag_types(samples, tags)
    norm_stats = compute_normalization_stats(samples, tags, tag_types)

    continuous_indices = [i for i, t in enumerate(tags) if tag_types[t] == TagType.CONTINUOUS]
    binary_indices = [i for i, t in enumerate(tags) if tag_types[t] == TagType.BINARY]

    X, y = build_windows(samples, tags, window_size)
    if X.shape[0] == 0:
        raise ValueError(
            f"not enough calibration samples to build any training window "
            f"(have {len(samples)} samples, need > {window_size})"
        )

    for idx in continuous_indices:
        tag = tags[idx]
        stats = norm_stats[tag]
        X[:, :, idx] = (X[:, :, idx] - stats.mean) / stats.std
        y[:, idx] = (y[:, idx] - stats.mean) / stats.std

    model = AnomalyGRU(
        num_tags=len(tags),
        continuous_indices=continuous_indices,
        binary_indices=binary_indices,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    n_windows = X.shape[0]
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(n_windows)
        for start in range(0, n_windows, batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch_X = X[batch_idx]
            batch_y = y[batch_idx]

            optimizer.zero_grad()
            continuous_out, binary_logits = model(batch_X)

            loss = torch.tensor(0.0)
            if continuous_indices:
                loss = loss + mse_loss_fn(continuous_out, batch_y[:, continuous_indices])
            if binary_indices:
                loss = loss + bce_loss_fn(binary_logits, batch_y[:, binary_indices])

            loss.backward()
            optimizer.step()

    error_stats = _compute_error_stats(model, X, y, tags, continuous_indices, binary_indices)

    return ModelArtifact(
        tags=tags,
        tag_types={t: tag_types[t].value for t in tags},
        norm_stats={t: (s.mean, s.std) for t, s in norm_stats.items()},
        error_stats=error_stats,
        window_size=window_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        state_dict=model.state_dict(),
    )


def _compute_error_stats(
    model: AnomalyGRU,
    X: torch.Tensor,
    y: torch.Tensor,
    tags: tuple[str, ...],
    continuous_indices: list[int],
    binary_indices: list[int],
) -> dict[str, tuple[float, float]]:
    """학습 완료 후 캘리브레이션 데이터 전체에 대한 태그별 "정상 오차" 평균/표준편차.
    실시간 이상 점수 계산(다음 계획)에서 원본 오차를 재정규화하는 기준으로 쓰인다."""
    model.eval()
    with torch.no_grad():
        continuous_out, binary_logits = model(X)

    errors: dict[str, torch.Tensor] = {}
    for pos, idx in enumerate(continuous_indices):
        errors[tags[idx]] = torch.abs(y[:, idx] - continuous_out[:, pos])
    for pos, idx in enumerate(binary_indices):
        pred_prob = torch.sigmoid(binary_logits[:, pos])
        errors[tags[idx]] = torch.abs(y[:, idx] - pred_prob)

    result: dict[str, tuple[float, float]] = {}
    for tag, err in errors.items():
        mean = err.mean().item()
        if err.numel() > 1:
            variance = ((err - err.mean()) ** 2).mean().item()
            std = variance ** 0.5
        else:
            std = 0.0
        result[tag] = (mean, std if std > 0 else 1.0)
    return result


def save_artifact(path: str | Path, artifact: ModelArtifact) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tags": artifact.tags,
            "tag_types": artifact.tag_types,
            "norm_stats": artifact.norm_stats,
            "error_stats": artifact.error_stats,
            "window_size": artifact.window_size,
            "hidden_size": artifact.hidden_size,
            "num_layers": artifact.num_layers,
            "state_dict": artifact.state_dict,
        },
        path,
    )


def load_artifact(path: str | Path) -> ModelArtifact:
    data = torch.load(Path(path), weights_only=False)
    return ModelArtifact(
        tags=tuple(data["tags"]),
        tag_types=data["tag_types"],
        norm_stats=data["norm_stats"],
        error_stats=data["error_stats"],
        window_size=data["window_size"],
        hidden_size=data["hidden_size"],
        num_layers=data["num_layers"],
        state_dict=data["state_dict"],
    )


def make_train_fn(
    tags: tuple[str, ...],
    window_size: int,
    model_path: str | Path,
    epochs: int = DEFAULT_EPOCHS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> TrainFn:
    def _train_fn(samples: list[CalibrationSample]) -> None:
        artifact = train_model(
            samples,
            tags=tags,
            window_size=window_size,
            epochs=epochs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        save_artifact(model_path, artifact)

    return _train_fn
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd jetson_app && uv run pytest tests/test_training.py -v`
Expected: PASS (4 passed). 학습이 CPU에서 도는 만큼 몇 초 정도 걸릴 수 있다.

- [ ] **Step 5: 커밋**

```bash
cd jetson_app
git add src/jetson_app/training.py tests/test_training.py
git commit -m "feat: add GRU training loop and model artifact save/load"
```

---

### Task 5: CLI에 실제 학습 연결 + Jetson 배포 문서화

**Files:**
- Modify: `jetson_app/src/jetson_app/subscriber_cli.py`
- Modify: `jetson_app/README.md`

**Interfaces:**
- Consumes: `make_train_fn`(from `jetson_app.training`, Task 4), 기존 `build_pipeline`(from `jetson_app.pipeline`, 변경 없음 — 이미 `train_fn: TrainFn`을 인자로 받는다)

- [ ] **Step 1: `subscriber_cli.py`에서 자리표시자 제거하고 실제 학습 연결**

`jetson_app/src/jetson_app/subscriber_cli.py` 전체를 아래로 교체:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_equipment_config
from .pipeline import build_pipeline
from .training import make_train_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="DX1 -> Jetson MQTT 데이터 파이프라인")
    parser.add_argument("--config", required=True, help="설비 config YAML 경로")
    parser.add_argument("--host", required=True, help="MQTT 브로커 호스트 (Jetson 자신의 IP 또는 localhost)")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument(
        "--calibration-dir",
        default="calibration_data",
        help="캘리브레이션 버퍼 파일을 저장할 디렉터리 (기본: calibration_data)",
    )
    parser.add_argument(
        "--model-dir",
        default="model_data",
        help="학습된 모델 아티팩트를 저장할 디렉터리 (기본: model_data)",
    )
    args = parser.parse_args()

    try:
        config = load_equipment_config(args.config)
        model_path = Path(args.model_dir) / f"{config.equipment_id}.pt"
        train_fn = make_train_fn(
            tags=config.tags,
            window_size=config.window_size,
            model_path=model_path,
        )
        pipeline = build_pipeline(
            config=config,
            calibration_dir=args.calibration_dir,
            train_fn=train_fn,
        )
        pipeline.mqtt_subscriber.connect(args.host, args.port)
    except (ConfigError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)

    pipeline.snapshotter.start()
    print(
        f"[{config.equipment_id}] {len(config.subscribe_topics)}개 토픽 구독 시작 "
        f"({args.host}:{args.port}), 캘리브레이션 데이터: {args.calibration_dir}, "
        f"모델 저장 위치: {args.model_dir}"
    )
    try:
        pipeline.mqtt_subscriber.loop_forever()
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        pipeline.snapshotter.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 전체 테스트 스위트 + CLI 스모크 체크**

Run: `cd jetson_app && uv run pytest -v`
Expected: 모두 PASS (Task 1~4에서 추가된 테스트 포함, 기존 61개 + 신규 테스트).

Run: `cd jetson_app && uv run jetson-app --help`
Expected: `--model-dir` 옵션이 포함된 argparse 도움말이 출력되고 exit 0 (실제 연결 시도 없음).

- [ ] **Step 3: README 갱신**

`jetson_app/README.md`의 7번째 줄(현재 범위 설명)을 다음으로 교체:

```markdown
현재 범위: 설비 config 로더(다중 토픽) + MQTT 파싱/구독자 + Tag Buffer/슬라이딩 윈도우 + 주기 스냅샷 스케줄러 + 캘리브레이션 저장/상태머신 + MQTT train/recalibrate 명령 구독자 + 설비 통합 PyTorch GRU 모델 학습(태그 타입별 손실, 정규화/오차 통계 저장/로드) + CLI 진입점(코드, 유닛테스트 완료). 실시간 이상 점수 계산/디바운스/Result Publisher는 이후 별도 계획.
```

"### 2-4. 설비 config 준비" 절 바로 뒤에 아래 절을 새로 추가:

```markdown
### 2-5. Jetson 전용 PyTorch로 교체 (GPU 가속용, 선택)

`uv sync`는 PyPI의 일반 torch wheel(CPU 전용, aarch64도 지원)을 설치한다. 이것만으로도 학습/추론은 동작하지만 Jetson의 GPU를 쓰지 못한다. GPU 가속이 필요해지면 NVIDIA가 JetPack 버전별로 배포하는 전용 wheel로 교체한다:

1. 현재 JetPack 버전 확인: `sudo apt-cache show nvidia-jetpack | grep Version` (또는 `cat /etc/nv_tegra_release`)
2. https://developer.download.nvidia.com/compute/redist/jp/ 에서 해당 JetPack 버전 폴더의 torch wheel(`.whl`) URL을 확인한다.
3. 프로젝트 가상환경 안에서 PyPI 버전을 제거하고 해당 wheel을 직접 설치한다:
   ```bash
   uv pip uninstall torch
   uv pip install <위에서 확인한 .whl URL 또는 로컬 경로>
   ```
4. 확인: `uv run python -c "import torch; print(torch.cuda.is_available())"`가 `True`를 출력하면 GPU 인식 성공.

이후 `uv sync`를 다시 실행하면 `pyproject.toml`에 적힌 일반 torch로 되돌아간다 — JetPack wheel을 유지하려면 `uv sync --no-install-package torch`를 쓰거나, `uv sync` 이후 3번 단계를 다시 실행한다.
```

"### 캘리브레이션 데이터 위치와 train/recalibrate 명령" 절 마지막(현재 "주의: ..." 문단 앞)에 아래 두 문단을 추가:

```markdown
- `--model-dir` 플래그로 학습된 모델을 저장할 위치를 지정한다(기본: `model_data`). `train` 명령이 성공하면 `<model-dir>/<equipment_id>.pt`에 GRU 가중치 + 정규화 통계 + 태그 타입 + 오차 분포 통계가 함께 저장된다.
- `train` 명령은 이제 실제로 PyTorch GRU 모델을 학습한다(캘리브레이션 샘플 수·태그 수에 따라 CPU에서 수 초~수십 초 소요될 수 있다). 학습이 끝나기 전까지는 다른 MQTT 명령 처리가 지연될 수 있다.
```

- [ ] **Step 4: 커밋**

```bash
cd jetson_app
git add src/jetson_app/subscriber_cli.py README.md
git commit -m "feat: wire real GRU training into train command, document Jetson torch deployment"
```

---

## Self-Review 체크리스트 (참고용, 실행 시 삭제하지 않음)

- **스펙 커버리지**: 스펙 5절(모델 구조: 통합 GRU, 타입별 손실, 학습 시 저장 항목)은 Task 1~4가 구현한다. 스펙 6절(실시간 채점)·7절(추론 관련 에러 처리)은 의도적으로 이 계획 범위 밖 — "범위" 절에 명시.
- **자리표시자 없음**: 모든 스텝에 실제 코드가 포함되어 있다.
- **타입 일관성**: `TrainFn`은 `jetson_app.calibration`의 기존 정의를 재사용(재정의 없음). `ModelArtifact`의 필드명(`tags`, `tag_types`, `norm_stats`, `error_stats`, `window_size`, `hidden_size`, `num_layers`, `state_dict`)은 `save_artifact`/`load_artifact`/테스트에서 동일하게 사용된다.
