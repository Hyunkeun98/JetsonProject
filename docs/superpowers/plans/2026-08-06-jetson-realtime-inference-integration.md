# Jetson 실시간 추론 통합 (Inference Engine + Result Publisher) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 이미 학습·저장된 GRU 모델을 불러와 실시간 윈도우로 다음 시점을 예측하고, 캘리브레이션 구간 "정상 오차" 통계로 재정규화한 이상 점수를 계산해, 디바운스를 거쳐 MQTT로 발행한다. 또한 Jetson 재시작 시 저장된 모델이 있으면 자동으로 MONITORING을 재개하고, 모델 파일이 손상되었으면 CALIBRATING으로 안전하게 폴백한다.

**Architecture:** `PeriodicSnapshotter`의 매 틱에서, 새 스냅샷을 슬라이딩 윈도우에 넣기 *전에* 그 시점까지의 윈도우로 `InferenceEngine`이 다음 값을 예측 → 실제 도착한 값과 비교해 태그별 원본 오차 계산 → 학습 시 저장해둔 "정상 오차" 평균/표준편차로 재정규화 → 태그 중 최댓값 z-score를 이상 점수로 사용 → `Debouncer`가 연속 N틱 초과를 확인해야 알람 확정 → `ResultPublisher`가 MQTT로 발행. `CalibrationManager`는 상태(`CALIBRATING`/`MONITORING`)를 파일에 영속화해, 재시작 시 기존 모델을 불러와 이어서 재개하거나(성공 시) CALIBRATING으로 폴백한다(실패 시).

**Tech Stack:** 기존 `jetson_app` 패키지(Python 3.8, uv 관리, torch 이미 의존성에 있음 — 이번 계획에서 새 의존성 추가 없음).

**범위:** [`2026-08-04-jetson-anomaly-inference-pipeline-design.md`](../specs/2026-08-04-jetson-anomaly-inference-pipeline-design.md)의 6절(실시간 이상 점수 계산)을 구현하고, 7절의 "모델 파일 손상/로드 실패" 폴백을 완성한다. [`2026-07-31-jetson-dx1-anomaly-framework-design.md`](../specs/2026-07-31-jetson-dx1-anomaly-framework-design.md) 1절 아키텍처의 나머지 두 컴포넌트(Inference Engine, Result Publisher)를 채운다. Phase 2(기여도 분석) 이후는 범위 밖 — `top_deviant_tag` 하나만 발행하고, 태그별 상세 랭킹 API는 만들지 않는다.

## 설계 결정 (사용자 확인 완료)

- **재시작 시 기존 모델이 있으면 자동으로 MONITORING 재개.** 모델 로드 실패(손상/누락) 시에만 CALIBRATING 폴백.
- **`recalibrate` 명령은 모델 파일을 디스크에서 지우지 않는다** — 메모리상에서만 미사용 처리한다. 이 선택은 "재시작 시 모델 있으면 자동 재개"와 상충하므로(recalibrate 직후 재시작하면 방금 되돌린 CALIBRATING을 무시하고 옛 모델로 MONITORING이 되어버림), **모델 파일과 별도로 상태 마커 파일**(`<equipment_id>.state`, `CALIBRATING`/`MONITORING` 텍스트 한 줄)을 둔다. `train`/`recalibrate` 성공 시마다 이 마커를 갱신하고, 재시작 시 모델 파일이 아니라 **이 마커를 기준**으로 재개 여부를 판단한다. 모델 파일 존재 여부만으로 판단하지 않는다.

## Global Constraints

- **Python 3.8 호환**: 모든 신규/수정 파일 최상단에 `from __future__ import annotations`. 애노테이션 위치의 소문자 제네릭(`dict[...]`, `list[...]`, `X | None`)은 안전(지연 평가). 모듈 최상위 실행 시점 제네릭 별칭(`Foo = Callable[...]` 형태)은 새로 만들지 않는다 — 기존 `jetson_app.calibration.TrainFn`을 그대로 재사용한다.
- **새 의존성 없음**: torch/paho-mqtt/pyyaml 모두 이미 설치되어 있다. `pyproject.toml`을 건드리지 않는다.
- **자동 온라인 재학습 금지 원칙 유지**: 이 계획은 저장된 모델을 "불러와서 채점"만 한다. 모니터링 중 점수를 이용해 모델을 갱신하는 로직을 추가하지 않는다.
- **디바운스/임계값은 코드에 하드코딩**: `DEFAULT_THRESHOLD = 3.0`(z-score), `DEFAULT_CONFIRM_TICKS = 3`(스펙 6절 "기본 3틱"과 동일) — `config.yaml`이나 CLI 플래그로 아직 노출하지 않는다(향후 실제 데이터로 튜닝 예정, 스펙의 "미해결 항목"에 이미 명시됨).
- **발행 스키마는 상위 아키텍처 문서 3절과 정확히 일치**: `{"records": [{"timestamp": ISO8601, "jetson:anomaly_score": float, "jetson:alarm": bool, "jetson:top_deviant_tag": str}]}`.
- **학습/추론 정규화·오차 계산 로직은 반드시 공유 함수로 재사용**한다(`training.py`의 함수를 `inference.py`가 그대로 호출) — 두 곳에 같은 수식을 따로 구현하면 미묘하게 어긋나는 리스크가 있다(이전 계획 최종 리뷰에서 실제로 지적된 문제).
- **기존 파일 시그니처 변경으로 인한 파급**: `CalibrationManager.__init__`(새 필수 인자 `state_store`)과 `build_pipeline`(새 필수 인자 `model_dir`)이 바뀐다. 이 두 시그니처를 직접 호출하는 **기존 코드/테스트 전부**(`src/`, `tests/` 양쪽)를 해당 태스크 안에서 함께 고쳐서, 그 태스크가 끝날 때 전체 테스트 스위트가 0 실패여야 한다 — 다음 태스크로 파급을 미루지 않는다.
- **MQTT 발행은 기존 구독 클라이언트를 그대로 재사용**한다(`MqttRecordSubscriber.client`) — 별도 연결을 새로 만들지 않는다. paho-mqtt의 `Client.publish()`는 `loop_forever()`가 도는 스레드와 다른 스레드에서 호출해도 안전하다(문서화된 표준 사용법).

---

### Task 1: 태그 타입→인덱스 파생 공유 헬퍼 (`tag_stats.py` 확장)

**Files:**
- Modify: `jetson_app/src/jetson_app/tag_stats.py`
- Test: `jetson_app/tests/test_tag_stats.py` (기존 파일에 추가)

**Interfaces:**
- Consumes: 없음(순수 파이썬, 기존 `tag_stats.py`의 `TagType`만 사용)
- Produces: `type_indices(tags: tuple[str, ...], tag_types: dict[str, str]) -> tuple[list[int], list[int]]` (continuous_indices, binary_indices) — Task 2(`training.py`)와 Task 5(`inference.py`)가 태그 인덱스를 파생할 때 **오직 이 함수만** 사용한다(학습 쪽과 추론 쪽이 각자 따로 파생 로직을 만들면 어긋날 위험이 있어서, 한 곳으로 통일).

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_tag_stats.py` 파일 끝에 추가:

```python
from jetson_app.tag_stats import type_indices


def test_type_indices_separates_continuous_and_binary():
    tags = ("a", "b", "c")
    tag_types = {"a": "binary", "b": "continuous", "c": "binary"}
    continuous_indices, binary_indices = type_indices(tags, tag_types)
    assert continuous_indices == [1]
    assert binary_indices == [0, 2]


def test_type_indices_all_continuous():
    tags = ("x", "y")
    tag_types = {"x": "continuous", "y": "continuous"}
    continuous_indices, binary_indices = type_indices(tags, tag_types)
    assert continuous_indices == [0, 1]
    assert binary_indices == []


def test_type_indices_all_binary():
    tags = ("x", "y")
    tag_types = {"x": "binary", "y": "binary"}
    continuous_indices, binary_indices = type_indices(tags, tag_types)
    assert continuous_indices == []
    assert binary_indices == [0, 1]


def test_type_indices_preserves_tags_order():
    tags = ("c", "a", "b")
    tag_types = {"a": "binary", "b": "continuous", "c": "continuous"}
    continuous_indices, binary_indices = type_indices(tags, tag_types)
    assert continuous_indices == [0, 2]  # c(0), b(2) — tags 순서 기준
    assert binary_indices == [1]  # a(1)
```

(맨 위 `from jetson_app.calibration import CalibrationSample` 등 기존 import는 그대로 두고, 위 import/테스트만 추가한다.)

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd jetson_app && uv run pytest tests/test_tag_stats.py -v`
Expected: FAIL (`ImportError: cannot import name 'type_indices'`)

- [ ] **Step 3: 구현**

`jetson_app/src/jetson_app/tag_stats.py`의 `compute_normalization_stats` 함수 **뒤**에 추가:

```python
def type_indices(tags: tuple[str, ...], tag_types: dict[str, str]) -> tuple[list[int], list[int]]:
    """tag_types(문자열 값 — ModelArtifact에 저장되는 형태)를 기준으로, tags 순서에
    맞는 연속/이진 태그의 인덱스 목록을 만든다. 학습(training.py)과 실시간 추론
    (inference.py)이 태그 인덱스를 항상 이 함수로만 파생시켜, 두 군데서 파생 로직이
    어긋나는 것을 방지한다."""
    continuous_indices = [i for i, t in enumerate(tags) if tag_types[t] == TagType.CONTINUOUS.value]
    binary_indices = [i for i, t in enumerate(tags) if tag_types[t] == TagType.BINARY.value]
    return continuous_indices, binary_indices
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd jetson_app && uv run pytest tests/test_tag_stats.py -v`
Expected: PASS (11 passed — 기존 7개 + 신규 4개)

- [ ] **Step 5: 커밋**

```bash
cd jetson_app
git add src/jetson_app/tag_stats.py tests/test_tag_stats.py
git commit -m "feat: add shared tag-type-to-index derivation helper"
```

---

### Task 2: `training.py` 리팩터링 — 정규화/오차계산 공유 함수 추출

**Files:**
- Modify: `jetson_app/src/jetson_app/training.py`
- Test: `jetson_app/tests/test_training.py` (기존 파일에 추가)

**Interfaces:**
- Consumes: `type_indices`(Task 1, `tag_stats.py`)
- Produces: `normalize_continuous_columns(X, y, tags, continuous_indices, norm_stats: dict[str, tuple[float,float]]) -> None`(in-place), `compute_raw_errors(model, X, y, tags, continuous_indices, binary_indices, batch_size) -> dict[str, torch.Tensor]`(태그별 배치 전체의 원본 절대오차), `model_artifact_path(model_dir: str | Path, equipment_id: str) -> Path` — Task 5(`inference.py`)가 `normalize_continuous_columns`/`compute_raw_errors`를 그대로 재사용하고, Task 8(`pipeline.py`/`subscriber_cli.py`)이 `model_artifact_path`를 재사용한다.

**동작은 바뀌지 않는다** — 기존 `train_model`/`_compute_error_stats`의 로직을 그대로 함수 경계만 다시 그리는 리팩터링이다. 기존 `test_training.py`의 모든 테스트가 수정 없이 계속 통과해야 한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_training.py` 파일 끝에 추가 (파일 상단 import에 `from pathlib import Path`가 이미 있는지 확인하고 없으면 추가):

```python
import torch

from jetson_app.model import AnomalyGRU
from jetson_app.training import compute_raw_errors, model_artifact_path, normalize_continuous_columns


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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd jetson_app && uv run pytest tests/test_training.py -v`
Expected: FAIL (`ImportError: cannot import name 'compute_raw_errors'`)

- [ ] **Step 3: 리팩터링 구현**

`jetson_app/src/jetson_app/training.py`에서:

1. `from .tag_stats import TagType, compute_normalization_stats, detect_tag_types` 를 `from .tag_stats import compute_normalization_stats, detect_tag_types, type_indices` 로 교체 (더 이상 `TagType`을 직접 쓰지 않는다).

2. `train_model` 함수의 아래 부분:

```python
    tag_types = detect_tag_types(samples, tags)
    norm_stats = compute_normalization_stats(samples, tags, tag_types)

    continuous_indices = [i for i, t in enumerate(tags) if tag_types[t] == TagType.CONTINUOUS]
    binary_indices = [i for i, t in enumerate(tags) if tag_types[t] == TagType.BINARY]

    X, y = build_windows(samples, tags, window_size)
    if X.shape[0] == 0:
```

를 다음으로 교체:

```python
    tag_types = detect_tag_types(samples, tags)
    norm_stats = compute_normalization_stats(samples, tags, tag_types)
    tag_types_str = {t: tag_types[t].value for t in tags}
    continuous_indices, binary_indices = type_indices(tags, tag_types_str)

    X, y = build_windows(samples, tags, window_size)
    if X.shape[0] == 0:
```

3. 이어지는 부분:

```python
    for idx in continuous_indices:
        tag = tags[idx]
        stats = norm_stats[tag]
        X[:, :, idx] = (X[:, :, idx] - stats.mean) / stats.std
        y[:, idx] = (y[:, idx] - stats.mean) / stats.std

    model = AnomalyGRU(
```

를 다음으로 교체:

```python
    norm_stats_tuples = {t: (s.mean, s.std) for t, s in norm_stats.items()}
    normalize_continuous_columns(X, y, tags, continuous_indices, norm_stats_tuples)

    model = AnomalyGRU(
```

4. `train_model`의 반환문에서:

```python
    return ModelArtifact(
        tags=tags,
        tag_types={t: tag_types[t].value for t in tags},
        norm_stats={t: (s.mean, s.std) for t, s in norm_stats.items()},
```

를 다음으로 교체(이미 위에서 만든 변수 재사용):

```python
    return ModelArtifact(
        tags=tags,
        tag_types=tag_types_str,
        norm_stats=norm_stats_tuples,
```

5. `_compute_error_stats` 함수를 아래 **두 함수**로 나눈다 — 기존 함수 전체를 삭제하고 이걸로 교체:

```python
def compute_raw_errors(
    model: AnomalyGRU,
    X: torch.Tensor,
    y: torch.Tensor,
    tags: tuple[str, ...],
    continuous_indices: list[int],
    binary_indices: list[int],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, torch.Tensor]:
    """정규화된 X(윈도우 배치)로 다음 시점을 예측하고, 정규화된 실제값 y와의 절대오차를
    태그별로 반환한다(태그당 shape (n,) 텐서, n=배치 크기). 학습 시 "정상 오차" 통계
    산출(_compute_error_stats)과 실시간 추론(inference.py)의 단일 샘플 오차 계산이
    이 함수를 공유한다 — 두 곳에서 예측/오차 수식이 어긋나지 않도록 하기 위함이다.
    전체 X를 한 번에 forward하면 메모리가 무제한으로 커지므로 batch_size 단위로 나눠 돈다."""
    model.eval()
    continuous_chunks: list[torch.Tensor] = []
    binary_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            chunk_continuous, chunk_binary = model(X[start : start + batch_size])
            if chunk_continuous is not None:
                continuous_chunks.append(chunk_continuous)
            if chunk_binary is not None:
                binary_chunks.append(chunk_binary)
    continuous_out = torch.cat(continuous_chunks) if continuous_chunks else None
    binary_logits = torch.cat(binary_chunks) if binary_chunks else None

    errors: dict[str, torch.Tensor] = {}
    for pos, idx in enumerate(continuous_indices):
        errors[tags[idx]] = torch.abs(y[:, idx] - continuous_out[:, pos])
    for pos, idx in enumerate(binary_indices):
        pred_prob = torch.sigmoid(binary_logits[:, pos])
        errors[tags[idx]] = torch.abs(y[:, idx] - pred_prob)
    return errors


def _compute_error_stats(
    model: AnomalyGRU,
    X: torch.Tensor,
    y: torch.Tensor,
    tags: tuple[str, ...],
    continuous_indices: list[int],
    binary_indices: list[int],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, tuple[float, float]]:
    """학습 완료 후 캘리브레이션 데이터 전체에 대한 태그별 "정상 오차" 평균/표준편차.
    실시간 이상 점수 계산(inference.py)에서 원본 오차를 재정규화하는 기준으로 쓰인다."""
    errors = compute_raw_errors(model, X, y, tags, continuous_indices, binary_indices, batch_size)

    result: dict[str, tuple[float, float]] = {}
    for tag, err in errors.items():
        mean = err.mean().item()
        if err.numel() > 1:
            variance = ((err - err.mean()) ** 2).mean().item()
            std = variance ** 0.5
        else:
            std = 0.0
        result[tag] = (mean, _floor_std(mean, std))
    return result
```

6. 아까 지운 `normalize_continuous_columns` 함수를, `_floor_std` 함수 **바로 앞**에 추가:

```python
def normalize_continuous_columns(
    X: torch.Tensor,
    y: torch.Tensor,
    tags: tuple[str, ...],
    continuous_indices: list[int],
    norm_stats: dict[str, tuple[float, float]],
) -> None:
    """continuous_indices에 해당하는 열을 (값-평균)/표준편차로 정규화한다(in-place).
    binary 열은 건드리지 않는다. 학습(train_model)과 실시간 추론(inference.py) 양쪽이
    반드시 동일한 정규화를 쓰도록 공유한다."""
    for idx in continuous_indices:
        tag = tags[idx]
        mean, std = norm_stats[tag]
        X[:, :, idx] = (X[:, :, idx] - mean) / std
        y[:, idx] = (y[:, idx] - mean) / std
```

7. 파일 끝(`make_train_fn` 뒤)에 추가:

```python
def model_artifact_path(model_dir: str | Path, equipment_id: str) -> Path:
    return Path(model_dir) / f"{equipment_id}.pt"
```

- [ ] **Step 4: 테스트 통과 확인 — 신규 + 기존 전부**

Run: `cd jetson_app && uv run pytest tests/test_training.py -v`
Expected: PASS, 기존 테스트(`train_model`/`save_artifact`/`load_artifact`/`make_train_fn` 관련) 전부 포함해서 전부 통과 — 동작이 바뀌지 않았는지 확인하는 것이 이 스텝의 핵심.

Run: `cd jetson_app && uv run pytest -q`
Expected: 전체 스위트 통과, 실패 없음.

- [ ] **Step 5: 커밋**

```bash
cd jetson_app
git add src/jetson_app/training.py tests/test_training.py
git commit -m "refactor: extract shared normalization and raw-error functions from training"
```

---

### Task 3: 캘리브레이션 상태 영속화 (`StateStore`)

**Files:**
- Modify: `jetson_app/src/jetson_app/calibration.py`
- Test: `jetson_app/tests/test_calibration.py` (기존 파일 수정 + 추가)

**Interfaces:**
- Consumes: 없음(표준 라이브러리만)
- Produces: `StateStore(path: str | Path)` — `.read() -> CalibrationState`(파일 없거나 손상되면 `CALIBRATING` 기본값), `.write(state: CalibrationState) -> None`. `CalibrationManager.__init__`이 이제 `state_store: StateStore` 필수 인자를 받아 생성 시점에 `self.state = state_store.read()`로 초기화하고, `handle_train_command`/`handle_recalibrate_command` 성공 시마다 `state_store.write(self.state)`로 갱신한다. Task 8(`pipeline.py`)이 재시작 시 이 초기 상태를 보고 모델 로드 여부를 결정한다.

**⚠️ 파급 효과**: `CalibrationManager.__init__`의 시그니처가 바뀐다. `grep -rn "CalibrationManager(" jetson_app/src jetson_app/tests`로 **이 프로젝트 전체**에서 `CalibrationManager(...)`를 직접 생성하는 곳을 모두 찾아, 각 호출에 `state_store=StateStore(<적절한 tmp_path>)`를 추가해야 한다(테스트 파일에도 있을 수 있다 — `test_calibration.py` 뿐 아니라 `test_scheduler.py` 등도 확인). 이 태스크가 끝날 때 전체 테스트 스위트가 0 실패여야 한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_calibration.py` 파일 끝에 추가 (파일 상단에 `from pathlib import Path`가 없으면 추가):

```python
from pathlib import Path

from jetson_app.calibration import StateStore


def test_state_store_missing_file_defaults_calibrating(tmp_path: Path):
    store = StateStore(tmp_path / "line_A.state")
    assert store.read() == CalibrationState.CALIBRATING


def test_state_store_round_trip(tmp_path: Path):
    store = StateStore(tmp_path / "line_A.state")
    store.write(CalibrationState.MONITORING)
    assert store.read() == CalibrationState.MONITORING
    store.write(CalibrationState.CALIBRATING)
    assert store.read() == CalibrationState.CALIBRATING


def test_state_store_corrupted_content_defaults_calibrating(tmp_path: Path):
    path = tmp_path / "line_A.state"
    path.write_text("garbage", encoding="utf-8")
    store = StateStore(path)
    assert store.read() == CalibrationState.CALIBRATING


def test_state_store_creates_parent_directory(tmp_path: Path):
    store = StateStore(tmp_path / "nested" / "line_A.state")
    store.write(CalibrationState.MONITORING)
    assert store.read() == CalibrationState.MONITORING


def test_calibration_manager_resumes_state_from_store(tmp_path: Path):
    state_path = tmp_path / "state"
    StateStore(state_path).write(CalibrationState.MONITORING)
    manager = CalibrationManager(
        buffer_writer=CalibrationBufferWriter(tmp_path / "buf.jsonl"),
        min_samples=1,
        max_duration=timedelta(days=7),
        train_fn=lambda samples: None,
        state_store=StateStore(state_path),
    )
    assert manager.state == CalibrationState.MONITORING


def test_calibration_manager_persists_state_on_train(tmp_path: Path):
    state_path = tmp_path / "state"
    buffer_writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")
    buffer_writer.append(Snapshot(values={"a": 1.0}), "2026-01-01T00:00:00+00:00")
    manager = CalibrationManager(
        buffer_writer=buffer_writer,
        min_samples=1,
        max_duration=timedelta(days=7),
        train_fn=lambda samples: None,
        state_store=StateStore(state_path),
    )
    manager.handle_train_command()
    assert StateStore(state_path).read() == CalibrationState.MONITORING


def test_calibration_manager_persists_state_on_recalibrate(tmp_path: Path):
    state_path = tmp_path / "state"
    StateStore(state_path).write(CalibrationState.MONITORING)
    manager = CalibrationManager(
        buffer_writer=CalibrationBufferWriter(tmp_path / "buf.jsonl"),
        min_samples=1,
        max_duration=timedelta(days=7),
        train_fn=lambda samples: None,
        state_store=StateStore(state_path),
    )
    manager.handle_recalibrate_command()
    assert StateStore(state_path).read() == CalibrationState.CALIBRATING
```

(`Snapshot`과 `timedelta` 등은 파일에 이미 import되어 있을 가능성이 높다 — 없으면 상단에 `from datetime import timedelta`, `from jetson_app.buffer import Snapshot`을 추가한다. 기존 파일의 다른 `CalibrationManager(...)` 생성 호출들도 이 스텝에서 함께 `state_store=StateStore(tmp_path / "...")`를 추가해서 고쳐야 다음 스텝의 RED가 "의도한 이유"로만 실패한다.)

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd jetson_app && uv run pytest tests/test_calibration.py -v`
Expected: FAIL (`ImportError: cannot import name 'StateStore'`, 그리고 기존 `CalibrationManager(...)` 호출들은 `state_store` 누락으로 `TypeError`)

- [ ] **Step 3: 구현**

`jetson_app/src/jetson_app/calibration.py`에서 `CalibrationState` Enum 정의 **바로 뒤, `CalibrationSample` 앞**에 추가:

```python
class StateStore:
    """CALIBRATING/MONITORING 상태를 파일로 영속화해, Jetson 재시작 시 이어서
    복구할 수 있게 한다. 모델 파일과 별도로 관리한다 — recalibrate는 모델 파일은
    남겨두고 이 마커만 CALIBRATING으로 되돌리므로, 재시작 시 반드시 이 마커를
    기준으로 판단해야 한다(모델 파일이 있다고 바로 MONITORING으로 재개하면 안 된다)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def read(self) -> CalibrationState:
        if not self._path.exists():
            return CalibrationState.CALIBRATING
        text = self._path.read_text(encoding="utf-8").strip()
        try:
            return CalibrationState(text)
        except ValueError:
            return CalibrationState.CALIBRATING

    def write(self, state: CalibrationState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(state.value, encoding="utf-8")
```

`CalibrationManager.__init__`을:

```python
    def __init__(
        self,
        buffer_writer: CalibrationBufferWriter,
        min_samples: int,
        max_duration: timedelta,
        train_fn: TrainFn,
    ) -> None:
        self._buffer_writer = buffer_writer
        self._min_samples = min_samples
        self._max_duration = max_duration
        self._train_fn = train_fn
        self._lock = threading.Lock()
        self._tick_count = 0
        self.state = CalibrationState.CALIBRATING
```

에서 아래로 교체:

```python
    def __init__(
        self,
        buffer_writer: CalibrationBufferWriter,
        min_samples: int,
        max_duration: timedelta,
        train_fn: TrainFn,
        state_store: StateStore,
    ) -> None:
        self._buffer_writer = buffer_writer
        self._min_samples = min_samples
        self._max_duration = max_duration
        self._train_fn = train_fn
        self._state_store = state_store
        self._lock = threading.Lock()
        self._tick_count = 0
        self.state = state_store.read()
```

`handle_train_command`의 `self.state = CalibrationState.MONITORING` 줄 바로 뒤에 `self._state_store.write(self.state)`를 추가:

```python
            self._train_fn(samples)
            self._buffer_writer.clear()
            self.state = CalibrationState.MONITORING
            self._state_store.write(self.state)
```

`handle_recalibrate_command`의 `self.state = CalibrationState.CALIBRATING` 줄 바로 뒤에도 동일하게 추가:

```python
    def handle_recalibrate_command(self) -> None:
        with self._lock:
            self._buffer_writer.clear()
            self.state = CalibrationState.CALIBRATING
            self._state_store.write(self.state)
```

- [ ] **Step 4: 전체 파급 효과 수정**

`grep -rn "CalibrationManager(" jetson_app/src jetson_app/tests`로 이 파일 밖의 모든 호출 지점(다른 소스 파일, 다른 테스트 파일)을 찾아, 각각에 `state_store=StateStore(<그 테스트의 tmp_path 기반 경로>)`를 추가한다. (`src/jetson_app/pipeline.py`의 `build_pipeline`이 `CalibrationManager(...)`를 생성하는 곳도 여기 포함된다 — 지금은 최소한으로 `state_store=StateStore(Path(calibration_dir) / f"{config.equipment_id}.state")`처럼 임시로 연결해 컴파일이 되게만 해둔다. 이 상태 파일을 어디에 어떻게 제대로 두는지는 Task 8에서 `model_dir` 기준으로 다시 정리하므로, 지금은 "빌드가 깨지지 않고 전체 테스트가 통과하는 것"만 목표로 한다.)

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd jetson_app && uv run pytest -q`
Expected: 전체 스위트 통과, 실패 없음.

- [ ] **Step 6: 커밋**

```bash
cd jetson_app
git add -A
git commit -m "feat: persist calibration state across restarts via StateStore"
```

---

### Task 4: 디바운스 (`debounce.py`)

**Files:**
- Create: `jetson_app/src/jetson_app/debounce.py`
- Test: `jetson_app/tests/test_debounce.py`

**Interfaces:**
- Consumes: 없음
- Produces: `Debouncer(threshold: float = 3.0, confirm_ticks: int = 3)` — `.update(score: float) -> bool`(연속 confirm_ticks번 이상 threshold를 넘어야 True, 한 번이라도 밑돌면 카운터 리셋). `DEFAULT_THRESHOLD`, `DEFAULT_CONFIRM_TICKS` 모듈 상수. Task 7(`scheduler.py`)과 Task 8(`pipeline.py`)이 사용한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_debounce.py`:

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd jetson_app && uv run pytest tests/test_debounce.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'jetson_app.debounce'`)

- [ ] **Step 3: 구현**

`jetson_app/src/jetson_app/debounce.py`:

```python
from __future__ import annotations

DEFAULT_THRESHOLD = 3.0
DEFAULT_CONFIRM_TICKS = 3


class Debouncer:
    """이상 점수가 threshold를 confirm_ticks번 연속으로 넘어야 알람을 확정한다.
    한 틱이라도 threshold 밑으로 내려가면 카운터를 리셋한다 — 순간적 노이즈로 인한
    오탐을 줄이기 위함(설계 스펙 2026-08-04 문서 6절 5번)."""

    def __init__(
        self, threshold: float = DEFAULT_THRESHOLD, confirm_ticks: int = DEFAULT_CONFIRM_TICKS
    ) -> None:
        if confirm_ticks <= 0:
            raise ValueError("confirm_ticks must be positive")
        self._threshold = threshold
        self._confirm_ticks = confirm_ticks
        self._consecutive_over = 0

    def update(self, score: float) -> bool:
        """새 점수를 반영하고, 현재 알람이 확정 상태인지 반환한다."""
        if score >= self._threshold:
            self._consecutive_over += 1
        else:
            self._consecutive_over = 0
        return self._consecutive_over >= self._confirm_ticks
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd jetson_app && uv run pytest tests/test_debounce.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: 커밋**

```bash
cd jetson_app
git add src/jetson_app/debounce.py tests/test_debounce.py
git commit -m "feat: add anomaly-score debouncer"
```

---

### Task 5: 실시간 추론 엔진 (`inference.py`)

**Files:**
- Create: `jetson_app/src/jetson_app/inference.py`
- Test: `jetson_app/tests/test_inference.py`

**Interfaces:**
- Consumes: `type_indices`(Task 1), `normalize_continuous_columns`/`compute_raw_errors`/`ModelArtifact`(Task 2), `AnomalyGRU`(기존 `model.py`), `Snapshot`(기존 `buffer.py`)
- Produces: `AnomalyResult(anomaly_score: float, top_deviant_tag: str)` frozen dataclass, `InferenceEngine(artifact: ModelArtifact)` — `.score(window: list[Snapshot], actual: Snapshot) -> AnomalyResult | None`, `ActiveModelHolder` — 스레드세이프 `.get() -> InferenceEngine | None` / `.set(engine)`. Task 7(`scheduler.py`)과 Task 8(`pipeline.py`)이 사용한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_inference.py`:

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd jetson_app && uv run pytest tests/test_inference.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'jetson_app.inference'`)

- [ ] **Step 3: 구현**

`jetson_app/src/jetson_app/inference.py`:

```python
from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from .buffer import Snapshot
from .model import AnomalyGRU
from .tag_stats import type_indices
from .training import ModelArtifact, compute_raw_errors, normalize_continuous_columns


@dataclass(frozen=True)
class AnomalyResult:
    anomaly_score: float
    top_deviant_tag: str


class InferenceEngine:
    """저장된 ModelArtifact로 GRU를 복원해, 실시간 윈도우로 다음 시점을 예측하고
    실제값과의 오차를 캘리브레이션 구간 "정상 오차" 통계로 재정규화해 이상 점수를 낸다
    (설계 스펙 2026-08-04 문서 6절)."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self._artifact = artifact
        self._tags = artifact.tags
        self._continuous_indices, self._binary_indices = type_indices(
            artifact.tags, artifact.tag_types
        )
        self._model = AnomalyGRU(
            num_tags=len(artifact.tags),
            continuous_indices=self._continuous_indices,
            binary_indices=self._binary_indices,
            hidden_size=artifact.hidden_size,
            num_layers=artifact.num_layers,
        )
        self._model.load_state_dict(artifact.state_dict)
        self._model.eval()

    def score(self, window: list[Snapshot], actual: Snapshot) -> AnomalyResult | None:
        """window(길이 window_size, 오래된→최신 순)로 다음 시점을 예측하고, 실제로
        도착한 actual과 비교해 이상 점수를 계산한다. window나 actual에 한 번도 관측
        안 된(None) 태그가 있으면 None을 반환해 그 틱의 채점을 건너뛴다(학습 시
        build_windows가 None 포함 윈도우를 버리는 것과 동일한 정책)."""
        if len(window) != self._artifact.window_size:
            return None

        window_rows: list[list[float]] = []
        for snapshot in window:
            row = [snapshot.values.get(tag) for tag in self._tags]
            if any(v is None for v in row):
                return None
            window_rows.append([float(v) for v in row])

        actual_row = [actual.values.get(tag) for tag in self._tags]
        if any(v is None for v in actual_row):
            return None

        X = torch.tensor([window_rows], dtype=torch.float32)
        y = torch.tensor([[float(v) for v in actual_row]], dtype=torch.float32)
        normalize_continuous_columns(
            X, y, self._tags, self._continuous_indices, self._artifact.norm_stats
        )

        raw_errors = compute_raw_errors(
            self._model,
            X,
            y,
            self._tags,
            self._continuous_indices,
            self._binary_indices,
            batch_size=1,
        )

        best_tag: str | None = None
        best_z: float | None = None
        for tag, err_tensor in raw_errors.items():
            raw_error = err_tensor.item()
            error_mean, error_std = self._artifact.error_stats[tag]
            z = (raw_error - error_mean) / error_std
            if best_z is None or z > best_z:
                best_z = z
                best_tag = tag

        return AnomalyResult(anomaly_score=best_z, top_deviant_tag=best_tag)


class ActiveModelHolder:
    """PeriodicSnapshotter(매 틱, 읽기)와 학습 완료 콜백(train 명령 시, 쓰기)이 서로
    다른 스레드에서 접근하는 현재 InferenceEngine을 스레드세이프하게 공유한다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine: InferenceEngine | None = None

    def get(self) -> InferenceEngine | None:
        with self._lock:
            return self._engine

    def set(self, engine: InferenceEngine | None) -> None:
        with self._lock:
            self._engine = engine
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd jetson_app && uv run pytest tests/test_inference.py -v`
Expected: PASS (5 passed). 작은 모델을 실제로 학습시키므로 몇 초 걸릴 수 있다.

- [ ] **Step 5: 커밋**

```bash
cd jetson_app
git add src/jetson_app/inference.py tests/test_inference.py
git commit -m "feat: add real-time InferenceEngine and ActiveModelHolder"
```

---

### Task 6: 결과 발행자 (`publisher.py`)

**Files:**
- Create: `jetson_app/src/jetson_app/publisher.py`
- Test: `jetson_app/tests/test_publisher.py`

**Interfaces:**
- Consumes: 없음(MQTT client는 duck-typing으로 `.publish(topic, payload)`만 있으면 됨)
- Produces: `ResultPublisher(client, publish_topic: str)` — `.publish(timestamp: str, anomaly_score: float, alarm: bool, top_deviant_tag: str) -> None`. Task 7(`scheduler.py`)과 Task 8(`pipeline.py`)이 사용한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_publisher.py`:

```python
import json

from jetson_app.publisher import ResultPublisher


class _FakeClient:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload):
        self.published.append((topic, payload))


def test_publish_sends_expected_schema_to_configured_topic():
    client = _FakeClient()
    publisher = ResultPublisher(client=client, publish_topic="jetson/line_A/anomaly")
    publisher.publish(
        timestamp="2026-08-06T00:00:00+00:00",
        anomaly_score=4.2,
        alarm=True,
        top_deviant_tag="PLC_Collector_Actuator_1:AirBlower.Cmd[0]",
    )
    assert len(client.published) == 1
    topic, payload = client.published[0]
    assert topic == "jetson/line_A/anomaly"
    data = json.loads(payload)
    assert data == {
        "records": [
            {
                "timestamp": "2026-08-06T00:00:00+00:00",
                "jetson:anomaly_score": 4.2,
                "jetson:alarm": True,
                "jetson:top_deviant_tag": "PLC_Collector_Actuator_1:AirBlower.Cmd[0]",
            }
        ]
    }


def test_publish_multiple_calls_each_send_one_message():
    client = _FakeClient()
    publisher = ResultPublisher(client=client, publish_topic="t")
    publisher.publish("ts1", 1.0, False, "tag1")
    publisher.publish("ts2", 2.0, True, "tag2")
    assert len(client.published) == 2
    assert client.published[0][0] == "t"
    assert client.published[1][0] == "t"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd jetson_app && uv run pytest tests/test_publisher.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'jetson_app.publisher'`)

- [ ] **Step 3: 구현**

`jetson_app/src/jetson_app/publisher.py`:

```python
from __future__ import annotations

import json


class ResultPublisher:
    """이상 점수/알람/최대 기여 태그를 상위 아키텍처 문서 3절의 발행 스키마로
    설비 config의 publish_topic에 MQTT 발행한다."""

    def __init__(self, client, publish_topic: str) -> None:
        self._client = client
        self._publish_topic = publish_topic

    def publish(
        self, timestamp: str, anomaly_score: float, alarm: bool, top_deviant_tag: str
    ) -> None:
        payload = json.dumps(
            {
                "records": [
                    {
                        "timestamp": timestamp,
                        "jetson:anomaly_score": anomaly_score,
                        "jetson:alarm": alarm,
                        "jetson:top_deviant_tag": top_deviant_tag,
                    }
                ]
            }
        )
        self._client.publish(self._publish_topic, payload)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd jetson_app && uv run pytest tests/test_publisher.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
cd jetson_app
git add src/jetson_app/publisher.py tests/test_publisher.py
git commit -m "feat: add MQTT ResultPublisher"
```

---

### Task 7: 스케줄러에 추론/디바운스/발행 연결 (`scheduler.py`)

**Files:**
- Modify: `jetson_app/src/jetson_app/scheduler.py`
- Test: `jetson_app/tests/test_scheduler.py` (기존 파일에 추가, 기존 테스트는 수정 없이 통과해야 함)

**Interfaces:**
- Consumes: `CalibrationState`(기존 `calibration.py`), `ActiveModelHolder`(Task 5), `Debouncer`(Task 4), `ResultPublisher`(Task 6)
- Produces: `PeriodicSnapshotter.__init__`이 새 선택 인자 `inference_engine_holder`, `debouncer`, `result_publisher`(모두 기본값 `None`)를 받는다. Task 8(`pipeline.py`)이 이 셋을 채워서 연결한다.

**핵심 순서**: 새 스냅샷을 `sliding_window.push()`하기 **전에** 채점해야 한다 — 채점은 "그 시점까지 쌓인 과거 윈도우로 다음 값을 예측"하는 것이므로, push 후에 채점하면 예측 대상 값이 이미 윈도우 안에 들어가 있어 무의미해진다.

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_scheduler.py` 파일 끝에 추가 (파일 상단에 `from jetson_app.buffer import SlidingWindow, TagBuffer` 등 필요한 import가 이미 있는지 확인하고, 없는 것만 추가):

```python
from jetson_app.buffer import Snapshot
from jetson_app.calibration import CalibrationState
from jetson_app.inference import AnomalyResult


class _FakeCalibrationManager:
    def __init__(self, state):
        self.state = state
        self.recorded = []

    def record_sample(self, snapshot, timestamp):
        self.recorded.append((snapshot, timestamp))


class _FakeEngine:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def score(self, window, actual):
        self.calls.append((list(window), actual))
        return self._result


class _FakeHolder:
    def __init__(self, engine):
        self._engine = engine

    def get(self):
        return self._engine


class _FakeDebouncer:
    def __init__(self, alarm):
        self._alarm = alarm
        self.scores = []

    def update(self, score):
        self.scores.append(score)
        return self._alarm


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, timestamp, anomaly_score, alarm, top_deviant_tag):
        self.published.append((timestamp, anomaly_score, alarm, top_deviant_tag))


def _fill_window(tag_buffer, sliding_window, n):
    for i in range(n):
        tag_buffer.update({"a": float(i)})
        sliding_window.push(tag_buffer.snapshot())


def test_tick_does_not_score_when_calibrating():
    tag_buffer = TagBuffer(("a",))
    sliding_window = SlidingWindow(2)
    _fill_window(tag_buffer, sliding_window, 2)
    calibration_manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    holder = _FakeHolder(engine)
    publisher = _FakePublisher()
    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=50,
        inference_engine_holder=holder,
        debouncer=_FakeDebouncer(alarm=True),
        result_publisher=publisher,
    )
    tag_buffer.update({"a": 99.0})
    snapshotter._tick()
    assert engine.calls == []
    assert publisher.published == []
    assert calibration_manager.recorded  # CALIBRATING이어도 캘리브레이션 기록은 계속됨


def test_tick_does_not_score_when_window_not_full():
    tag_buffer = TagBuffer(("a",))
    sliding_window = SlidingWindow(5)
    _fill_window(tag_buffer, sliding_window, 2)  # 용량 5인데 2개뿐
    calibration_manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    holder = _FakeHolder(engine)
    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=50,
        inference_engine_holder=holder,
        debouncer=_FakeDebouncer(alarm=False),
        result_publisher=_FakePublisher(),
    )
    tag_buffer.update({"a": 99.0})
    snapshotter._tick()
    assert engine.calls == []


def test_tick_scores_with_pre_push_window_when_monitoring_and_full():
    tag_buffer = TagBuffer(("a",))
    sliding_window = SlidingWindow(2)
    _fill_window(tag_buffer, sliding_window, 2)
    pre_push_window = sliding_window.to_list()

    calibration_manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    holder = _FakeHolder(engine)
    debouncer = _FakeDebouncer(alarm=True)
    publisher = _FakePublisher()
    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=50,
        inference_engine_holder=holder,
        debouncer=debouncer,
        result_publisher=publisher,
    )
    tag_buffer.update({"a": 99.0})
    snapshotter._tick()

    assert len(engine.calls) == 1
    scored_window, scored_actual = engine.calls[0]
    assert scored_window == pre_push_window  # push되기 *전* 윈도우로 채점됐는지 확인
    assert scored_actual.values == {"a": 99.0}
    assert debouncer.scores == [5.0]
    assert len(publisher.published) == 1
    _, anomaly_score, alarm, top_deviant_tag = publisher.published[0]
    assert anomaly_score == 5.0
    assert alarm is True
    assert top_deviant_tag == "a"


def test_tick_skips_publish_when_engine_returns_none():
    tag_buffer = TagBuffer(("a",))
    sliding_window = SlidingWindow(2)
    _fill_window(tag_buffer, sliding_window, 2)
    calibration_manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(None)
    holder = _FakeHolder(engine)
    publisher = _FakePublisher()
    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=50,
        inference_engine_holder=holder,
        debouncer=_FakeDebouncer(alarm=False),
        result_publisher=publisher,
    )
    tag_buffer.update({"a": 99.0})
    snapshotter._tick()
    assert publisher.published == []


def test_tick_without_inference_collaborators_still_records_calibration():
    tag_buffer = TagBuffer(("a",))
    sliding_window = SlidingWindow(2)
    calibration_manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=50,
    )
    tag_buffer.update({"a": 1.0})
    snapshotter._tick()
    assert calibration_manager.recorded
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd jetson_app && uv run pytest tests/test_scheduler.py -v`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'inference_engine_holder'`)

- [ ] **Step 3: 구현**

`jetson_app/src/jetson_app/scheduler.py` 전체를 아래로 교체:

```python
from __future__ import annotations

import threading
from datetime import datetime, timezone

from .buffer import SlidingWindow, TagBuffer
from .calibration import CalibrationManager, CalibrationState
from .debounce import Debouncer
from .inference import ActiveModelHolder
from .publisher import ResultPublisher

_HEARTBEAT_TICK_INTERVAL = 100  # 기본 50ms 간격 기준 약 5초마다


class PeriodicSnapshotter:
    """`interval_ms`마다 TagBuffer 스냅샷을 SlidingWindow에 밀어넣고,
    CalibrationManager에도 전달한다 (CALIBRATING 상태일 때만 실제로 저장됨).
    MONITORING 상태이고 학습된 모델(inference_engine_holder)이 있으면, 새 스냅샷을
    윈도우에 넣기 *전에* 그 시점까지의 윈도우로 다음 값을 예측해 이상 점수를 계산하고
    발행한다(스냅샷을 먼저 넣으면 "미래"를 보고 예측하는 꼴이 되어 스코어링이
    무의미해진다). 모든 태그가 None인(한 번도 값을 못 받은) 스냅샷은 건너뛴다.
    """

    def __init__(
        self,
        tag_buffer: TagBuffer,
        sliding_window: SlidingWindow,
        calibration_manager: CalibrationManager,
        interval_ms: int,
        inference_engine_holder: ActiveModelHolder | None = None,
        debouncer: Debouncer | None = None,
        result_publisher: ResultPublisher | None = None,
    ) -> None:
        self._tag_buffer = tag_buffer
        self._sliding_window = sliding_window
        self._calibration_manager = calibration_manager
        self._interval_seconds = interval_ms / 1000
        self._inference_engine_holder = inference_engine_holder
        self._debouncer = debouncer
        self._result_publisher = result_publisher
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_count = 0

    def _score_and_publish(self, timestamp: str, snapshot) -> None:
        if self._inference_engine_holder is None:
            return
        if self._calibration_manager.state != CalibrationState.MONITORING:
            return
        if not self._sliding_window.is_full():
            return
        engine = self._inference_engine_holder.get()
        if engine is None:
            return
        result = engine.score(self._sliding_window.to_list(), snapshot)
        if result is None:
            return
        alarm = self._debouncer.update(result.anomaly_score)
        self._result_publisher.publish(
            timestamp, result.anomaly_score, alarm, result.top_deviant_tag
        )

    def _tick(self) -> None:
        snapshot = self._tag_buffer.snapshot()
        if all(value is None for value in snapshot.values.values()):
            return
        timestamp = datetime.now(timezone.utc).isoformat()

        self._score_and_publish(timestamp, snapshot)

        self._sliding_window.push(snapshot)
        self._calibration_manager.record_sample(snapshot, timestamp)
        self._tick_count += 1
        if self._tick_count % _HEARTBEAT_TICK_INTERVAL == 0:
            print(
                f"[snapshotter] {self._tick_count}번째 스냅샷 처리, "
                f"윈도우 {len(self._sliding_window.to_list())}/{self._sliding_window.window_size}, "
                f"캘리브레이션 상태={self._calibration_manager.state.value}"
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                # 백그라운드 스레드가 죽으면 데이터 수집이 조용히 멈추므로
                # 어떤 예외도 로그만 남기고 계속 진행한다.
                print(f"[PeriodicSnapshotter] tick 처리 중 오류 발생, 계속 진행: {e}")
            self._stop_event.wait(self._interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
```

- [ ] **Step 4: 테스트 통과 확인 — 신규 + 기존 전부**

Run: `cd jetson_app && uv run pytest tests/test_scheduler.py -v`
Expected: PASS, 기존 테스트 전부 포함(생성자에 새 인자 없이 만들던 기존 테스트도 기본값 `None`으로 그대로 통과해야 한다).

Run: `cd jetson_app && uv run pytest -q`
Expected: 전체 스위트 통과.

- [ ] **Step 5: 커밋**

```bash
cd jetson_app
git add src/jetson_app/scheduler.py tests/test_scheduler.py
git commit -m "feat: wire inference scoring, debounce, and publishing into snapshotter tick"
```

---

### Task 8: 파이프라인/CLI 최종 통합 (`pipeline.py`, `subscriber_cli.py`)

**Files:**
- Modify: `jetson_app/src/jetson_app/pipeline.py`
- Modify: `jetson_app/src/jetson_app/subscriber_cli.py`
- Test: `jetson_app/tests/test_pipeline.py` (기존 6개 테스트 호출부 수정 + 신규 3개 추가)

**Interfaces:**
- Consumes: 이 계획의 모든 이전 태스크 산출물 전부
- Produces: `Pipeline`이 새 필드 `inference_engine_holder: ActiveModelHolder`를 갖는다. `build_pipeline`이 새 필수 인자 `model_dir: str | Path`를 받는다.

**⚠️ 파급 효과**: `build_pipeline`의 시그니처가 바뀐다. `jetson_app/tests/test_pipeline.py`의 기존 6개 테스트 함수가 모두 `build_pipeline(config=..., calibration_dir=..., train_fn=...)` 형태로 호출하고 있다 — 전부 `model_dir=tmp_path / "model_data"`를 추가해야 한다. 이 태스크가 끝날 때 전체 테스트 스위트가 0 실패여야 한다.

- [ ] **Step 1: 기존 테스트 호출부 수정 + 신규 실패하는 테스트 작성**

`jetson_app/tests/test_pipeline.py`에서 기존 6개 `build_pipeline(...)` 호출 전부에 `model_dir=tmp_path / "model_data",` 한 줄을 `calibration_dir=...,` 다음 줄에 추가한다. 예를 들어:

```python
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        train_fn=lambda samples: None,
    )
```

전부를:

```python
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=tmp_path / "model_data",
        train_fn=lambda samples: None,
    )
```

식으로 바꾼다(`train_fn=...` 자리에 `make_train_fn(...)`이 오는 두 개의 e2e 테스트도 동일하게 `model_dir=tmp_path / "model_data",` 한 줄만 추가). 파일 맨 위 import에 다음을 추가:

```python
from jetson_app.calibration import CalibrationState, StateStore
```

(이미 `CalibrationState`는 import되어 있을 것이므로 `StateStore`만 추가하면 된다.)

그리고 파일 끝에 아래 3개 테스트를 추가한다:

```python
def test_pipeline_scores_and_publishes_after_training(tmp_path):
    config = EquipmentConfig(
        equipment_id="e2e_score",
        subscribe_topics=("dx1/e2e_score/data",),
        publish_topic="jetson/e2e_score/anomaly",
        command_topic="jetson/e2e_score/cmd",
        tags=("tag_a", "tag_b"),
        resample_interval_ms=5,
        window_size=3,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=10),
    )
    model_dir = tmp_path / "model_data"
    train_fn = make_train_fn(
        tags=config.tags,
        window_size=config.window_size,
        model_path=model_dir / f"{config.equipment_id}.pt",
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )

    def _send(i):
        payload = json.dumps(
            {
                "records": [
                    {
                        "timestamp": "2026-08-04T00:00:00+0000",
                        "tag_a": float(i),
                        "tag_b": i % 2,
                    }
                ]
            }
        ).encode("utf-8")
        pipeline.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))

    pipeline.snapshotter.start()
    try:
        for i in range(20):
            _send(i)
            time.sleep(0.01)
    finally:
        pipeline.snapshotter.stop()

    pipeline.command_subscriber._handle_command_message(
        None, None, SimpleNamespace(payload=b'{"command": "train"}')
    )
    assert pipeline.calibration_manager.state == CalibrationState.MONITORING
    assert pipeline.inference_engine_holder.get() is not None

    published = []
    pipeline.mqtt_subscriber.client.publish = lambda topic, payload: published.append(
        (topic, payload)
    )

    pipeline.snapshotter.start()
    try:
        for i in range(20, 30):
            _send(i)
            time.sleep(0.01)
    finally:
        pipeline.snapshotter.stop()

    assert published, "MONITORING 진입 후에도 이상 점수가 발행되지 않았다"
    topic, payload = published[0]
    assert topic == config.publish_topic
    record = json.loads(payload)["records"][0]
    assert "jetson:anomaly_score" in record
    assert "jetson:alarm" in record
    assert record["jetson:top_deviant_tag"] in config.tags


def test_pipeline_resumes_monitoring_after_restart(tmp_path):
    config = EquipmentConfig(
        equipment_id="e2e_resume",
        subscribe_topics=("dx1/e2e_resume/data",),
        publish_topic="jetson/e2e_resume/anomaly",
        command_topic="jetson/e2e_resume/cmd",
        tags=("tag_a", "tag_b"),
        resample_interval_ms=5,
        window_size=3,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=10),
    )
    model_dir = tmp_path / "model_data"
    train_fn = make_train_fn(
        tags=config.tags,
        window_size=config.window_size,
        model_path=model_dir / f"{config.equipment_id}.pt",
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    first.snapshotter.start()
    try:
        for i in range(20):
            payload = json.dumps(
                {
                    "records": [
                        {
                            "timestamp": "2026-08-04T00:00:00+0000",
                            "tag_a": float(i),
                            "tag_b": i % 2,
                        }
                    ]
                }
            ).encode("utf-8")
            first.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))
            time.sleep(0.01)
    finally:
        first.snapshotter.stop()
    first.command_subscriber._handle_command_message(
        None, None, SimpleNamespace(payload=b'{"command": "train"}')
    )
    assert first.calibration_manager.state == CalibrationState.MONITORING

    # "재시작": 같은 calibration_dir/model_dir로 파이프라인을 새로 만든다
    second = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    assert second.calibration_manager.state == CalibrationState.MONITORING
    assert second.inference_engine_holder.get() is not None


def test_pipeline_falls_back_to_calibrating_when_model_file_corrupted(tmp_path):
    config = _make_config(tmp_path)
    model_dir = tmp_path / "model_data"
    model_dir.mkdir(parents=True)
    (model_dir / f"{config.equipment_id}.pt").write_bytes(b"not a real torch file")
    StateStore(model_dir / f"{config.equipment_id}.state").write(CalibrationState.MONITORING)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=lambda samples: None,
    )

    assert pipeline.calibration_manager.state == CalibrationState.CALIBRATING
    assert pipeline.inference_engine_holder.get() is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd jetson_app && uv run pytest tests/test_pipeline.py -v`
Expected: FAIL (`TypeError: build_pipeline() missing 1 required positional argument: 'model_dir'` 등)

- [ ] **Step 3: `pipeline.py` 구현**

`jetson_app/src/jetson_app/pipeline.py` 전체를 아래로 교체:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .buffer import SlidingWindow, TagBuffer
from .calibration import (
    CalibrationBufferWriter,
    CalibrationManager,
    CalibrationState,
    StateStore,
    TrainFn,
)
from .command_subscriber import CommandSubscriber
from .config import EquipmentConfig
from .debounce import Debouncer
from .inference import ActiveModelHolder, InferenceEngine
from .mqtt_subscriber import MqttRecordSubscriber, Record
from .publisher import ResultPublisher
from .scheduler import PeriodicSnapshotter
from .training import load_artifact, model_artifact_path


@dataclass(frozen=True)
class Pipeline:
    config: EquipmentConfig
    tag_buffer: TagBuffer
    sliding_window: SlidingWindow
    calibration_manager: CalibrationManager
    inference_engine_holder: ActiveModelHolder
    snapshotter: PeriodicSnapshotter
    mqtt_subscriber: MqttRecordSubscriber
    command_subscriber: CommandSubscriber


def build_pipeline(
    config: EquipmentConfig,
    calibration_dir: str | Path,
    model_dir: str | Path,
    train_fn: TrainFn,
) -> Pipeline:
    tag_buffer = TagBuffer(config.tags)
    sliding_window = SlidingWindow(config.window_size)

    def on_record(record: Record) -> None:
        tag_buffer.update(record.values)

    mqtt_subscriber = MqttRecordSubscriber(config, on_record=on_record)
    result_publisher = ResultPublisher(
        client=mqtt_subscriber.client, publish_topic=config.publish_topic
    )

    buffer_path = Path(calibration_dir) / f"{config.equipment_id}.jsonl"
    # 잘못된 --calibration-dir이 백그라운드 스레드가 아니라 기동 시점에 드러나도록
    # 디렉터리를 미리 만든다 (CLI의 OSError 처리에 걸린다).
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    buffer_writer = CalibrationBufferWriter(buffer_path)

    model_path = model_artifact_path(model_dir, config.equipment_id)
    state_path = Path(model_dir) / f"{config.equipment_id}.state"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_store = StateStore(state_path)

    inference_engine_holder = ActiveModelHolder()

    def wrapped_train_fn(samples: list) -> None:
        train_fn(samples)
        # 학습이 방금 성공적으로 저장한 모델을 즉시 메모리에 올려, 다음 틱부터
        # 바로 채점을 시작할 수 있게 한다 (재시작을 기다릴 필요 없음).
        inference_engine_holder.set(InferenceEngine(load_artifact(model_path)))

    calibration_manager = CalibrationManager(
        buffer_writer=buffer_writer,
        min_samples=config.calibration.min_samples,
        max_duration=config.calibration.max_duration,
        train_fn=wrapped_train_fn,
        state_store=state_store,
    )

    if calibration_manager.state == CalibrationState.MONITORING:
        try:
            inference_engine_holder.set(InferenceEngine(load_artifact(model_path)))
            print(f"[build_pipeline] 저장된 모델을 불러와 MONITORING으로 재개: {model_path}")
        except Exception as e:
            # 모델 파일 손상/누락 — MONITORING 진입을 막고 CALIBRATING으로 폴백해
            # 사람이 재학습을 판단하도록 한다 (상위 문서 5절 에러 처리 원칙).
            print(f"[build_pipeline] 모델 로드 실패, CALIBRATING으로 폴백: {e}")
            calibration_manager.handle_recalibrate_command()

    debouncer = Debouncer()

    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=config.resample_interval_ms,
        inference_engine_holder=inference_engine_holder,
        debouncer=debouncer,
        result_publisher=result_publisher,
    )

    command_subscriber = CommandSubscriber(config.command_topic, calibration_manager)
    command_subscriber.attach(mqtt_subscriber.client)

    return Pipeline(
        config=config,
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        inference_engine_holder=inference_engine_holder,
        snapshotter=snapshotter,
        mqtt_subscriber=mqtt_subscriber,
        command_subscriber=command_subscriber,
    )
```

- [ ] **Step 4: `subscriber_cli.py`에 `model_dir` 전달**

`jetson_app/src/jetson_app/subscriber_cli.py`에서:

```python
        pipeline = build_pipeline(
            config=config,
            calibration_dir=args.calibration_dir,
            train_fn=train_fn,
        )
```

를:

```python
        pipeline = build_pipeline(
            config=config,
            calibration_dir=args.calibration_dir,
            model_dir=args.model_dir,
            train_fn=train_fn,
        )
```

로 바꾼다.

- [ ] **Step 5: 테스트 통과 확인 — 신규 + 기존 전부**

Run: `cd jetson_app && uv run pytest tests/test_pipeline.py -v`
Expected: PASS, 9개(기존 6 + 신규 3) 전부.

Run: `cd jetson_app && uv run pytest -q`
Expected: 전체 스위트 통과, 실패 없음.

Run: `cd jetson_app && uv run jetson-app --help`
Expected: exit 0, `--model-dir` 옵션 포함된 도움말 출력(실제 연결 시도 없음).

- [ ] **Step 6: 커밋**

```bash
cd jetson_app
git add src/jetson_app/pipeline.py src/jetson_app/subscriber_cli.py tests/test_pipeline.py
git commit -m "feat: integrate inference engine, debounce, and result publisher into pipeline"
```

---

## Self-Review 체크리스트 (참고용, 실행 시 삭제하지 않음)

- **스펙 커버리지**: 스펙 6절(실시간 이상 점수 계산 — 예측→오차→재정규화→최댓값→디바운스)은 Task 5(InferenceEngine)+Task 4(Debouncer)+Task 7(스케줄러 연결)이 구현한다. 7절의 "모델 파일 손상/로드 실패 → CALIBRATING 폴백"은 Task 8이 구현한다. 상위 문서 3절 발행 스키마는 Task 6(ResultPublisher)이 구현한다. "재시작 시 자동 MONITORING 재개"(사용자 확인 완료)는 Task 3(StateStore)+Task 8(build_pipeline 기동 로직)이 함께 구현한다.
- **자리표시자 없음**: 모든 스텝에 실제 코드가 포함되어 있다.
- **타입 일관성**: `TrainFn`은 기존 `jetson_app.calibration` 정의를 재사용(재정의 없음). `AnomalyResult`의 필드명(`anomaly_score`, `top_deviant_tag`)은 `inference.py`/`scheduler.py`/`publisher.py`/테스트에서 동일하게 사용된다. `ModelArtifact`/`type_indices`/`normalize_continuous_columns`/`compute_raw_errors`/`model_artifact_path`는 이전 계획(모델 레이어)에서 이미 만든 `ModelArtifact`를 그대로 확장하며 필드명을 바꾸지 않는다.
- **파급 효과 처리**: `CalibrationManager.__init__`(Task 3)과 `build_pipeline`(Task 8)의 시그니처 변경이 각 태스크 안에서 전체 테스트 스위트 0 실패로 마무리되도록 각 태스크에 명시했다.
