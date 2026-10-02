# 점수 대상 태그와 입력 전용 태그 분리 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 설정에서 점수 대상 태그, 상태 신호, 입력 전용 태그를 구분해, 알람과 점수가 서보 값만으로 계산되게 한다.

**Architecture:** `EquipmentConfig.tags`를 "모델 입력 태그(점수 태그 + 점수 밖 상태 태그 + 입력 전용 태그)"로 재정의해 학습·MQTT 필터·모델 호환성 검사는 그대로 쓴다. 추론은 그룹의 `tags`에 속한 태그만 z 점수를 내고, 전체 점수를 그룹 점수의 최댓값으로 정한다.

**Tech Stack:** Python 3.8, PyYAML, PyTorch(CPU), pytest 7, uv (프로젝트 루트는 `jetson_app/`).

**Spec:** `docs/superpowers/specs/2026-10-02-scored-and-context-tags-design.md`

## Global Constraints

- Python 3.8 호환: 새 코드는 파일 맨 위에 `from __future__ import annotations`가 이미 있는 모듈에 추가한다(타입 힌트 `tuple[str, ...]` 허용).
- `config.py`는 `torch`를 import하지 않는다.
- 테스트는 `jetson_app/` 폴더에서 `uv run pytest`로 실행한다. 명령은 이 폴더 기준이다.
- 기존 설정(`state_tag`가 `tags`에 포함된 형식, 평면 `tags:` 형식)의 동작과 기존 테스트는 바뀌지 않는다(예외: Task 1에서 명시한 한 개 테스트 교체).
- 커밋 메시지 끝에 `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` 줄을 붙인다.
- 사용자가 만든 파일(`DataList.xlsx`, `Dorco_*.py`, `JSONData/`)은 수정하거나 커밋하지 않는다.
- 사용자의 Mosquitto 서비스(포트 1883)와 `mosquitto_sub` 프로세스는 건드리지 않는다. 데이터 검증은 비공개 포트 18831의 브로커만 쓰고, 내가 띄운 프로세스만 PID로 종료한다.

## Review Focus

1. 점수 밖 `state_tag`가 모델 입력에서 빠지면 상태별 기준을 못 고른다 → Task 1 테스트(입력 태그에 포함), Task 3 파이프라인 테스트(학습·재개).
2. `context_tags` 또는 상태 태그가 점수 태그와 중복되면 한 태그가 점수와 입력 양쪽에서 이중 처리된다 → Task 1 오류 테스트.
3. 여러 그룹이 같은 `state_tag`를 가리키면 입력 태그가 중복되면 안 된다 → Task 1 테스트(한 번만 포함).
4. 입력 전용 태그의 오차가 아무리 커도 점수와 알람이 오르면 안 된다 → Task 2 테스트.
5. 입력 전용 태그 토픽이 오지 않으면(윈도우에 None) 채점이 조용히 멈춘다 → Task 2 테스트(None 반환 확인), Task 4 README에 안내.
6. `context_tags`를 바꾼 뒤 재시작하면 저장된 모델이 호환되지 않는다고 판정돼야 한다 → Task 3 테스트.
7. 기존 형식 설정의 점수가 변하면 안 된다 → 기존 테스트 통과(전체 스위트)로 확인.

---

### Task 1: 설정 — `context_tags`와 점수 밖 `state_tag`

**Files:**
- Modify: `jetson_app/src/jetson_app/config.py` (`EquipmentConfig`, `_parse_groups`, 새 함수 2개, `load_equipment_config`)
- Test: `jetson_app/tests/test_config.py` (한 테스트 교체 + 새 테스트)

**Interfaces:**
- Produces: `EquipmentConfig.context_tags: tuple[str, ...] = ()`. `EquipmentConfig.tags`는 로더가 `점수 태그(그룹 순서) + 점수 밖 상태 태그 + context_tags`(중복 제거, 이 순서)로 채운다. `GroupConfig`와 `group_specs()`는 변경 없음(`tags`는 점수 태그만).
- Consumes: 없음.

- [ ] **Step 1: 기존 테스트 교체와 새 테스트 작성**

`tests/test_config.py`에서 아래 기존 테스트를

```python
def test_state_tag_must_be_one_of_the_group_tags(tmp_path):
    block = 'groups:\n  a:\n    state_tag: "z"\n    tags: ["x", "y"]\n'
    with pytest.raises(ConfigError, match="state_tag"):
        _load_text(tmp_path, _groups_yaml_with(block))
```

다음으로 교체하고(상태 태그가 `tags` 밖일 수 있게 되었으므로), 이어서 새 테스트를 파일 끝에 추가한다.

```python
@pytest.mark.parametrize("bad", ["5", "''"])
def test_state_tag_must_be_a_non_empty_string(tmp_path, bad):
    block = f'groups:\n  a:\n    state_tag: {bad}\n    tags: ["x"]\n'
    with pytest.raises(ConfigError, match="state_tag"):
        _load_text(tmp_path, _groups_yaml_with(block))


def test_state_tag_may_be_outside_the_scored_tags_and_becomes_a_model_input(tmp_path):
    block = 'groups:\n  a:\n    state_tag: "z"\n    tags: ["x", "y"]\n'

    config = _load_text(tmp_path, _groups_yaml_with(block))

    assert config.groups[0].state_tag == "z"
    assert config.groups[0].tags == ("x", "y")  # 점수 대상은 x, y뿐
    assert config.tags == ("x", "y", "z")  # 모델 입력에는 z도 들어간다
    assert config.group_specs() == {"a": ("z", ("x", "y"))}
```

파일 끝에 추가:

```python
def _with_context(block):
    return GROUPS_YAML.replace("groups:\n", block + "groups:\n", 1)


def test_context_tags_are_model_inputs_but_not_scored(tmp_path):
    config = _load_text(tmp_path, _with_context('context_tags: ["C:U0_TaktTime", "C:U4_TaktTime"]\n'))

    assert config.context_tags == ("C:U0_TaktTime", "C:U4_TaktTime")
    assert config.tags[-2:] == ("C:U0_TaktTime", "C:U4_TaktTime")
    scored = {tag for _state, tags in config.group_specs().values() for tag in tags}
    assert not scored & set(config.context_tags)


def test_model_input_order_is_scored_then_state_only_then_context_without_duplicates(tmp_path):
    block = (
        'context_tags: ["c1"]\n'
        "groups:\n"
        '  a:\n    state_tag: "s"\n    tags: ["x", "y"]\n'
        '  b:\n    state_tag: "s"\n    tags: ["w"]\n'  # 같은 상태 태그를 두 그룹이 공유
    )

    config = _load_text(tmp_path, _groups_yaml_with(block))

    assert config.tags == ("x", "y", "w", "s", "c1")  # s는 한 번만


def test_context_tags_default_to_empty(tmp_path):
    assert _load_text(tmp_path, GROUPS_YAML).context_tags == ()


@pytest.mark.parametrize(
    "block, needle",
    [
        ("context_tags: []\n", "context_tags"),
        ('context_tags: "x"\n', "context_tags"),
        ("context_tags: [5]\n", "context_tags"),
        ('context_tags: ["c", "c"]\n', "more than once"),
        ('context_tags: ["A:AxX_Act_Pos"]\n', "already used"),  # 점수 태그와 중복
        ('context_tags: ["P:U0_ProcStart"]\n', "already used"),  # 상태 태그와 중복
    ],
)
def test_invalid_context_tags_are_rejected(tmp_path, block, needle):
    with pytest.raises(ConfigError, match=needle):
        _load_text(tmp_path, _with_context(block))


def test_context_tags_with_flat_tags_config_is_rejected(tmp_path):
    text = _groups_yaml_with('context_tags: ["c"]\ntags: ["x", "y"]\n')

    with pytest.raises(ConfigError, match="context_tags.*groups"):
        _load_text(tmp_path, text)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_config.py -q`
Expected: 새 테스트들이 FAIL (`context_tags` 속성 없음, `state_tag` 검증 메시지 불일치 등). 기존 다른 테스트는 PASS.

- [ ] **Step 3: 구현**

`src/jetson_app/config.py`:

(a) `EquipmentConfig`의 마지막 필드 뒤(`training: TrainingConfig = TrainingConfig()` 다음 줄)에 추가:

```python
    # 모델 입력으로만 쓰고 점수에는 반영하지 않는 태그. `tags`는 점수 태그, 점수 밖 상태 태그,
    # 이 태그를 모두 합한 모델 입력 태그다(학습/MQTT 필터/모델 호환성 검사가 이 값을 쓴다).
    context_tags: tuple[str, ...] = ()
```

(b) `_parse_groups`에서 `state_tag` 검증을 교체한다. 기존:

```python
        state_tag = body.get("state_tag")
        if state_tag is not None and (not isinstance(state_tag, str) or state_tag not in tags):
            raise ConfigError(f"group '{name}': state_tag must be one of the group's tags")
```

교체:

```python
        state_tag = body.get("state_tag")
        if state_tag is not None and (not isinstance(state_tag, str) or not state_tag):
            raise ConfigError(f"group '{name}': state_tag must be a non-empty string")
```

(c) `_parse_alarm` 바로 위에 새 함수 두 개를 추가한다.

```python
def _parse_context_tags(raw: object, groups: tuple[GroupConfig, ...]) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw or not all(isinstance(t, str) and t for t in raw):
        raise ConfigError("context_tags must be a non-empty list of tag names")
    seen: set[str] = set()
    for tag in raw:
        if tag in seen:
            raise ConfigError(f"context_tags: tag '{tag}' is listed more than once")
        seen.add(tag)
    taken = {tag for g in groups for tag in g.tags} | {g.state_tag for g in groups if g.state_tag}
    for tag in raw:
        if tag in taken:
            raise ConfigError(
                f"context_tags: tag '{tag}' is already used by a group (tags or state_tag)"
            )
    return tuple(raw)


def _model_input_tags(groups: tuple[GroupConfig, ...], context_tags: tuple[str, ...]) -> tuple[str, ...]:
    """점수 태그(그룹 순서), 점수 밖 상태 태그, 입력 전용 태그 순서로 중복 없이 합친다."""
    ordered: list[str] = []
    for source in (
        [tag for g in groups for tag in g.tags],
        [g.state_tag for g in groups if g.state_tag],
        list(context_tags),
    ):
        for tag in source:
            if tag not in ordered:
                ordered.append(tag)
    return tuple(ordered)
```

(d) `load_equipment_config`: `if "tags" not in data and "groups" not in data:` 블록 바로 뒤에 추가:

```python
    if "context_tags" in data and "groups" not in data:
        raise ConfigError("context_tags can only be used together with 'groups'")
```

그리고 기존

```python
    if "groups" in data:
        groups = _parse_groups(data["groups"])
        tags = [tag for group in groups for tag in group.tags]
    else:
        groups = ()
        tags = data["tags"]
```

를 다음으로 교체한다.

```python
    if "groups" in data:
        groups = _parse_groups(data["groups"])
        context_tags = (
            _parse_context_tags(data["context_tags"], groups) if "context_tags" in data else ()
        )
        tags = _model_input_tags(groups, context_tags)
    else:
        groups = ()
        context_tags = ()
        tags = data["tags"]
```

(e) 마지막 `return EquipmentConfig(...)`의 인자 `training=training,` 다음에 `context_tags=context_tags,`를 추가한다.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_config.py -q`
Expected: 전부 PASS.

- [ ] **Step 5: 전체 스위트와 커밋**

Run: `uv run pytest -q`
Expected: 전부 PASS (기존 253개 중 한 개는 교체되어 개수가 약간 늘어남). 간헐적으로 실패하는 wall-clock 종단 테스트가 있으면 한 번 더 실행해 확인한다.

```bash
git add jetson_app/src/jetson_app/config.py jetson_app/tests/test_config.py
git commit -m "feat: context_tags and state_tag outside scored tags in config

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 추론 — 점수 대상만 계산, 전체 점수는 그룹 점수의 최댓값

**Files:**
- Modify: `jetson_app/src/jetson_app/inference.py` (`InferenceEngine.__init__`, `score`)
- Test: `jetson_app/tests/test_inference.py` (파일 끝에 추가)

**Interfaces:**
- Consumes: Task 1의 모델 입력 태그 구성(점수 밖 상태 태그와 입력 전용 태그가 모델 `tags`에 들어 있음). `ModelArtifact.groups = {name: (state_tag, scored_tags)}`.
- Produces: `InferenceEngine.score()`가 그룹의 `tags`에 속한 태그만 z 점수를 계산하고, `AnomalyResult.anomaly_score`/`top_deviant_tag`를 그룹 점수 최댓값과 그 그룹의 원인 태그로 채운다. 점수 대상 태그의 오차가 하나도 없으면 `None`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_inference.py` 끝에 추가:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_inference.py -q`
Expected: 새 테스트 중 `test_unscored_...`, `test_state_tag_outside_...`, `test_score_is_none_when_no_scored_tag_has_an_error`가 FAIL(현재는 모든 태그가 전체 점수에 들어감). 나머지 기존 테스트는 PASS.

- [ ] **Step 3: 구현**

`src/jetson_app/inference.py`:

(a) `__init__`에서 `self._state_tag_of = {...}` 블록 바로 뒤에 추가:

```python
        # 점수를 내는 태그(그룹의 tags). 점수 밖 상태 태그와 입력 전용 태그는 예측만 하고 채점하지 않는다.
        self._scored_tags = tuple(tag for _state, tags in self._groups.values() for tag in tags)
```

(b) `score()`에서 `actual_values = dict(zip(self._tags, actual_row))`부터 `return AnomalyResult(...)`까지를 아래로 교체한다.

```python
        actual_values = dict(zip(self._tags, actual_row))
        z_by_tag: dict[str, float] = {}
        for tag in self._scored_tags:
            err_tensor = raw_errors.get(tag)
            if err_tensor is None:
                continue
            error_mean, error_std = self._stats_for(
                tag, actual_values.get(self._state_tag_of.get(tag))
            )
            z_by_tag[tag] = (err_tensor.item() - error_mean) / error_std

        group_results: dict[str, GroupResult] = {}
        for name, (_state_tag, group_tags) in self._groups.items():
            group_best_tag: str | None = None
            group_best_z: float | None = None
            for tag in group_tags:
                z = z_by_tag.get(tag)
                if z is not None and (group_best_z is None or z > group_best_z):
                    group_best_z = z
                    group_best_tag = tag
            if group_best_tag is not None:
                group_results[name] = GroupResult(score=group_best_z, top_tag=group_best_tag)

        if not group_results:
            return None

        # 전체 점수는 점수 대상 그룹의 점수 중 최댓값이다(점수 밖 태그는 영향을 주지 않는다).
        top_group = max(group_results.values(), key=lambda group: group.score)
        return AnomalyResult(
            anomaly_score=top_group.score,
            top_deviant_tag=top_group.top_tag,
            group_results=group_results,
        )
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_inference.py -q`
Expected: 전부 PASS (기존 `test_overall_score_is_the_maximum_group_score`, `test_artifact_without_groups_scores_as_a_single_all_group` 포함).

- [ ] **Step 5: 전체 스위트와 커밋**

Run: `uv run pytest -q`
Expected: 전부 PASS.

```bash
git add jetson_app/src/jetson_app/inference.py jetson_app/tests/test_inference.py
git commit -m "feat: score only group tags; overall score is the max group score

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: 파이프라인 회귀 테스트 (학습·재개·호환성)

**Files:**
- Test: `jetson_app/tests/test_pipeline.py` (파일 끝에 추가). 구현 코드 변경은 없다(학습과 파이프라인은 `config.tags`를 이미 모델 입력 태그로 쓴다). 실패하면 원인을 찾아 해당 모듈을 고친다.

**Interfaces:**
- Consumes: Task 1의 `EquipmentConfig.context_tags`와 점수 밖 `state_tag`, Task 2의 추론. 파일 안의 기존 헬퍼 `_e2e_config`, `_feed`, `_feed_and_train`, `_send_train`, `_train_fn_for`와 `GroupConfig`(이미 import됨).
- Produces: 없음.

- [ ] **Step 1: 테스트 작성**

`tests/test_pipeline.py` 끝에 추가:

```python
# ---- 점수 대상 / 입력 전용 태그 ----


def _context_config(equipment_id, window_size=3, min_samples=10):
    # _send는 첫 태그에 float(i), 나머지에 i % 2를 넣는다.
    # a: 점수 대상, s: 점수 밖 상태 태그, c: 입력 전용 태그.
    base = _e2e_config(
        equipment_id, tags=("a", "s", "c"), window_size=window_size, min_samples=min_samples
    )
    return replace(
        base,
        groups=(GroupConfig(name="proc", state_tag="s", tags=("a",)),),
        context_tags=("c",),
    )


def test_context_pipeline_trains_on_all_inputs_and_scores_only_the_scored_tag(tmp_path):
    config = _context_config("e2e_ctx_publish")
    model_dir = tmp_path / "model_data"
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=_train_fn_for(config, model_dir),
    )
    _feed(pipeline, config.tags, 0, 20)
    _send_train(pipeline)
    assert pipeline.calibration_manager.state == CalibrationState.MONITORING

    artifact = load_artifact(model_dir / f"{config.equipment_id}.pt")
    assert artifact.tags == ("a", "s", "c")
    assert artifact.groups == config.group_specs() == {"proc": ("s", ("a",))}
    assert set(artifact.regime_error_stats) == {"a"}

    published = []

    def _fake_publish(topic, payload):
        published.append((topic, payload))
        return SimpleNamespace(rc=0)

    pipeline.mqtt_subscriber.client.publish = _fake_publish
    _feed(pipeline, config.tags, 20, 30)

    assert published, "MONITORING 진입 후에도 이상 점수가 발행되지 않았다"
    record = json.loads(published[0][1])["records"][0]
    assert record["jetson:top_deviant_tag"] == "a"


def test_context_pipeline_resumes_monitoring_after_restart(tmp_path):
    config = _context_config("e2e_ctx_resume")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)
    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed_and_train(first, config.tags)
    assert first.calibration_manager.state == CalibrationState.MONITORING

    second = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )

    assert second.calibration_manager.state == CalibrationState.MONITORING
    assert second.inference_engine_holder.get() is not None


def test_pipeline_falls_back_to_calibrating_when_context_tags_changed(tmp_path):
    config = _context_config("e2e_ctx_changed")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)
    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed_and_train(first, config.tags)
    assert first.calibration_manager.state == CalibrationState.MONITORING

    # "재시작": 입력 전용 태그 c가 c2로 바뀐 설정(모델 입력 태그가 달라진다)
    changed = replace(config, tags=("a", "s", "c2"), context_tags=("c2",))
    second = build_pipeline(
        config=changed,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )

    assert second.calibration_manager.state == CalibrationState.CALIBRATING
    assert second.inference_engine_holder.get() is None
```

- [ ] **Step 2: 실행**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: 전부 PASS (구현은 Task 1·2에서 끝났다). 실패하면 실패 메시지로 원인을 확인한다. 예를 들어 `artifact.groups == config.group_specs()`가 실패하면 저장·로드 과정에서 튜플/리스트 형태가 달라진 것이므로 기존 `test_grouped_pipeline_resumes_monitoring_after_restart`와 같은 비교 방식인지 확인하고, 모델 코드가 아니라 비교 방식을 맞춘다.

- [ ] **Step 3: 전체 스위트와 커밋**

Run: `uv run pytest -q`
Expected: 전부 PASS.

```bash
git add jetson_app/tests/test_pipeline.py
git commit -m "test: pipeline coverage for context tags and out-of-group state tags

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: NX5 서보 설정과 README

**Files:**
- Create: `jetson_app/configs/nx5_servo.yaml`
- Modify: `jetson_app/README.md` ("그룹과 상태별 기준" 절 끝, `### 새 설비를 붙이는 순서` 바로 앞에 새 절 삽입)
- Test: `jetson_app/tests/test_config.py` (파일 끝에 추가)

**Interfaces:**
- Consumes: Task 1의 설정 형식.
- Produces: `configs/nx5_servo.yaml`(Task 5의 데이터 검증이 사용).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_config.py` 끝에 추가:

```python
def test_nx5_servo_config_scores_only_the_servo_axes():
    from pathlib import Path

    path = Path(__file__).parent.parent / "configs" / "nx5_servo.yaml"
    config = load_equipment_config(path)

    assert [g.name for g in config.groups] == ["axis_x", "axis_z", "axis_cv"]
    assert {g.name: g.state_tag for g in config.groups} == {
        "axis_x": "NX5_ProcStart:U0_ProcStart",
        "axis_z": "NX5_ProcStart:U4_ProcStart",
        "axis_cv": "NX5_ProcStart:U9_ProcStart",
    }
    scored = [tag for g in config.groups for tag in g.tags]
    assert len(scored) == 9
    assert all(tag.startswith("NX5_AxisData:Ax") for tag in scored)
    assert config.context_tags == tuple(f"NX5_TaktTime:U{n}_TaktTime" for n in range(11))
    assert len(config.tags) == 9 + 3 + 11
    assert set(config.subscribe_topics) == {"dx1/AxisData", "dx1/ProcStart", "dx1/TaktTime"}
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_config.py::test_nx5_servo_config_scores_only_the_servo_axes -q`
Expected: FAIL (파일이 없음, `ConfigError`).

- [ ] **Step 3: 설정 파일 작성**

`configs/nx5_servo.yaml`:

```yaml
# NX5 데모 라인: 서보 3축(AxX, AxZ, AxCV)의 Pos/Vel/Trq만 이상 판단에 쓴다.
# ProcStart는 어느 공정이 도는지 알려 주는 조건 신호로만 쓰고 점수에는 반영하지 않는다.
# TaktTime(U0~U10)은 모델 입력으로만 쓰고 점수에는 반영하지 않는다.
equipment_id: nx5
mqtt:
  subscribe_topics:
  - dx1/AxisData
  - dx1/ProcStart
  - dx1/TaktTime
  publish_topic: jetson/nx5/anomaly
  command_topic: jetson/nx5/cmd
context_tags:
- NX5_TaktTime:U0_TaktTime
- NX5_TaktTime:U1_TaktTime
- NX5_TaktTime:U2_TaktTime
- NX5_TaktTime:U3_TaktTime
- NX5_TaktTime:U4_TaktTime
- NX5_TaktTime:U5_TaktTime
- NX5_TaktTime:U6_TaktTime
- NX5_TaktTime:U7_TaktTime
- NX5_TaktTime:U8_TaktTime
- NX5_TaktTime:U9_TaktTime
- NX5_TaktTime:U10_TaktTime
groups:
  axis_x:
    state_tag: NX5_ProcStart:U0_ProcStart
    tags:
    - NX5_AxisData:AxX_Act_Pos
    - NX5_AxisData:AxX_Act_Vel
    - NX5_AxisData:AxX_Act_Trq
  axis_z:
    state_tag: NX5_ProcStart:U4_ProcStart
    tags:
    - NX5_AxisData:AxZ_Act_Pos
    - NX5_AxisData:AxZ_Act_Vel
    - NX5_AxisData:AxZ_Act_Trq
  axis_cv:
    state_tag: NX5_ProcStart:U9_ProcStart
    tags:
    - NX5_AxisData:AxCV_Act_Pos
    - NX5_AxisData:AxCV_Act_Vel
    - NX5_AxisData:AxCV_Act_Trq
resample_interval_ms: 100
max_lateness_ms: 500
window_size: 30
calibration:
  max_duration: 7d
  min_samples: 3000
training:
  max_samples: 45000
  epochs: 20
```

- [ ] **Step 4: README 절 추가**

`README.md`에서 `### 새 설비를 붙이는 순서` 줄 바로 앞(그 앞의 `training.max_samples` 글머리표 다음 빈 줄 뒤)에 아래를 삽입한다.

````markdown
### 점수 대상 태그와 입력 전용 태그

모델은 설정에 적힌 **모든 태그**를 입력으로 받아 다음 스텝을 예측하지만, 이상 점수와 알람은 **그룹의 `tags`에 적은 태그만**으로 계산한다. 점수에서 빼고 싶은 신호는 아래 두 가지 방법으로 입력에만 쓸 수 있다.

```yaml
context_tags: ["NX5_TaktTime:U0_TaktTime", "NX5_TaktTime:U4_TaktTime"]   # 입력만, 점수 제외 (groups와 함께만 사용)
groups:
  axis_z:
    state_tag: "NX5_ProcStart:U4_ProcStart"   # tags 밖에 두면 입력 전용 조건 신호가 된다
    tags: ["NX5_AxisData:AxZ_Act_Pos", "NX5_AxisData:AxZ_Act_Vel", "NX5_AxisData:AxZ_Act_Trq"]
```

- `state_tag`는 그 그룹의 정상 기준을 동작 중/대기 중으로 나누는 조건이다. `tags` 안에 두면 이전처럼 점수에도 반영되고, 밖에 두면 점수에는 반영되지 않는다.
- 전체 점수는 그룹 점수의 최댓값이다. 입력 전용 태그의 오차는 점수나 알람에 영향을 주지 않는다.
- 켜짐/꺼짐 신호나 계단형 값(`ProcStart`, `TaktTime`, 센서)은 값이 바뀌는 순간의 시각이 조금씩 달라서 예측 오차가 크게 나오고 오탐의 원인이 된다. 이런 신호는 점수 대상이 아니라 `state_tag`나 `context_tags`로 쓰는 것을 권한다.
- **입력 전용 태그의 토픽이 오지 않으면 채점이 멈춘다.** 모델은 모든 입력 태그가 한 번씩 관측돼야 채점을 시작하므로, `context_tags`에 넣은 태그의 토픽이 구독돼 있고 발행되고 있는지 확인한다.
- `context_tags`나 `state_tag`를 바꾸면 모델 입력 태그가 달라져 저장된 모델과 맞지 않으므로, 재시작 시 `CALIBRATING`으로 폴백한다. `recalibrate`로 다시 학습한다.
- 예: `configs/nx5_servo.yaml`은 서보 3축 9개 값만 점수로 쓰고, `ProcStart` 3개를 조건으로, `TaktTime` 11개를 입력 전용으로 쓴다.

````

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest tests/test_config.py -q`
Expected: 전부 PASS.

- [ ] **Step 6: 전체 스위트와 커밋**

Run: `uv run pytest -q`
Expected: 전부 PASS.

```bash
git add jetson_app/configs/nx5_servo.yaml jetson_app/README.md jetson_app/tests/test_config.py
git commit -m "feat: nx5_servo config and docs for scored vs context tags

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `run1` 데이터 검증 (비교 실험, 최종 학습)

코드 변경은 없고, 임시 스크립트(`scratchpad/rehearsal/`)로 로컬 비공개 브로커(포트 18831)에서 실행한 뒤 결과를 스펙에 기록한다. 저장소 파일은 스펙 문서 한 개만 바꾼다.

**Files:**
- Modify (scratch): `<scratchpad>/rehearsal/replay.py` (결과 파일 경로를 환경 변수로 지정), `summarize.py`, `sweep.py` (축 그룹 이름 반영)
- Modify: `docs/superpowers/specs/2026-10-02-scored-and-context-tags-design.md` (끝에 "9. 검증 결과" 절 추가)

`<scratchpad>` = `C:\Users\hyunkeun.kim\AppData\Local\Temp\claude\C--WORK-10--Jetson\23705cd7-b1c4-4b7d-837c-1069f0080571\scratchpad`

**Interfaces:**
- Consumes: Task 4의 `configs/nx5_servo.yaml`, 이전 모델 검증 결과(`rehearsal/monitor_results.json`, 임계값 3/연속 3스텝에서 전체 224건, 서보 그룹 96건).

- [ ] **Step 1: 이전 결과 보존과 스크립트 수정**

```bash
cp "<scratchpad>/rehearsal/monitor_results.json" "<scratchpad>/rehearsal/monitor_results_prev.json"
```

`replay.py`의 `OUT = ...` 줄 바로 아래에 추가:

```python
OUT = os.environ.get("REH_OUT", OUT)
```

`sweep.py`에서 `servo = ["process_0", "process_4", "process_9"]` 줄을 `servo = groups`로 바꾼다(새 설정의 그룹이 모두 서보 축이다). `summarize.py`는 그룹 이름을 데이터에서 읽으므로 그대로 쓴다.

- [ ] **Step 2: 비교 실험 — 앞 두 구간으로 학습**

1. 비공개 브로커를 띄운다: `mosquitto.exe -c <scratchpad>/rehearsal/mosq.conf`를 백그라운드로 실행하고 PID를 `mosq.pid`에 저장한다. 1883 포트의 서비스 PID(6640)가 그대로인지 확인한다.
2. 앱을 띄운다(백그라운드):
   `uv run jetson-app --config configs/nx5_servo.yaml --host 127.0.0.1 --port 18831 --calibration-dir <scratchpad>/rehearsal/servo_cmp/calibration_data --model-dir <scratchpad>/rehearsal/servo_cmp/model_data`
   앱 로그에 `구독 시작`이 보일 때까지 기다린다.
3. 학습 구간을 재생하고 학습한다(S1+S2, 약 19,350스텝):
   `REH_DIR=<scratchpad>/rehearsal/servo_cmp uv run python <scratchpad>/rehearsal/replay.py train`
   출력의 `학습 완료: N초`를 기록한다. 로그에 `상태별 기준 표본 부족` 줄이 있으면 어떤 태그인지 기록한다.

- [ ] **Step 3: 비교 실험 — 검증 구간 채점**

1. 앱을 종료(내가 띄운 PID만)하고 같은 `--model-dir`로 다시 띄운다(저장된 모델로 `MONITORING` 재개).
2. `REH_OUT=<scratchpad>/rehearsal/monitor_results.json REH_DIR=<scratchpad>/rehearsal/servo_cmp uv run python <scratchpad>/rehearsal/replay.py monitor`
3. `summarize.py`와 `sweep.py`를 실행한다. 기록할 값:
   - 임계값 3/연속 3스텝에서 알람 에피소드 수(그룹별과 합계). 이전 모델은 합계 224, 서보 그룹 96.
   - `ProcStart`나 `TaktTime`이 `top_tag`로 나온 횟수(0이어야 한다).
   - 임계값 5, 8, 10에서의 에피소드 수.
4. 이전 결과와 나란히 비교한다(`monitor_results_prev.json`은 그룹 구성이 달라 점수 분포만 참고한다).

- [ ] **Step 4: (선택) `TaktTime` 영향 비교**

`configs/nx5_servo.yaml`을 읽어 `context_tags`를 지운 설정을 `<scratchpad>/rehearsal/nx5_servo_noctx.yaml`로 만든다(`topics`에서 `dx1/TaktTime`도 지운다). 다른 `--calibration-dir`/`--model-dir`(`servo_cmp_noctx`)로 Step 2·3을 반복해 에피소드 수와 학습 시간을 비교한다.

- [ ] **Step 5: 최종 학습 — 전체 정상 데이터**

Error로 적힌 두 구간(11:32:57~11:36:59, 11:53:11~11:56:16, 몇 초 여유)만 뺀 전체 약 39,000스텝으로 학습한다:
`REH_DIR=<scratchpad>/rehearsal/full_servo uv run python <scratchpad>/rehearsal/replay.py trainall`
(앱은 `--calibration-dir`/`--model-dir`를 `full_servo/` 아래로 새로 띄운다.) 학습 시간을 기록한다. 모델은 임시 폴더에 둔다.

- [ ] **Step 6: 정리**

내가 띄운 앱과 비공개 브로커를 PID로 종료하고, 1883 서비스(PID 6640)와 `Get-Service mosquitto`가 `Running`인지 확인한다.

- [ ] **Step 7: 결과 기록과 커밋**

스펙 끝에 "## 9. 검증 결과" 절을 추가한다. 포함할 내용: 비교 실험 표(이전 모델 vs 새 모델: 합계/서보 그룹 에피소드 수, 임계값별), `ProcStart`/`TaktTime`의 `top_tag` 횟수, 학습 시간(앞 두 구간, 전체), `TaktTime` 유무 비교(수행한 경우), 남은 문제와 다음 단계. 숫자는 실제로 측정한 값만 쓴다.

```bash
git add docs/superpowers/specs/2026-10-02-scored-and-context-tags-design.md
git commit -m "docs: record run1 validation results for scored/context tags

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

그리고 메모(`project_run1_data.md`)에 새 설정의 결과를 한 줄 갱신한다.
