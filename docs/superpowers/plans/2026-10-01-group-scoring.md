# 그룹별 점수와 동작/대기 상태별 기준 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (인라인). Steps use checkbox (`- [ ]`) syntax. **이 계획서는 간결한 형식이다**: Task마다 파일, 인터페이스, 테스트 목록, 핵심 규칙만 적고 코드는 TDD로 작성한다. 동작의 세부는 스펙이 기준이다.

**Goal:** 태그를 사용자가 정한 그룹으로 묶어 그룹별 점수/알람을 발행하고, 그룹의 상태 태그(동작 중 신호)로 동작/대기 상태별 정상 오차 기준을 써서 대기 중 서보 충격까지 감지한다. 사용자가 적은 목록(엑셀)에서 설정 초안을 만드는 도구를 제공한다.

**Architecture:** 설정에 `groups:`(그룹 → `state_tag` + 태그 목록)를 추가하고, 학습이 (태그, 상태) 단위의 정상 오차 통계를 artifact에 저장한다. `InferenceEngine`이 상태에 맞는 기준으로 z-score를 구해 그룹 점수를 내고, `SnapshotProcessor`가 그룹마다 디바운서를 두어 알람을 확정, `ResultPublisher`가 그룹 필드를 추가해 발행한다. 설정 생성 도구는 엑셀을 읽어 YAML 초안을 만들고, 샘플 JSON으로 태그를 검증한다.

**Tech Stack:** 기존 `jetson_app`(Python 3.8, uv, torch<2.4, paho-mqtt<2, pytest 7). 신규 `openpyxl`(도구용, 선택 의존성 + dev 그룹).

**Spec:** [`docs/superpowers/specs/2026-10-01-process-aware-scoring-design.md`](../specs/2026-10-01-process-aware-scoring-design.md)

## Global Constraints

- Python 3.8 호환, 새/수정 파일 최상단 `from __future__ import annotations`(런타임 평가 위치에 `X | None`, `list[...]` 금지).
- 평면 `tags:` 설정은 단일 그룹 `all`(상태 태그 없음)로 취급하고 **발행 스키마를 바꾸지 않는다**(그룹이 둘 이상일 때만 그룹 필드 추가).
- `training.py`는 torch를 import하므로 `config.py`/도구는 `training.py`를 import하지 않는다. 기본값 상수는 `config.py`에 두고 `training.py`가 가져다 쓴다.
- 기존 테스트 의도를 유지한다(모두 통과해야 함).
- 커밋 메시지 끝에 `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`(두 번째 `-m`). 명령은 `C:\WORK\10. Jetson\jetson_app`에서 `uv run pytest ...`.

## Review Focus

1. **상태 태그가 학습 내내 한 번도 켜지지/꺼지지 않음** → 그 상태의 기준은 전체 통계로 대체되고 로그를 남긴다. (Task 2)
2. **실행 중 `state_tag` 값이 `None`** → 그 스텝은 전체 통계를 쓴다(예외 없음). (Task 3)
3. **같은 태그가 두 그룹에 있거나 `state_tag`가 그룹에 없음** → 기동 시 `ConfigError`. (Task 1)
4. **그룹별 알람이 서로 독립**(한 그룹의 연속 초과가 다른 그룹 카운터에 영향 없음), 리셋 시 모두 초기화. (Task 4)
5. **config의 그룹 구성이 학습 후 바뀜** → `CALIBRATING` 폴백. (Task 5)
6. **엑셀에 샘플에 없는 태그/오탈자(`[`)/빈 행** → 도구가 오류 또는 경고로 알려주고, 목록에 없는 태그를 설정에 넣지 않는다. (Task 6)

## File Structure

| 파일 | 작업 | 책임 |
|---|---|---|
| `src/jetson_app/config.py` | 수정 | `groups:` 파싱, `GroupConfig`/`AlarmConfig`/`TrainingConfig`, `resolved_groups()` |
| `src/jetson_app/training.py` | 수정 | 상태별 오차 통계, artifact 필드, 설정 가능한 학습 상한/에폭 |
| `src/jetson_app/inference.py` | 수정 | 상태별 기준 선택, `GroupResult`, 그룹 점수 |
| `src/jetson_app/snapshot_processor.py` | 수정 | 그룹별 디바운서와 발행 |
| `src/jetson_app/publisher.py` | 수정 | 그룹 필드 발행 |
| `src/jetson_app/pipeline.py`, `subscriber_cli.py` | 수정 | 연결, 호환성 검사, 학습 설정 전달 |
| `src/jetson_app/datalist_to_config.py` | 신규 | 엑셀 → 설정 초안 도구(+ `jetson-datalist` 진입점) |
| `pyproject.toml`, `uv.lock` | 수정 | `openpyxl` 선택 의존성/dev 그룹, 진입점 |
| `configs/nx5.yaml`, `README.md`, 스펙 | 신규/수정 | NX5 설정, 문서 |

---

### Task 1: 설정의 그룹/알람/학습 항목

**Files:** Modify `src/jetson_app/config.py`, `src/training.py`(상수 import만); Test `tests/test_config.py`

**Interfaces — Produces:**
- `GroupConfig(name: str, state_tag: str | None, tags: tuple[str, ...])` (frozen dataclass)
- `AlarmConfig(threshold: float = 3.0, confirm_steps: int = 3)`, `TrainingConfig(max_samples: int = 20_000, epochs: int = 20)` (frozen dataclass, 기본값 상수는 `config.py`에 `DEFAULT_*`로 정의, 알람 기본값은 `debounce.DEFAULT_*`를 가져옴)
- `EquipmentConfig`에 기본값이 있는 필드 `groups: tuple[GroupConfig, ...] = ()`, `alarm: AlarmConfig = AlarmConfig()`, `training: TrainingConfig = TrainingConfig()` 추가, 메서드 `resolved_groups() -> tuple[GroupConfig, ...]` (`groups`가 있으면 그대로, 없으면 `(GroupConfig("all", None, self.tags),)`)
- `load_equipment_config`: `groups:` 매핑을 읽어 `tags`를 그룹 선언 순서로 이어 붙여 만든다.

**Tests (먼저 작성, 실패 확인):**
- `groups:` 파싱: 그룹 이름/`state_tag`/태그 순서, 전체 `tags`가 그룹 순서로 이어 붙는지.
- `tags:`와 `groups:` 동시 사용 → `ConfigError`; 둘 다 없음 → `ConfigError`; `groups`가 비었거나 매핑이 아님 → `ConfigError`.
- 같은 태그가 두 그룹에 있음 → `ConfigError`; `state_tag`가 그룹 `tags`에 없음 → `ConfigError`; 그룹의 `tags`가 비었음 → `ConfigError`.
- 평면 `tags:` → `groups == ()`, `resolved_groups()`가 단일 `all` 그룹.
- `alarm`/`training` 기본값, 지정값, 잘못된 값(0, 음수, 문자열, bool)이 `ConfigError`.
- 기존 설정 로드 테스트와 `test_dx1.example.yaml` 로드가 그대로 통과.

**Steps:**
- [ ] 위 테스트를 `tests/test_config.py`에 추가하고 실행해 실패(`AttributeError`/`ConfigError` 미발생)를 확인한다.
- [ ] `config.py` 구현 후 `uv run pytest tests/test_config.py -q` 통과, `training.py`는 `DEFAULT_EPOCHS`/`DEFAULT_MAX_TRAINING_SAMPLES`를 `config`에서 import하도록 바꾸고(상수 정의 제거) 전체 `uv run pytest -q` 통과를 확인한다.
- [ ] 커밋: `git add jetson_app/src/jetson_app/config.py jetson_app/src/jetson_app/training.py jetson_app/tests/test_config.py && git commit -m "feat: add groups, alarm and training sections to config" -m "Co-Authored-By: ..."`

---

### Task 2: 상태별 정상 오차 기준 (학습, artifact)

**Files:** Modify `src/jetson_app/training.py`; Test `tests/test_training.py`

**Interfaces — Produces:**
- 상수 `MIN_REGIME_SAMPLES = 100`.
- `ModelArtifact`에 기본값이 있는 필드 `groups: dict = field(default_factory=dict)`, `regime_error_stats: dict = field(default_factory=dict)` 추가(맨 끝). `groups: dict[str, tuple[str | None, tuple[str, ...]]]`, `regime_error_stats: dict[str, dict[str, tuple[float, float, int]]]` (태그 → {`"on"`/`"off"`: (평균, 표준편차, 표본 수)}).
- `compute_regime_error_stats(errors: dict[str, torch.Tensor], state_values: dict[str, torch.Tensor], groups: dict[str, tuple[str | None, tuple[str, ...]]], min_samples: int = MIN_REGIME_SAMPLES) -> tuple[dict[str, dict[str, tuple[float, float, int]]], list[str]]` — 순수 함수. 그룹에 `state_tag`가 있으면 그 그룹의 태그(자기 자신 제외)마다 `state_values[state_tag] >= 0.5`로 on/off를 나눠 평균/표준편차(`_floor_std` 적용)/표본 수를 구한다. 표본이 `min_samples`보다 적은 상태는 결과에서 빼고 두 번째 반환값(대체된 `"태그(on|off:표본수)"` 문자열 목록)에 넣는다.
- `train_model(..., groups: dict | None = None, ...)`: 정규화 **전** 원본 `y`에서 상태 태그 값을 떼어 두었다가 `compute_regime_error_stats`에 넘기고, artifact에 `groups`(None이면 `{}`)와 `regime_error_stats`를 담는다. 대체 목록이 비어 있지 않으면 한 줄로 로그를 남긴다.
- `make_train_fn(..., groups: dict | None = None)`가 `train_model`로 전달. `save_artifact`/`load_artifact`가 두 필드를 보존(없으면 `{}`).

**Tests:**
- `compute_regime_error_stats`: on/off가 분리되어 평균이 다르게 나오는지(on 오차 큼/off 오차 작음), 표본 수 기록, 표본 부족 상태는 결과에서 빠지고 대체 목록에 이름이 들어가는지, `state_tag` 자신과 상태 태그 없는 그룹은 결과에 없는지.
- `train_model`(작은 합성 데이터): 상태 태그 `s`가 0/1로 번갈아 변하는 샘플에서 artifact에 `groups`, `regime_error_stats`가 채워지고 판정이 정규화 전 값으로 이뤄지는지(상태 태그가 연속값으로 분류돼 정규화되는 경우에도 on/off 표본 수 합이 윈도우 수와 같다).
- 상태 태그가 한 번도 켜지지 않는 데이터 → on 통계 없음 + 로그 출력(`capsys`).
- 저장/로드 왕복에서 두 필드 보존, 필드가 없는 기존 파일은 `{}`.
- `make_train_fn`이 `groups`를 전달해 artifact에 반영.

**Steps:**
- [ ] 테스트 작성 → 실패 확인 → 구현 → 해당 파일 통과 → 전체 통과 → 커밋(`feat: learn per-state error baselines for grouped tags`).

---

### Task 3: 상태별 기준으로 그룹 점수 계산

**Files:** Modify `src/jetson_app/inference.py`; Test `tests/test_inference.py`

**Interfaces — Consumes:** Task 2의 `ModelArtifact.groups`/`regime_error_stats`.
**Produces:**
- `GroupResult(score: float, top_tag: str)` (frozen dataclass)
- `AnomalyResult(anomaly_score, top_deviant_tag, group_results: dict[str, GroupResult] = field(default_factory=dict))`
- `InferenceEngine`: `artifact.groups`가 비어 있으면 모든 태그를 한 그룹 `all`로 취급. 태그별 기준 선택: 그 태그가 `state_tag`가 있는 그룹에 속하고 `actual.values[state_tag]`가 `None`이 아니면 `"on"`(>=0.5)/`"off"`로 `regime_error_stats[tag][상태]`를 쓰고, 없으면 `error_stats[tag]`. 그룹 점수 = 그룹 태그 z의 최댓값(`top_tag` 포함), 전체 점수/`top_deviant_tag` = 전체 최댓값. `group_results`는 항상 모든 그룹을 담는다.

**Tests (결정적, 학습 없이 artifact를 직접 구성):**
- 같은 원본 오차가 `off` 상태(좁은 기준)에서는 높은 점수, `on` 상태(넓은 기준)에서는 낮은 점수를 내는지(모델 출력을 고정하기 위해 가중치를 0으로 만든 GRU나 `compute_raw_errors` 패치로 오차를 주입).
- `state_tag` 값이 `None`이면 `error_stats`로 계산하고 예외가 없는지.
- 그룹 점수/`top_tag`, 전체 점수가 그룹 점수의 최댓값과 일치하는지, 그룹 하나(`all`)일 때 기존 결과와 동일한지.
- 상태 통계가 없는 상태(표본 부족으로 대체됨)는 `error_stats`를 쓰는지.
- 기존 `test_inference.py` 테스트 통과.

**Steps:**
- [ ] 테스트 작성 → 실패 확인 → 구현 → 통과 → 전체 통과 → 커밋(`feat: score tags per group against state-specific baselines`).

---

### Task 4: 그룹별 알람과 발행

**Files:** Modify `src/jetson_app/publisher.py`, `src/jetson_app/snapshot_processor.py`; Test `tests/test_publisher.py`, `tests/test_snapshot_processor.py`

**Interfaces — Consumes:** `AnomalyResult.group_results`(Task 3), `Debouncer`.
**Produces:**
- `GroupOutput(score: float, alarm: bool, top_tag: str)` (frozen dataclass, `publisher.py`)
- `ResultPublisher.publish(timestamp, anomaly_score, alarm, top_deviant_tag, groups: dict[str, GroupOutput] | None = None)` — `groups`가 둘 이상일 때만 각 그룹의 `jetson:<그룹>:score`, `:alarm`, `:top_tag`를 추가(기존 3개 필드는 항상).
- `SnapshotProcessor(..., debouncers: dict[str, Debouncer] | None = None, ...)` — 단일 `debouncer` 인자를 **대체**한다. 점수가 나오면 그룹마다 `debouncers[이름].update(score)`로 그룹 알람을 구하고, 전체 알람 = 그룹 알람 중 하나라도 참. `reset_window`면 모든 디바운서를 리셋. `debouncers`가 없으면 채점을 건너뛴다(기존의 "협력자 하나라도 없으면 건너뜀" 규칙 유지).

**Tests:**
- 발행: 그룹이 하나일 때 기존 3개 필드만, 둘 이상일 때 그룹 필드가 추가되는지(키 이름 포함).
- 처리기: 그룹별 디바운서가 독립(그룹 A만 연속 초과 → A만 알람, 전체 알람 참), 그룹 점수가 `GroupOutput`으로 발행되는지, 리셋 시 모든 디바운서가 리셋되는지, 기존 처리기 테스트(채점 건너뛰는 조건들, 이벤트 시각 발행, 큐/스레드)가 `debouncers=` 형태로 통과.

**Steps:**
- [ ] 테스트 수정/작성 → 실패 확인 → 구현 → 통과 → 전체 통과(파이프라인 테스트가 아직 `debouncer=`를 쓰면 Task 5에서 맞춘다 — 이 Task의 커밋 시점에는 `pipeline.py`의 생성 호출만 `debouncers={"all": debouncer}`로 임시 연결해 전체가 통과하게 한다) → 커밋(`feat: per-group alarms and group fields in published results`).

---

### Task 5: 파이프라인 연결과 호환성 검사

**Files:** Modify `src/jetson_app/pipeline.py`, `src/jetson_app/subscriber_cli.py`; Test `tests/test_pipeline.py`

**Interfaces — Consumes:** Task 1~4.
**Behavior:**
- `build_pipeline`이 `config.resolved_groups()`로 `groups_dict = {이름: (state_tag, tags)}`를 만들고, 그룹마다 `Debouncer(config.alarm.threshold, config.alarm.confirm_steps)`를 만들어 `debouncers`로 전달. `wrapped_train_fn`은 학습 후 모든 디바운서를 리셋.
- `_check_artifact_matches_config`에 `artifact.groups != groups_dict`를 추가(불일치 또는 없음 → `ValueError` → `CALIBRATING` 폴백, 기존 메시지 형식에 그룹 항목 추가).
- CLI의 `make_train_fn` 호출에 `groups=...`, `epochs=config.training.epochs`, `max_training_samples=config.training.max_samples` 전달(`groups`는 `{이름: (state_tag, tags)}`, 평면 설정이면 `{"all": (None, tags)}`).

**Tests:**
- 기존 파이프라인 테스트(평면 설정) 전부 통과(artifact `groups`가 `{"all": (None, tags)}`로 저장되므로 폴백되지 않는지 포함).
- 그룹 설정으로 학습한 뒤 같은 설정으로 재시작하면 `MONITORING` 재개, **그룹 구성이 바뀌면 `CALIBRATING` 폴백**(그룹 이름 변경, 상태 태그 변경).
- 그룹 설정 end-to-end: 상태 태그를 포함한 두 그룹으로 학습 후 점수 발행 시 `jetson:<그룹>:score` 등 그룹 필드가 발행되는지(가짜 `client.publish` 사용).
- `train_fn`이 `groups`/`epochs`/`max_samples`를 받는지(`make_train_fn` 경유 artifact 확인).

**Steps:**
- [ ] 테스트 작성/수정 → 실패 확인 → 구현 → 통과 → 전체 통과 → 커밋(`feat: wire group scoring into pipeline and CLI`).

---

### Task 6: 설정 생성 도구

**Files:** Create `src/jetson_app/datalist_to_config.py`; Modify `pyproject.toml`, `uv.lock`; Test `tests/test_datalist_to_config.py`

**Interfaces — Produces:**
- `build_config_yaml(rows, options) -> str`와 `main(argv: list[str] | None = None) -> int`(CLI, 진입점 `jetson-datalist`). `openpyxl`은 엑셀을 읽는 함수 안에서 지연 import하고, 없으면 `uv sync --extra tools` 안내 메시지와 함께 종료 코드 2.
- 인자: `datalist`(엑셀 경로), `--equipment-id`(필수), `--collector-prefix`, `--output`, `--samples`(파일/폴더, 선택), `--allow-missing`, `--group-regex`, `--group-template`, `--state-pattern`, `--resample-interval-ms`, `--max-lateness-ms`, `--window-size`, `--min-samples`.
- 엑셀 열(헤더 대소문자 무시): `Variables`, `MQTT Topic` 필수 / `Description`, `Tag`, `Group`, `StateTag` 선택. `Variables`가 빈 행은 건너뛴다.
- 태그 이름: `Tag` 열이 있으면 그대로, 없으면 `<prefix>_<토픽 마지막 마디>:<변수명의 '.'→'_'>`(변수명 끝 `[`/공백 제거, 경고). 그룹: `Group` 열 → 없으면 `--group-regex`(Description에서 첫 그룹 캡처)와 `--group-template` → 없으면 `general`. 상태 태그: `StateTag` 열의 `Y` → 없으면 `--state-pattern`(`{}`에 캡처 값)이 변수명과 같은 행. 그룹은 첫 등장 순서, 태그는 행 순서.
- 샘플 검증/권장: 샘플 줄(`{"records":[…]}`)의 키 집합과 `timestamp` 간격의 중앙값(10ms 단위 반올림)을 구해, 목록에 있는데 샘플에 없는 태그 → 이름 출력 후 종료 코드 1(`--allow-missing`이면 경고), 샘플에만 있는 태그 → 참고로 출력만, 값이 안 변한 태그 → 경고, `--resample-interval-ms` 미지정 시 권장값을 YAML 기본값으로 쓰고 그 사실을 출력. 기본값: 격자 100, 지연 500, 윈도우 30, `min_samples` 3000.
- 생성한 YAML을 `load_equipment_config`로 검증해 실패하면 종료 코드 1.

**Tests (작은 엑셀을 `openpyxl`로 tmp에 생성):**
- `Group`/`StateTag`/`Tag` 열 사용 시 그룹/상태 태그/태그 이름이 그대로 반영.
- 열이 없을 때 `--group-regex`/`--group-template`/`--state-pattern` 추론, 일치하지 않는 행은 `general`.
- 이름 변환(`AxX.Act.Pos` → `NX5_AxisData:AxX_Act_Pos`), 끝의 `[` 제거와 경고, 빈 행 건너뜀, 토픽 중복 제거 순서 유지.
- 샘플 검증: 누락 태그 → 종료 코드 1, `--allow-missing` → 0, 샘플에만 있는 태그가 YAML에 **없음**, 간격 권장(100ms 샘플 → 100)과 기본값 사용 안내.
- 생성 YAML이 `load_equipment_config`로 로드되고 `groups`/`state_tag`가 의도대로인지.
- 실제 파일 확인(자동 테스트 아님): `DataList.xlsx` + `JSONData` 샘플로 실행해 NX5 설정을 생성한다.

**Steps:**
- [ ] `pyproject.toml`에 `[project.optional-dependencies] tools = ["openpyxl"]`, dev 그룹에 `openpyxl`, `[project.scripts] jetson-datalist = "jetson_app.datalist_to_config:main"` 추가 후 `uv sync --extra tools`(lock 갱신).
- [ ] 테스트 작성 → 실패 확인 → 구현 → 통과 → 전체 통과.
- [ ] 실제 실행: `uv run jetson-datalist ../DataList.xlsx --equipment-id nx5 --collector-prefix NX5 --group-regex "공정(\d+)" --group-template "process_{}" --state-pattern "U{}_ProcStart" --samples ../JSONData --output configs/nx5.yaml` → 종료 코드 0, 태그 46개, 그룹 12개(`process_0`~`process_10`, `general`), 격자 100 권장 출력을 확인한다.
- [ ] 커밋(`feat: add DataList-to-config generator and NX5 config`).

---

### Task 7: 문서와 최종 검증

**Files:** Modify `README.md`, `docs/superpowers/specs/2026-10-01-process-aware-scoring-design.md`

- [ ] README에 (1) `groups:`/`state_tag`/`alarm`/`training` 설정, (2) 그룹별 발행 필드, (3) 새 장비를 붙이는 순서(받을 태그 목록 작성 → `jetson-datalist` → 앱 실행 → 충분히 쌓이면 `train` → 모니터링), (4) 상태별 기준과 정상 데이터에 동작/대기 상태가 모두 들어가야 한다는 주의, (5) 이 도구가 태그를 추가하지 않고 샘플은 검증에만 쓰인다는 점을 적는다.
- [ ] 스펙 `Status`를 `Implemented`로 바꾸고 구현 노트를 추가한다(계획서의 간결 형식 사용, 미해결: 임계값/하한 튜닝).
- [ ] 전체 `uv run pytest -q` 통과와 `uv run python -c "import jetson_app.subscriber_cli"` 확인.
- [ ] 커밋(`docs: document group scoring and the datalist tool`).
