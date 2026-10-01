# 그룹별 점수와 동작/대기 상태별 기준 설계

- Status: Implemented (2026-10-01), 구현 계획: [`../plans/2026-10-01-group-scoring.md`](../plans/2026-10-01-group-scoring.md)
- Date: 2026-10-01
- 관련 문서: [`2026-10-01-event-time-resampler-design.md`](2026-10-01-event-time-resampler-design.md) (구현 완료), [`2026-08-04-jetson-anomaly-inference-pipeline-design.md`](2026-08-04-jetson-anomaly-inference-pipeline-design.md), [`2026-07-31-jetson-dx1-anomaly-framework-design.md`](2026-07-31-jetson-dx1-anomaly-framework-design.md)

## 배경과 목표

첫 대상 설비(NX5 데모 라인)는 공정0~10이 컨베이어로 이어진 파이프라인이다. 워크 하나가 공정0→10을 거치는 데 약 150초이고, 워크를 하나 더 넣으면 약 30초 늘어난다(앞 워크가 공정 i+1을 처리하는 동안 뒤 워크가 공정 i를 처리). 서보는 공정0(`AxX`), 공정4(`AxZ`), 공정9(`AxCV`)에 있고, 공정마다 `U{N}_ProcStart`(동작 중 신호)와 `U{N}_TaktTime`이 있다. `ConvSensor0~10`, `ConvAct0~3`은 특정 공정에 속하지 않는 컨베이어 신호다(`DataList.xlsx`).

현재 구현은 모든 태그를 하나로 묶어 "다음 시점 예측 오차의 태그별 z-score 중 최댓값" 하나를 이상 점수로 낸다. 이 구조로는 다음 두 가지가 안 된다.

1. **어느 부분의 이상인지 바로 알 수 없다.** `top_deviant_tag`에서 공정을 추론해야 한다.
2. **동작하지 않는 동안의 서보 충격을 놓칠 수 있다.** 태그의 "정상 오차" 기준(평균/표준편차)을 동작 중과 대기 중을 섞어 하나로 계산하므로, 동작 중에 크게 나는 오차에 기준이 맞춰져 대기 중의 작은 충격은 점수가 낮게 나온다.

**목표**
- 태그를 사용자가 정한 **그룹**으로 묶어 그룹별로 이상 점수와 알람을 따로 계산해 발행한다. 전체 점수/알람도 그대로 유지한다.
- 그룹에 **상태 태그**(동작 중 신호)를 지정하면, 그 그룹의 태그를 동작 중 / 대기 중 상태별 정상 기준으로 판정해 동작 중 이상과 대기 중 충격을 모두 감지한다.
- **장비에 독립적**이어야 한다. 새 장비는 "받을 태그를 적은 목록 + 설정 파일"만으로 붙는다. 그룹 구조는 NX5 전용이 아니라 일반 개념이다.

**성공 기준**
- 그룹별 점수/알람이 발행되고, 어느 그룹의 어느 태그가 원인인지 알 수 있다.
- 같은 크기의 오차가 대기 상태에서는 더 큰 점수를 내고 동작 상태에서는 더 작은 점수를 낸다(단위 테스트로 검증).
- 기존 평면 `tags:` 설정(그룹 없음)은 지금처럼 동작한다.
- 기존 188개 테스트가 계속 통과한다.

## 결정 사항 (대화에서 합의)

- **태그는 사용자가 직접 정한다.** 어떤 데이터를 받을지는 초기에 사용자가 적는 목록(엑셀, `DataList.xlsx`와 같은 형태)이 유일한 기준이다. 도구는 태그를 자동으로 찾아 추가하거나 빼지 않는다. 샘플 데이터는 검증과 권장값 제안에만 쓴다.
- 앞으로 붙일 다른 장비도 DX1이 같은 JSON 형식(`{"records":[…]}`)으로 MQTT 발행한다 → 입력부는 지금 그대로 쓴다.
- 모델은 **설비 통합 GRU 1개**를 유지한다(그룹별 모델로 쪼개지 않는다). 점수 계산과 기준만 그룹별/상태별로 나눈다.
- "동작하지 않음"은 그룹의 상태 태그가 꺼진 것으로 판단한다. 사이클 시간은 필요하지 않다.
- 워크 1개를 사이클로 자르지 않는다. 학습 데이터는 시연 방식(2개 투입, 종료, 1분 대기 반복)의 연속 수집을 그대로 쓴다.
- 격자 간격(`resample_interval_ms`)은 config에 직접 적는다(자동 감지 없음, ① 결정 유지). 도구는 샘플에서 권장값을 계산해 설정 초안의 기본값으로 넣어줄 뿐이다.
- 컨베이어 센서/실린더는 상태 태그가 없는 그룹 `general`에 둔다.

## 1. 설정 스키마

`groups:`를 쓰는 새 형식을 추가한다. 기존 평면 `tags:` 형식도 계속 지원한다.

```yaml
equipment_id: "nx5"
mqtt: {subscribe_topics: [...], publish_topic: "jetson/nx5/anomaly", command_topic: "jetson/nx5/cmd"}
groups:
  process_0:
    state_tag: "NX5_ProcStart:U0_ProcStart"
    tags:
      - "NX5_ProcStart:U0_ProcStart"
      - "NX5_TaktTime:U0_TaktTime"
      - "NX5_AxisData:AxX_Act_Pos"
      - "NX5_AxisData:AxX_Act_Vel"
      - "NX5_AxisData:AxX_Act_Trq"
  process_4: { state_tag: ..., tags: [...] }
  # ... process_10까지
  general:
    tags: ["NX5_SenData:ConvSensor0", ..., "NX5_SenData:ConvAct3"]
resample_interval_ms: 100
max_lateness_ms: 500
window_size: 30
calibration: {max_duration: "7d", min_samples: 3000}
alarm: {threshold: 3.0, confirm_steps: 3}      # 선택, 기본값은 현재 Debouncer 기본값
training: {max_samples: 60000, epochs: 20}     # 선택, 기본 max_samples 20000, epochs 20
```

**규칙**
- `tags:`와 `groups:`는 **동시에 쓸 수 없다**(둘 다 있으면 `ConfigError`). 둘 다 없어도 오류. `groups`는 비어 있지 않은 매핑이어야 한다.
- 한 태그는 정확히 한 그룹에만 속한다(중복 시 `ConfigError`). `state_tag`는 선택이며, 있으면 반드시 그 그룹의 `tags`에 포함돼야 한다(모델 입력에 필요하다).
- 전체 `tags`는 그룹 선언 순서대로 이어 붙여 만든다(순서 고정, 모델 입력 순서가 된다).
- 평면 `tags:` 설정은 `state_tag`가 없는 단일 그룹 `all`로 취급한다. 이때는 그룹별 필드를 발행하지 않는다(현재 발행 형식 유지).
- `alarm.threshold`(양수), `alarm.confirm_steps`(양의 정수), `training.max_samples`(양의 정수), `training.epochs`(양의 정수)를 검증한다. 모두 선택이며 기본값은 현재 코드의 값과 같다.

## 2. 상태별 정상 오차 기준 (학습)

- 그룹에 `state_tag`가 있으면, 그 그룹의 태그(단, `state_tag` 자신은 제외)마다 정상 오차 통계를 **동작 중(ON)** 과 **대기 중(OFF)** 으로 나눠 계산한다. 학습 윈도우의 타깃(다음 시점)에서 `state_tag`의 **원본** 값이 0.5 이상이면 ON, 아니면 OFF다. 판정은 정규화 전 값으로 한다.
- 통계 = (평균, 표준편차, 표본 수). 표준편차 하한은 기존 `_floor_std`를 그대로 쓴다.
- 한 상태의 표본이 `MIN_REGIME_SAMPLES`(100, 약 10초)보다 적으면 그 상태의 기준은 기존의 **전체(pooled) 통계로 대체**하고, 학습 로그에 대체된 (태그, 상태) 목록을 남긴다. `state_tag`가 학습 동안 한 번도 켜지지 않았거나 꺼지지 않은 경우가 여기에 해당한다.
- `state_tag`가 없는 그룹(`general`, 평면 설정)과 `state_tag` 자신은 기존의 전체 통계를 쓴다.
- 기존 `error_stats`(전체 통계)는 그대로 유지한다.

## 3. 모델 artifact

`ModelArtifact`에 두 필드를 추가한다.

- `groups: dict[str, tuple[str | None, tuple[str, ...]]]` — 그룹 이름 → (`state_tag`, 태그 목록).
- `regime_error_stats: dict[str, dict[str, tuple[float, float, int]]]` — 태그 → {`"on"`/`"off"` → (평균, 표준편차, 표본 수)}. 상태별 통계가 없는 태그는 키가 없다.

저장/로드가 두 필드를 보존한다. `pipeline.py`의 호환성 검사(`_check_artifact_matches_config`)가 `groups`가 config와 다르면 기존과 같은 방식으로 `CALIBRATING`으로 폴백한다. 이 필드가 없는 기존 artifact는 불일치로 취급해 재학습을 유도한다(① 때의 `resample_interval_ms` 처리와 같은 방식).

## 4. 실시간 점수 계산

`InferenceEngine.score(window, actual)`가 태그별 z-score를 구하는 방식은 같지만, 기준(평균/표준편차)을 고르는 규칙이 바뀐다.

- 태그가 `state_tag`가 있는 그룹에 속하면, `actual` 스냅샷의 `state_tag` 값으로 ON/OFF를 정하고 그 상태의 통계를 쓴다(해당 상태 통계가 없으면 전체 통계). `state_tag` 값이 `None`이면 전체 통계를 쓴다.
- **그룹 점수** = 그 그룹 태그들의 z-score 최댓값, **그룹 `top_tag`** = 그 태그.
- **전체 점수** = 모든 그룹 점수의 최댓값, **전체 `top_deviant_tag`** = 그 태그(기존과 동일한 의미).
- 반환 타입 `AnomalyResult`에 `group_results: dict[str, GroupResult]`를 추가한다(`GroupResult` = `score: float`, `top_tag: str`). 기존 필드(`anomaly_score`, `top_deviant_tag`)는 그대로.

## 5. 알람과 발행

- `Debouncer`를 **그룹마다 하나씩** 둔다(임계값 `alarm.threshold`, 연속 횟수 `alarm.confirm_steps`). 각 그룹의 점수로 그룹 알람을 확정한다. 모델 교체/윈도우 리셋 시 모든 그룹의 디바운서를 리셋한다.
- 전체 알람 = 그룹 알람 중 하나라도 참.
- **발행 스키마**(그룹이 둘 이상일 때만 그룹 필드를 추가. 평면 설정은 기존 3개 필드만):

```json
{"records": [{"timestamp": "...",
  "jetson:anomaly_score": 4.2, "jetson:alarm": true, "jetson:top_deviant_tag": "NX5_AxisData:AxZ_Act_Trq",
  "jetson:process_4:score": 4.2, "jetson:process_4:alarm": true, "jetson:process_4:top_tag": "NX5_AxisData:AxZ_Act_Trq",
  "jetson:process_0:score": 0.3, "jetson:process_0:alarm": false, "jetson:process_0:top_tag": "NX5_AxisData:AxX_Act_Vel",
  "jetson:general:score": 0.1, "...": "..."}]}
```

- 발행 주기는 지금과 같이 스텝마다다(100ms 격자면 초당 10회). 필드가 40여 개로 늘어 메시지가 커지지만 초당 수 KB 수준이다.

## 6. 설정 생성 도구 (사용자가 적은 목록 → 설정 초안)

`jetson_app/tools/datalist_to_config.py`(개발 도구이므로 `openpyxl`은 dev 의존성 그룹에만 추가). **입력은 사용자가 적은 목록(엑셀)뿐이며, 도구는 태그를 추가/삭제하지 않는다.**

**엑셀 열**(1행은 헤더, 순서 무관, 대소문자 무시)

| 열 | 필수 | 의미 |
|---|---|---|
| `Variables` | 필수 | 변수 이름 |
| `MQTT Topic` | 필수 | 그 변수가 발행되는 토픽 |
| `Description` | 선택 | 설명(그룹 추론에만 사용) |
| `Tag` | 선택 | DX1이 내보내는 **정확한 태그 이름**. 있으면 그대로 쓴다 |
| `Group` | 선택 | 그룹 이름. 비어 있으면 `general` |
| `StateTag` | 선택 | 그 그룹의 상태 태그에 표시(`Y`) |

- `Tag`가 없으면 태그 이름을 `<--collector-prefix>_<토픽 마지막 마디>:<변수명의 '.'→'_'>`로 만든다(예: 접두사 `NX5`, 토픽 `dx1/AxisData`, 변수 `AxX.Act.Pos` → `NX5_AxisData:AxX_Act_Pos`). 변수명 끝의 `[`처럼 잘못 붙은 문자는 제거하고 경고를 출력한다. 실제 이름은 샘플 검증(아래)에서 확인한다.
- `Group` 열이 없으면 `--group-regex`(예: `공정(\d+)`)와 `--group-template`(예: `process_{}`)으로 `Description`에서 그룹을 만든다. 둘 다 없으면 모든 태그를 `general` 하나에 둔다. `--state-pattern`(예: `U{}_ProcStart`, `{}`는 정규식의 첫 번째 그룹 값)으로 그룹의 상태 태그를 지정할 수 있다.
- 구독 토픽은 엑셀의 토픽 열에서 중복 없이 순서대로 뽑는다. `publish_topic`/`command_topic`은 `jetson/<equipment-id>/anomaly|cmd`.
- 격자/지연/윈도우/캘리브레이션 값은 인자(`--resample-interval-ms` 등)로 받고, 없으면 아래 샘플 권장값 또는 기본값(`resample_interval_ms` 100, `max_lateness_ms` 500, `window_size` 30, `min_samples` 3000)을 쓴다. 생성된 YAML에서 사용자가 언제든 고칠 수 있다.

**샘플 JSON 검증과 권장값 (`--samples <파일 또는 폴더>`, 선택)**
- 샘플의 각 줄(`{"records":[…]}`)에서 모든 키를 모아, **목록의 모든 태그가 샘플에 있는지** 확인한다. 없는 태그가 있으면 이름을 출력하고 오류로 종료한다(`--allow-missing`이면 경고만).
- 샘플에는 있지만 목록에는 없는 태그는 **참고로만 출력**하고 설정에는 넣지 않는다.
- record 간격의 중앙값으로 `resample_interval_ms`를 **권장**한다(10ms 단위로 반올림). 사용자가 인자로 정하지 않았으면 이 값을 설정의 기본값으로 쓰고, 그렇게 했다고 출력한다.
- 샘플 동안 값이 전혀 변하지 않은 태그를 경고로 출력한다(정지 상태 샘플이라는 신호일 수 있다. 제외하지는 않는다).

**검증**: 생성된 YAML이 `load_equipment_config`로 로드되는지 확인하고 실패하면 오류로 종료한다.

## 7. 에러 처리

| 상황 | 처리 |
|---|---|
| 한 상태(ON/OFF)의 학습 표본 부족 | 전체 통계로 대체, 로그 |
| `state_tag` 값이 `None`인 스텝 | 그 스텝은 전체 통계 사용 |
| 저장된 artifact의 `groups`가 config와 다름/없음 | `CALIBRATING`으로 폴백(기존 정책) |
| 같은 태그가 둘 이상의 그룹에 있음, `state_tag`가 그룹 `tags`에 없음 | 기동 시 `ConfigError` |
| 생성 도구: 목록의 태그가 샘플에 없음 | 오류 종료(`--allow-missing`이면 경고) |
| 생성 도구가 만든 YAML이 로드 불가 | 도구가 오류로 종료 |

## 8. 테스트 계획

- config: `groups` 파싱, 태그 순서, 중복/누락 검증, `tags:`와 동시 사용 거부, 평면 설정이 단일 그룹으로 동작, `alarm`/`training` 기본값과 검증.
- 학습: 상태별 오차 통계 계산(ON/OFF 분리, 정규화 전 값으로 판정), 표본 부족 시 전체 통계 대체와 로그, artifact 저장/로드/호환성 검사.
- 점수: **같은 크기의 오차가 OFF 상태에서는 높은 점수, ON 상태에서는 낮은 점수**가 나오는지(오차를 직접 주입하는 결정적 테스트), 그룹 점수/`top_tag`, 전체 점수가 그룹 점수의 최댓값인지, `state_tag`가 `None`일 때 전체 통계로 동작하는지.
- 알람/발행: 그룹별 디바운서 독립 동작, 그룹이 하나일 때와 둘 이상일 때의 발행 스키마, 리셋 시 모든 디바운서 초기화.
- 생성 도구: 작은 엑셀을 만들어 `Group`/`StateTag`/`Tag` 열, 열이 없을 때의 정규식 추론, 태그 이름 변환, 오탈자(`[`) 경고, 샘플 검증(누락 태그 오류, 목록에 없는 태그는 설정에 안 들어감), 간격 권장, 로드 검증을 확인한다. 실제 `DataList.xlsx`와 `JSONData` 샘플로 한 번 실행해 태그가 샘플 키와 일치하는지 확인한다.
- 기존 188개 테스트는 새 구조에 맞게 수정하되 의도는 유지한다.

## 범위 밖 / 추후

- **임계값과 표준편차 하한 튜닝**: 동작/대기 기준이 생겨도 대기 중 서보 노이즈가 매우 작으면 사소한 변동에도 점수가 크게 나올 수 있다. 실제 가동 데이터로 `alarm.threshold`, `_floor_std`를 튜닝한다(설정으로 조정 가능하게 둔다). 학습 데이터 일부를 따로 떼어 점수 분포로 임계값을 장비별로 제안하는 기능은 실제 데이터로 검증한 뒤 일반화한다.
- **Takt Time 지연 검사**(값 자체를 정상 분포와 비교): 별도 기능으로 나중에 설계한다.
- **공백을 건너뛰는 학습 윈도우**: 통신 단절로 캘리브레이션 버퍼에 생긴 공백을 가로지르는 윈도우를 학습에서 걸러내는 처리는 하지 않는다.
- 비숫자 값(문자열/`null`/`true`) 검증, 그룹별 별도 모델, "너무 오래 멈춤"(모든 상태 태그가 꺼진 시간) 규칙 검사, DX1 JSON 이외의 입력 형식.

## 영향 받는 파일

`config.py`(스키마), `training.py`(상태별 통계, artifact, 설정 가능한 학습 상한/에폭), `inference.py`(그룹 점수), `debounce.py`(변경 없음, 그룹별 인스턴스 사용), `snapshot_processor.py`(그룹별 디바운서와 발행), `publisher.py`(스키마 확장), `pipeline.py`(연결과 호환성 검사), `subscriber_cli.py`(학습 설정 전달), 신규 `tools/datalist_to_config.py`, `pyproject.toml`(dev 의존성), `README.md`, 대응 테스트 파일.

## 구현 노트

- 그룹 점수는 그룹 안의 모든 태그의 z-score 최댓값이며, **상태 태그 자신도 포함**한다(상태 태그는 전체 통계로 판정된다).
- 실행 중 `state_tag` 값이 `None`인 경우는 점수 계산 앞단에서 이미 그 스텝이 건너뛰어지므로(윈도우/실제값에 `None`이 있으면 채점하지 않는 기존 규칙) 실제로 도달하지 않는다. `InferenceEngine._stats_for`는 방어적으로 전체 통계를 쓰도록 구현되어 있다.
- 처리기는 결과에 그룹 정보가 없으면(예: 테스트용 가짜 엔진) 전체 점수를 단일 그룹 `all`로 보고 `debouncers["all"]`을 쓴다.
- 설정 생성 도구는 `jetson_app/datalist_to_config.py`(진입점 `jetson-datalist`)이며 `openpyxl`은 `tools` 선택 의존성과 dev 그룹에만 있다. 실제 `DataList.xlsx`와 `JSONData` 샘플로 `configs/nx5.yaml`(그룹 12개, 태그 46개, 격자 100ms 권장)을 생성했다.
- **미해결(추후 튜닝):** 대기 중 서보 노이즈가 매우 작으면 사소한 변동에도 점수가 크게 나올 수 있다. 실제 가동 데이터로 `alarm.threshold`와 표준편차 하한(`_floor_std`)을 튜닝한다.
