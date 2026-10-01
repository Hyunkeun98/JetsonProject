# 이벤트 시간 기반 리샘플러 설계

- Status: Implemented (2026-10-01), 구현 계획: [`../plans/2026-10-01-event-time-resampler.md`](../plans/2026-10-01-event-time-resampler.md)
- Date: 2026-10-01
- 관련 문서: [`2026-08-04-jetson-anomaly-inference-pipeline-design.md`](2026-08-04-jetson-anomaly-inference-pipeline-design.md) (Tag Buffer / 스냅샷 스케줄러 / 캘리브레이션 / 추론)

## 배경과 문제

현재 파이프라인은 Jetson 시계 기준으로 `resample_interval_ms`(기본 50ms)마다 `TagBuffer`의 **최신값 하나**를 스냅샷으로 찍는다. 파서(`parse_and_filter_records`)는 record의 `timestamp`를 문자열로 보관만 하고 해석하지 않으며, `on_record`는 한 MQTT 메시지의 record들을 도착 즉시 순서대로 `TagBuffer.update()`에 덮어쓴다.

DX1(SpeeDBee Synapse)은 Collector 주기(현재 100ms, 앞으로 더 짧아질 수 있음)로 취득한 값을 **여러 개 묶어 한 메시지로** 발행한다. 실제 예시는 메시지 하나에 `records` 10개(약 100ms 간격)가 들어 있다. 이 경우:

- 한 메시지의 중간 값 9개가 스냅샷에 한 번도 보이지 않는다(마지막 값만 남음).
- 시점이 취득 시각이 아니라 메시지 도착 시각으로 정해져, 신호가 사실상 메시지 주기(약 1Hz)로 샘플링된 것과 같아진다.

## 목표 / 성공 기준

- 취득 주기가 100ms든 그보다 짧든(또는 길든), **config의 `resample_interval_ms` 값만 바꿔서** 코드 수정 없이 동작한다.
- 한 메시지에 record가 몇 개 들어 있든 시점 정보를 잃지 않는다.
- 기존 테스트(131개)가 계속 통과하고, 배치/주기별 신규 테스트를 추가한다.

## 결정 사항

- 시점 기준은 도착 시각이 아니라 **각 record의 `timestamp`(이벤트 시간)**.
- 격자 간격은 **config에 수동으로 지정**한다(`resample_interval_ms`, 기존 항목 유지). 자동 감지는 하지 않는다.
- 격자보다 빠른 신호는 칸 안의 마지막 값, 느린 신호는 직전 값 유지(ffill).
- 보간, 칸 평균 다운샘플링은 범위 밖.

## 1. 타임스탬프 파싱 — 신규 `timeparse.py`

- DX1 형식 `2026-10-01T00:47:01.718651520+0000`(나노초 9자리, 오프셋 `+0000`)은 Python 3.8의 `datetime.fromisoformat`이 읽지 못한다. 직접 파싱해 **정수 ns(epoch)**로 변환한다.
- 허용 형식: `YYYY-MM-DDTHH:MM:SS[.fraction](Z|±HHMM|±HH:MM)`. 소수부는 0~9자리.
- 파싱 실패 시 `None`을 반환한다. 호출부(파서)는 해당 record를 버리고 누적 개수를 로그에 남긴다(메시지마다 출력하지 않고 일정 개수 단위로).
- `Record`에 이벤트 시각 필드(`epoch_ns: int`)를 추가한다. 기존 `timestamp: str`은 유지한다.

## 2. 이벤트 시간 리샘플러 — `buffer.py`의 `TagBuffer` 대체

격자 간격 `Δ = resample_interval_ms`. 칸 `k`는 `[k·Δ, (k+1)·Δ)` (epoch 기준 절대 격자).

**입력**: `add(record)` — MQTT 스레드에서 호출. 태그별로 칸 안의 마지막 값을 보관한다.

**칸 확정(워터마크)**: 지금까지 본 record 중 가장 큰 이벤트 시각을 `T_max`라 할 때, `T_max − max_lateness_ms` 보다 완전히 이전인 칸은 확정한다. 확정된 칸은 오름차순으로 스냅샷으로 방출한다.

- `max_lateness_ms`: 신규 config 항목, 기본값 **2000**. 다중 토픽에서 한 토픽이 다른 토픽보다 늦게 도착해도 같은 시점의 값이 한 스냅샷에 모이도록 기다리는 시간이다. 결과 발행은 최대 이 값만큼 지연된다.

**스냅샷 값 결정**: 칸 안에 그 태그의 record가 있으면 그중 마지막 값, 없으면 직전 칸까지의 값 유지(ffill). 한 번도 값을 못 받은 태그는 `None`(기존 정책과 동일 — 모든 태그가 `None`인 칸은 방출하지 않음).

**늦은 record**: 이미 확정된 칸에 속하는 record는 버리고 개수를 로그에 남긴다.

**긴 공백**: 방출할 칸 사이의 간격이 윈도우 길이(`window_size × Δ`)보다 길면 통신 단절로 보고 ffill 칸을 대량 생성하지 않는다. 대신 **슬라이딩 윈도우를 비우고** 새 시점부터 다시 시작한다(캘리브레이션 버퍼에는 공백 구간을 채우지 않음).

**출력**: 확정된 스냅샷은 `(칸 시작 시각, Snapshot)`으로 큐에 넣는다. 큐는 크기를 제한하고(기본 10,000), 넘치면 가장 오래된 항목을 버리며 로그를 남긴다.

## 3. 처리 스레드 — `scheduler.py`의 `PeriodicSnapshotter` 대체

- 시계 기반 `_stop_event.wait(interval)` 루프를 큐 소비 루프로 바꾼다. MQTT 스레드는 `add()`와 큐 적재만 한다.
- 소비 스레드는 큐에서 스냅샷을 꺼낼 때마다 현재 `_tick`의 본문을 그대로 수행한다: 점수 계산 → 슬라이딩 윈도우에 추가 → 캘리브레이션 기록 → 하트비트 로그(약 100개 처리마다).
- 발행하는 결과 / 캘리브레이션 샘플의 시각은 `datetime.now()`가 아니라 **해당 칸의 이벤트 시각**(ISO 8601, UTC)을 쓴다.
- 기존 `start()`/`stop()` 재시작 의미(stop 후 start 시 이벤트 초기화)를 유지한다.
- 시계를 쓰지 않으므로 테스트는 `sleep` 없이 결정적으로 작성한다.

## 4. 설정과 모델 호환성

- `resample_interval_ms`: 기존 항목 그대로(양의 정수). 취득 주기를 바꾸면 이 값을 같이 바꾼다.
- `max_lateness_ms`: 신규, 선택, 기본 2000, 0 이상 정수. 검증 실패 시 `ConfigError`.
- **학습 모델에 격자 값을 저장**한다(`ModelArtifact.resample_interval_ms`). `pipeline.py`의 `_check_artifact_matches_config`에서 `window_size`, `tags`와 함께 이 값도 비교해, 다르면 기존과 같은 방식으로 `CALIBRATING`으로 폴백한다. 이 필드가 없는 기존 artifact는 불일치로 취급해 재학습하게 한다.
- **알려진 제약**: `CALIBRATING` 도중 격자를 바꾸면 이전 버퍼(`<equipment_id>.jsonl`)의 간격과 섞인다. 격자를 바꿀 때는 `recalibrate`로 버퍼를 비워야 한다. README에 명시한다(자동 감지는 범위 밖).

## 5. 에러 처리 요약

| 상황 | 처리 |
|---|---|
| 타임스탬프 파싱 실패 | record 폐기 + 누적 로그 |
| 확정된 칸보다 늦게 도착한 record | 폐기 + 누적 로그 |
| 큐 초과 | 가장 오래된 스냅샷 폐기 + 로그 |
| 윈도우 길이를 넘는 공백 | 슬라이딩 윈도우 비우고 재시작 |
| artifact 격자/`window_size`/`tags` 불일치 | `CALIBRATING` 폴백(기존 정책) |

## 6. 테스트 계획

- `timeparse`: 나노초 9자리, `+0000`/`Z`/`+09:00`, 소수부 0~9자리, 잘못된 형식은 `None`.
- 파서: record가 이벤트 시각을 갖는지, 불량 타임스탬프 record 폐기.
- 리샘플러: 10개 배치 메시지 → 서로 다른 10개 칸. Δ가 20ms / 100ms / 1000ms일 때 각각의 칸 수와 값. 취득이 격자보다 빠르면 칸 안 마지막 값, 느리면 ffill. 다중 토픽 어긋남(워터마크 지연 허용 내 정렬). 늦은 record 폐기. 긴 공백 시 윈도우 리셋. 큐 초과 시 폐기.
- 소비 스레드: 캘리브레이션 기록 / 점수 계산 / 하트비트가 기존과 동일하게 동작(기존 `test_scheduler.py`, `test_pipeline.py`를 이벤트 구동 방식으로 이관).
- config: `max_lateness_ms` 검증.
- artifact: 격자 값 저장/로드, 불일치 시 폴백.
- 기존 131개 테스트는 새 구조에 맞게 수정하되 검증 의도는 유지한다.

## 7. 영향 받는 파일

`mqtt_subscriber.py`(Record, 파서), `buffer.py`(리샘플러), `scheduler.py`(큐 소비자), `config.py`, `training.py`(artifact 필드), `pipeline.py`(연결과 호환성 검사), `README.md`, `configs/test_dx1.example.yaml`, 신규 `timeparse.py`, 그리고 대응 테스트 파일.

## 구현 노트

- 스펙의 `buffer.py`/`scheduler.py` 수정 대신 `resampler.py`(`EventTimeResampler`, `ResampledStep`)와 `snapshot_processor.py`(`SnapshotProcessor`)를 새로 만들고, 기존 `TagBuffer`와 `scheduler.py`는 제거했다. 동작은 스펙과 같다.
- 처리 스레드는 큐를 `get(timeout=0.1)`로 소비하고, `stop()`은 스레드를 join한 뒤 남은 큐를 호출 스레드에서 마저 처리한다. 테스트는 `process_pending()`으로 스레드 없이 결정적으로 검증한다.
- 격자 정보가 없는 기존 artifact는 `resample_interval_ms = 0`으로 읽혀 항상 불일치로 처리된다(재학습 유도).
- **미해결(추후 보강):** record 값의 타입 검증(문자열/`null`/`true` 등 비숫자 값)은 아직 하지 않는다. 실제 토픽 샘플을 확보한 뒤 보강한다.
