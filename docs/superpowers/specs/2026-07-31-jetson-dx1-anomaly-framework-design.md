# Jetson 실시간 이상탐지 프레임워크 설계

- Status: Approved (design), pending implementation plan
- Date: 2026-07-31

## 배경 및 목표

DX1(OMRON Data Flow Controller, SpeeDBee Synapse 내장)이 현장 설비(PLC/서보모터/센서)에서 데이터를
취득하고, GPU 성능이 부족해 AI 추론은 Jetson PC에서 수행한다. 목표는 **"평소와 다른 이상"(고장 가능성)을
사전에 감지**하는 실시간 예지보전(PdM) 엔진을 Jetson에 구축하는 것이다.

핵심 제약:
- **실제 대상 설비는 아직 미정.** 지금까지 사용한 도루코(Dorco) 44호기 CSV 데이터는 파이프라인 메커니즘이
  정상 동작하는지 검증하기 위한 테스트용 샘플일 뿐, 실제 타겟 설비의 데이터가 아니다.
- **실제 설비에는 불량/고장 라벨이나 임계값이 없다.** 따라서 지도학습(도루코 스크립트의 `IS_NG` 공차 판정
  같은 방식)은 쓸 수 없고, 비지도 이상탐지 접근이 필요하다.
- 통신은 MQTT(JSON)를 기본 가정으로 한다. DX1의 SpeeDBee Synapse는 **MQTT Emitter(Publish)**와
  **MQTT Collector(Subscribe)**를 모두 지원하므로, DX1이 Publish하고 Jetson이 Subscribe하는 구조가
  자연스럽다.
- 새 설비가 추가될 때 Jetson이 태그를 자동 탐색하지 않는다. **엔지니어가 설비별 config로 태그 목록을 사전
  등록**한다.

## 1. 전체 아키텍처

```
[DX1] SpeeDBee Synapse
   PLC/Modbus/IO-Link Collector → (센서/서보모터 태그 수집)
   → JSON Serializer ("records": [{timestamp, "component:tag": value, ...}])
   → MQTT Emitter (Publish)
          │
          │  MQTT (JSON)
          ▼
[Jetson] 실시간 추론 프레임워크
   1. MQTT Subscriber   — 설비별 config에 정의된 태그만 필터링/수신
   2. Tag Buffer        — 설비별로 태그값을 슬라이딩 윈도우로 버퍼링
   3. Calibration Manager — 설비 최초 가동 시 일정 기간을 "정상"으로 간주,
                            그 구간 데이터로 예측 모델을 학습·고정
   4. Inference Engine  — 실시간 윈도우로 다음 시점 값을 예측 → 실제값과의
                          오차(prediction error)를 이상 점수로 환산
   5. Result Publisher  — 이상 점수/알람(및 하위 분석 결과)을 다시 MQTT로 Publish
          │
          │  MQTT (JSON, 이상 점수/알람)
          ▼
[DX1] MQTT Collector → 대시보드/기존 패키지에 통합
```

**설계 원칙**: 설비별로 독립된 파이프라인(설비마다 별도 config, 별도 모델 상태)을 유지하되, 도루코 스크립트의
하드코딩된 공차 규칙(`IS_NG` 판정) 같은 설비 전용 로직은 두지 않는다. 새 설비를 붙이는 데 필요한 건 config
파일 작성뿐이며, 코드 변경이 없어야 한다.

## 2. 모델 선택: LSTM/GRU 기반 시계열 예측

**결정한 방식**: 캘리브레이션(정상 가동 초기) 기간 데이터로 LSTM/GRU 예측 모델을 학습한다. 실시간으로는
최근 윈도우를 입력으로 다음 시점 값을 예측하고, 실제값과의 오차를 이상 점수로 쓴다.

**왜 지도학습이 아닌가**: 이 방식은 "이 시점이 불량이었다"는 결과 라벨을 한 번도 사용하지 않는다. 캘리브레이션
기간은 "정상 가동 초기일 것"이라는 가정 하나만 제공하고, 모델은 정상 패턴만 학습한다 — 산업 PdM에서 표준적인
비지도 이상탐지 패러다임이다.

**왜 오토인코더가 아닌 예측(Forecasting) 모델인가**: 오토인코더도 이미지 전용이 아니라 구조적 패턴(인코더로
압축, 디코더로 복원)일 뿐이라 시계열에도 적용 가능하지만(LSTM 오토인코더 등), 예측 모델이 "센서가 곧 이
값을 낼 것이다 → 크게 벗어나면 전조"라는 프레이밍이 더 직관적이고, 서보모터 추세성 전조 감지에 더 적합하다고
판단해 1차 목표로 선정했다. 데이터가 쌓이면 Transformer 기반 등으로 고도화하는 경로는 열어둔다.

## 3. 설비별 Config & 데이터 계약

**설비별 config** (설비 1대당 YAML 파일 1개, 예: `configs/dorco_44.yaml`):

```yaml
equipment_id: "dorco_44"
mqtt:
  subscribe_topic: "dx1/dorco_44/telemetry"
  publish_topic: "jetson/dorco_44/anomaly"
tags:
  - "servo1:torque"
  - "servo1:position"
  - "sensor:A_L_01"
  - "sensor:X_L_01"
  # ... 필요한 만큼
resample_interval: "1s"        # 태그별 원천 주기가 달라도 공통 리샘플 기준
window_size: 60                 # 모델 입력에 쓰는 과거 스텝 수
calibration:
  duration: "3d"                 # 이 기간 데이터로 "정상" 모델 학습
  min_samples: 10000              # 부족하면 캘리브레이션 자동 연장
```

**수신 스키마** (DX1 SpeeDBee MQTT Emitter 표준 포맷 그대로 사용):
```json
{"records": [{"timestamp": "2023-10-02T05:54:54.837742200Z", "servo1:torque": 12.3, "sensor:A_L_01": 110.2}]}
```

**발행 스키마** (Jetson → DX1, 동일 포맷 체계 사용):
```json
{"records": [{"timestamp": "...Z",
  "jetson:anomaly_score": 0.82,
  "jetson:alarm": true,
  "jetson:top_deviant_tag": "servo1:torque"}]}
```

> **참고**: 이 config/스키마 구조는 1차 합의 사항이며, 실제 대상 설비가 확정되고 초기 구현을 진행하면서
> 세부 항목(태그 네이밍, 리샘플 주기 기본값 등)은 조정될 수 있다.

## 4. 캘리브레이션 & 추론 동작

**상태 전이**: 설비 config 등록 시 `CALIBRATING` 상태로 시작 → config의 기간/샘플 수 조건을 채우면 그
구간 데이터로 예측 모델을 학습하고 `MONITORING`으로 전환. 이후 실시간 슬라이딩 윈도우로 다음 시점 값을
예측하고, 실제값과의 오차를 캘리브레이션 구간에서 산출한 오차 분포(평균 + k·표준편차) 기준으로 정규화해
이상 점수를 계산한다.

**재캘리브레이션**: 설비 정비/부품 교체 등으로 "정상" 기준 자체가 바뀌는 경우를 위해, MQTT로 `recalibrate`
명령을 받으면 수동으로 `CALIBRATING` 상태로 되돌릴 수 있다. 자동 드리프트 적응(온라인 학습)은 채택하지
않는다 — 서서히 진행되는 실제 고장 징후를 "새 정상"으로 잘못 학습해버릴 위험이 있기 때문.

## 5. 에러 처리

- **MQTT 연결 끊김**: 재연결 재시도. 끊긴 동안의 데이터는 유실(별도 버퍼링 없음) — 재연결 후 새 데이터부터 재개.
- **결측 태그/구간**: 짧은 결측은 ffill. 리샘플 구간 내 데이터가 전혀 없으면 해당 스텝은 윈도우에서
  제외하고 로그를 남긴다.
- **캘리브레이션 데이터 부족**: `min_samples` 미달 시 `CALIBRATING` 상태를 유지하며 경고 로그를 남기고
  자동 연장한다.
- **모델 파일 손상/로드 실패**: `MONITORING` 진입을 막고 `CALIBRATING`으로 폴백, 재학습을 트리거한다.

## 6. 테스트 전략

도루코 CSV 데이터를 MQTT 스트림처럼 "리플레이"하는 시뮬레이터로 구독 → 버퍼링 → 캘리브레이션 → 추론 →
발행 전 구간을 엔드투엔드로 검증한다. 도루코 데이터엔 `IS_NG` 라벨이 있으므로 "이상 점수가 실제 불량 급증
시점과 상관관계가 있는지"를 참고 지표로 볼 수는 있으나, 이는 모델 성능 벤치마크가 아니라 **파이프라인
메커니즘 정확성**(리키지 없음, 윈도우 정합성, 캘리브레이션 전환 타이밍) 검증이 목적임을 명확히 한다 — 실제
대상 설비에는 라벨이 없다는 전제가 최종 설계의 기준이다.

## 7. 단계적 로드맵 (추가 AI 분석)

1차 구현 이후, 아래 순서로 단계적으로 확장한다. 순서는 "기존 모델/파이프라인 재사용도가 높은 것부터"
기준으로 정렬했다.

| 단계 | 기능 | 개요 | 비고 |
|---|---|---|---|
| Phase 1 | 핵심 이상탐지 엔진 | 1~6절에 기술된 센서/서보모터 태그 기반 LSTM/GRU 예측 + 이상 점수 | 최초 구현 대상 |
| Phase 2 | 기여도 분석 | 이상 점수 급등 시 어떤 태그가 가장 크게 벗어났는지 태그별 오차를 랭킹화 (`top_deviant_tag` 등) | Phase 1 모델의 태그별 오차를 그대로 재사용 — 추가 모델 불필요 |
| Phase 3 | 잔존수명(RUL) 추정 | 예측 오차의 시간적 추세를 외삽해 "언제쯤 임계치를 넘을 것으로 보이는지" 회귀 예측 | Phase 1 모델의 출력(오차 시계열)을 입력으로 하는 별도 회귀 컴포넌트 필요 |
| Phase 4 | 비전 기반 결함 탐지 | DX1의 IP카메라 + Event-triggered Video Logging Package와 연동해 Jetson GPU에서 CNN 기반 시각 결함 탐지 수행, 센서 이상탐지와 시점 연계 | 새로운 데이터 모달리티(영상) + 새 모델 타입 — 범위가 가장 큼. 별도 설계 필요 |

Phase 2~4는 각각 착수 시점에 별도 브레인스토밍/설계를 거친다 (특히 Phase 4는 카메라 하드웨어 구성과
DX1 Event-triggered Video Logging Package 연동 방식을 별도로 다뤄야 한다).

## 참고: DX1 관련 확인 사실

- DX1은 OMRON Data Flow Controller (DX100-0010), 내장 SW는 SpeeDBee Synapse.
- SpeeDBee Synapse는 MQTT Collector(Subscribe, JSON `"records"` 파싱)와 MQTT Emitter(Publish)를
  컴포넌트로 제공한다. DX1 자체는 MQTT 브로커가 아니라 클라이언트로 동작한다.
- JSON Serializer의 표준 출력 포맷: `{"records": [{"timestamp": ISO8601, "컴포넌트명:데이터명": 값, ...}]}`.
- Equipment/Factory Monitoring Package는 PLC가 이미 판정한 Abnormal 신호나 정적 Threshold를 집계해
  OEE 등 KPI를 만드는 규칙 기반 모니터링이며, 예측형 AI 기능은 없다 — 본 설계가 채우는 영역.
- DX1은 Condition Monitoring Package(모터 전류/진동/온도 진단 디바이스)도 지원하며, 해당 태그도 동일
  config 방식으로 프레임워크에 편입 가능하다.

## 미해결/추후 확정 필요 항목

- 실제 대상 설비 및 구체적 태그 목록 (미정 — 설비 확정 시 config 작성)
- 태그 샘플링 주기 및 동시 처리 설비 수 (Jetson 리소스 산정에 영향, 실측 필요)
- MQTT QoS/재연결 정책의 구체 파라미터 (SpeeDBee MQTT Emitter/Collector 설정과 맞춰 확정 필요)
