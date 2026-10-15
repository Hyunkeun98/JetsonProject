# jetson_app 개발자 안내

이 문서는 개발에 새로 참여하는 사람이 **무엇을 만들고 있는지, 어디까지 됐는지, 코드가 어떻게 나뉘어 있는지, 어떻게 개발하고 있는지**를 한 번에 파악하도록 쓴 문서입니다. 사용법은 `USER_GUIDE.md`, 설정/명령 레퍼런스는 `README.md`를 보세요.

## 1. 무엇을 만드는가

DX1(SpeeDBee Synapse)이 MQTT로 내보내는 JSON을 Jetson에서 받아 **설비 이상을 실시간으로 탐지**하는 프레임워크입니다.

- 정상 데이터만으로 학습한 GRU가 **다음 시점의 값을 예측**하고, 실제 값과의 오차를 정상 오차 분포에 대한 z-점수로 바꿔 **이상 점수**로 씁니다.
- 위치 신호의 순간 충격처럼 GRU가 못 잡는 이상은 **기준값 이탈 점수(`deviation`)** 가 맡습니다. 직전 2초의 중앙값에서 얼마나 벗어났는지를 대기 잡음 크기로 나눈 값이고, GRU 없이 단독으로도, GRU와 함께도 쓸 수 있습니다.
- 점수가 임계값 이상으로 연속 N번 나오면 알람을 확정하고, 결과(점수/알람/원인 태그)를 MQTT로 다시 발행합니다.
- 특정 설비에 묶여 있지 않고, 설정 YAML만 바꾸면 다른 DX1 설비에도 쓸 수 있게 만들었습니다.
- 현재 데모 대상은 **NX5 데모 라인**입니다. GRU는 서보 3축(AxX, AxZ, AxCV)의 Pos/Vel/Trq를 점수에 쓰고 ProcStart(공정 진행 여부)와 TaktTime은 입력으로만 씁니다. 충격 감지는 3축 Pos에 `deviation`을 적용합니다(`configs/nx5_pos_deviation.yaml`).

학습에는 이상 데이터를 쓰지 않습니다. 이상 데이터는 **임계값을 정하고 탐지 성능을 검증**하는 데만 씁니다.

## 2. 현재 진행 상황 (2026-10 기준)

| 영역 | 상태 |
|---|---|
| 실시간 파이프라인(수신→리샘플→예측→점수→알람→발행) | 구현 완료, 테스트 통과 |
| 캘리브레이션(정상 데이터 수집→학습→MONITORING), 재시작 후 이어하기 | 구현 완료 |
| 태그 그룹별 점수/알람, 동작(ON)·대기(OFF) 상태별 기준 | 구현 완료 |
| DataList 엑셀 → 설정 YAML 생성 도구 | 구현 완료 |
| 저장한 파일로 학습/점수 (`jetson-replay`) | 구현 완료 |
| 임계값 추천 (`jetson-threshold`) | 구현 완료 |
| 점수 그래프 (실시간 웹 `--web-port`, 저장 파일 `jetson-plot`) | 구현 완료 |
| 기준값 이탈 점수 `deviation` (위치 신호의 충격 감지, GRU와 별개) | 구현 완료, 앱·replay 연결, 테스트 통과, 푸시 완료 |
| 화면 `jetson-studio` (연결·살펴보기·전체 수집·태그 추천·학습·감시 6단계, Windows exe, 토큰으로 원격 조작) | 구현 완료, 테스트 통과, exe 실행 확인, 로컬 mosquitto로 실제 MQTT 경로 확인(run3 수집·학습, run4 감시 7/9). **DX1 실기 브로커(192.168.250.200)는 미확인** |
| 후보 태그 추천 `jetson-deviation-suggest` | 구현 완료 (NX5에서 Pos 3개만 추천 확인) |
| 태그 조합 선택 `jetson-select-tags`, 공정별 점수 `jetson-procscores` | 구현 완료 |
| 실제 MQTT 연결로 `--web-port` 동작 확인 | **미확인** |
| Jetson 실기에서 학습 시간 측정 | **미측정** (PC 기준: 19,350스텝 약 12.5분) |
| 임계값 확정 | **진행 중** (아래 데이터 현황 참고) |
| 장비 정지 상태(TaktTime 0) 처리 방침 | **미정** |
| 상위 시스템(PLC/표시) 연동 | 미착수 |
| 자동 시작/상태 확인 명령 | 미착수 |

테스트는 562개이고 모두 통과합니다. 새 도구(replay/threshold/dashboard)는 별도의 독립 코드 리뷰를 아직 받지 않았습니다.

## 3. 데이터 현황과 지금까지 알게 된 것

수집 데이터는 `JSONData/`에 있습니다. 이 중 **run3·run4의 원본 `DX_*.json`과 run4 `batches.txt`만 저장소에 올라가 있습니다**(약 72MB). 나머지(run1·run2, 실험 산출물, 모델)는 로컬에만 있습니다.

- **run1** (2026-10-02): 장비 정상 구동 데이터. 기계 에러로 중지된 구간(batches.txt에 표시)은 학습에서 제외했습니다.
  - 서보 3축 모델로 정상 평가 구간에서 임계값 3, 연속 3번일 때 정상 알람이 67건 나왔고 대부분 **대기 구간(ProcStart=0)** 에서 났습니다.
  - 정상 오탐이 0건이 되는 최소 임계값은 9.9였고 추천값은 11.9였습니다.
- **run2** (2026-10-08): 장비를 **완전히 정지**한 상태에서 서보 축에 에러를 준 데이터입니다.
  - 정지 상태에서는 TaktTime이 전부 0이고 AxCV 위치도 달라서, run1로만 학습한 모델에는 완전히 처음 보는 입력입니다(점수가 94/61/294로 고정되어 비교 불가).
  - 정지 구간만으로 학습한 모델로 보면 X축(약함)과 CV축 접촉은 정상과 구분되고, Z축은 신호 자체에 변화가 없었습니다.
  - 실제 흔들림은 batches.txt의 START보다 3~7초 앞서 나타납니다. 라벨은 "손을 댄 순간" 기준으로 적어야 합니다.
- **run3** (정상 10사이클)와 **run4** (7사이클, 대기 중인 축에 충격 9건):
  - 서보 속도·토크만 보는 GRU와 공정 단위 템플릿 비교는 충격을 한 건도 못 잡았습니다. 충격이 가장 뚜렷하게 나타나는 신호는 **Pos의 급한 이탈**이고, 속도·토크는 정상 변동에 묻힙니다.
  - 라벨(`batches.txt`)은 DX1 시각보다 약 6초 늦습니다(PC 시계 차이). 이를 모르고 라벨 구간만 비교해서 "신호가 없다"고 잘못 결론 낸 적이 있습니다. 평가 전에 원본 신호로 시각을 맞추세요.
  - Pos 기준 `deviation`(run3로 학습, run4로 검증): 충격 9건 중 7건 검출, 정상 구간 알람 0건. 못 찾은 2건은 Pos에 흔적이 없습니다(충격이 약했던 것으로 보입니다). 한계: 너무 큰 충격은 "움직임"으로 취급되어 점수 0, 움직이는 구간은 점수 없음, 정상 10사이클뿐이라 기준 여유가 얇음. 학습은 몇 초 걸립니다(GRU는 약 16분). 방식과 한계는 README의 "기준값 이탈 점수" 절.

**남은 핵심 결정:** 장비가 멈춰 있는 동안에도 서보를 감시할지 여부입니다. 감시한다면 정지 상태 정상 데이터를 따로 모아 학습해야 하고, 아니라면 정지 상태를 감시에서 제외해야 합니다.

## 4. 데이터가 흐르는 길

```
DX1 ──MQTT JSON──▶ mqtt_subscriber ──▶ EventTimeResampler ──▶ SnapshotProcessor
                                         (100ms 격자)            │
                      ┌──────────────────────────────────────────┤
                      ▼                                          ▼
              CALIBRATING: 정상 데이터 저장 ──▶ 학습      MONITORING: InferenceEngine
                                                           (예측 → 오차 → z-점수)
                                                                 │
                                                  Debouncer(연속 N번) ──▶ ResultPublisher ──MQTT──▶ 외부
                                                                 └──▶ ScoreHistory ──▶ 웹 대시보드
```

`deviation` 신호는 같은 스텝에서 GRU와 병렬로 계산되어 점수 발행에 합쳐집니다(`DeviationEngine`). 판정이 `settle_seconds`(기본 1초) 늦고, 움직이는 구간은 점수가 0입니다. 학습은 몇 초이며 결과는 `<설비>.deviation.json`로 저장됩니다.

- 시간 기준은 수신 시각이 아니라 **레코드의 이벤트 시각(timestamp)** 입니다. 지연·순서 뒤바뀜은 `max_lateness_ms`까지 허용하고, 끊긴 구간은 윈도우를 리셋합니다.
- 상태 태그(ProcStart)가 있는 그룹은 동작/대기 상태별로 따로 정상 오차 기준을 갖습니다.

## 5. 모듈 지도 (`src/jetson_app/`)

### 실시간 앱

| 파일 | 역할 |
|---|---|
| `subscriber_cli.py` | `jetson-app` 진입점. 설정을 읽고 파이프라인을 만들고 MQTT를 구독. `--web-port`로 웹 대시보드 시작 |
| `pipeline.py` | 모든 부품을 연결하는 조립 지점(`build_pipeline`) |
| `config.py` | 설정 YAML 로드와 검증(그룹, 알람, 학습, 캘리브레이션, `deviation` 설정 등). `tags`는 GRU 입력 태그, `all_tags`는 구독할 태그 전체 |
| `mqtt_subscriber.py` | MQTT 메시지 수신, DX1 JSON의 records 파싱 및 필터 |
| `timeparse.py` | DX1 ISO 8601 타임스탬프 → epoch 나노초 |
| `resampler.py` | 이벤트 시각 기준으로 값을 100ms 격자에 배치, watermark와 gap 처리 |
| `snapshot_processor.py` | 확정된 스텝을 큐로 받아 처리하는 워커(캘리브레이션 저장 또는 추론). GRU 그룹과 deviation 신호의 점수를 한 발행으로 합친다 |
| `buffer.py` | 고정 크기 슬라이딩 윈도우 |
| `calibration.py` | CALIBRATING/MONITORING 상태 관리, 상태 파일 영속화, 캘리브레이션 데이터 저장 |
| `command_subscriber.py` | MQTT 명령(train/recalibrate) 수신 |
| `droplog.py` | 버려진 데이터를 세어 로그 폭주를 막음 |

### 모델·학습

| 파일 | 역할 |
|---|---|
| `model.py` | GRU 모델(연속 태그는 회귀, 이진 태그는 분류 head) |
| `windowing.py` | 학습용 (윈도우, 정답) 쌍 생성 |
| `tag_stats.py` | 태그 종류 판별과 정규화 통계 |
| `training.py` | 학습, 정상 오차 통계(상태별), 모델 파일 저장/로드. `make_equipment_train_fn`은 설정에 맞게 GRU와 deviation 학습을 모두 수행 |
| `deviation.py` | 기준값 이탈 점수(torch 불필요): 스트리밍 계산기, 정상 데이터로 잡음·움직임 기준·알람 기준 자동 학습, 엔진, 학습 결과 JSON 저장/로드 |
| `inference.py` | 실시간 예측과 점수 계산, 학습 완료 시 모델 교체 |
| `debounce.py` | 점수가 임계값을 연속 N번 넘으면 알람 확정 |
| `publisher.py` | 점수/알람/원인 태그를 MQTT로 발행 |

### 오프라인 도구 (저장 파일로 분석)

| 파일 | 역할 |
|---|---|
| `segments.py` | 구간 파일(YAML): 학습/정상 평가/이상 구간을 시각으로 지정 |
| `replay.py`, `replay_cli.py` | `jetson-replay train/score`. 브로커 없이 저장한 JSON으로 학습·점수. **앱과 같은 부품을 그대로 사용** |
| `scores.py` | 점수 CSV 입출력 |
| `threshold.py`, `threshold_cli.py` | `jetson-threshold`. 임계값 후보별 정상 오탐/이상 검출/지연을 시뮬레이션하고 추천 |
| `datalist_to_config.py` | `jetson-datalist`. DataList 엑셀 → 설정 YAML 생성 |
| `deviation_suggest.py`, `deviation_suggest_cli.py` | `jetson-deviation-suggest`. 정상 데이터를 훑어 deviation에 맞는 태그를 이유와 함께 추천하고 설정 조각을 출력 |
| `tag_select.py`, `tag_select_cli.py` | `jetson-select-tags`. GRU 그룹의 점수 태그 조합 선택(`score_exclude_tags`) |
| `process_scores.py`, `process_scores_cli.py` | `jetson-procscores`. 사이클·공정별 점수 표 |

### 시각화

| 파일 | 역할 |
|---|---|
| `score_history.py` | 최근 점수를 메모리 링 버퍼에 보관(웹용) |
| `studio/controller.py` | `jetson-studio`의 제어 계층: 6단계 상태기계, 버튼별 허용 여부(`can`), 감시 파이프라인 시작/중지, 학습 스레드. 새 계산 로직 없이 기존 부품을 호출 |
| `studio/raw_source.py`, `studio/probe.py` | 브로커 구독 어댑터와 2단계 "살펴보기"(저장 없음) |
| `studio/collector.py` | 3단계 전체 수집: 원본 payload 저장, 크기 상한, archive, 폴더 불러오기, 중단된 수집본 표시 |
| `studio/cycles.py`, `studio/payloads.py` | 사이클 세기(0/1 신호·카운터·주기 추정). 태그 이름을 가정하지 않는다 |
| `studio/trainer.py` | 수집본으로 deviation 학습(파이프라인·torch 없이) |
| `studio/wizard.py` | 연결 정보 저장, 태그 분석 결과 정리, 선택 태그로 설정 YAML 생성·검증·저장 |
| `studio/server.py` | 화면과 JSON API(토큰, Host/Origin 검사, 본문 64KB 제한) |
| `studio/cli.py`, `web/studio.html` | `jetson-studio` 진입점과 단일 HTML 화면 |
| `engine_types.py`, `artifact_paths.py` | torch 없는 공용 타입과 경로 규칙(deviation 전용 구성이 torch 없이 import 되게 분리) |
| `web_server.py` | 읽기 전용 웹 서버(표준 라이브러리), 기본 127.0.0.1 |
| `dashboard_page.py`, `web/dashboard.html` | 외부 라이브러리 없는 단일 HTML 대시보드 |
| `plot_cli.py` | `jetson-plot`. 점수 CSV를 서버 없이 여는 HTML 한 장으로 변환 |

임계값 계산과 화면의 알람 계산은 **같은 규칙**이어야 합니다. 화면 쪽 JS(`BEGIN-CORE`~`END-CORE`)는 `tests/test_plot.py`에서 Python의 `Debouncer`와 결과를 비교합니다. 알람 규칙을 바꾸면 양쪽을 같이 고쳐야 합니다.

## 6. 설계에서 지켜야 하는 원칙

1. **오프라인 도구와 앱이 같은 부품을 쓴다.** replay가 앱과 다르게 계산하면 임계값 추천을 믿을 수 없습니다. 새 로직은 앱 부품을 재사용하세요.
2. **학습에는 정상 데이터만 쓴다.** 이상 데이터는 검증용입니다.
3. **구간은 시각으로 나눈다.** 정상/이상은 사이클이 아니라 시간 구간 단위이고, 구간 사이는 이어 붙이지 않습니다(윈도우 리셋).
4. **점수 대상과 입력 전용 태그를 구분한다.** `groups`의 tags만 점수에 쓰고, `context_tags`와 그룹 밖 상태 태그는 입력으로만 씁니다.
5. **정상 점수 기준은 학습 데이터에서 자동으로 정한다.** `deviation`의 잡음·움직임 기준·알람 기준은 사람이 숫자를 넣지 않고 정상 데이터로 정합니다(다른 설비에서 Claude 없이도 쓸 수 있어야 하기 때문). 새 설비는 `jetson-deviation-suggest`로 태그를 고릅니다.

## 7. 개발 방식

- **순서:** 설계 문서(`docs/superpowers/specs/`) → 구현 계획(`plans/`) → 구현. 큰 기능은 spec부터 씁니다.
- **테스트:** pytest. 새 기능에는 테스트를 같이 둡니다. 실행은 `uv run pytest`.
- **환경:** Python 3.8, `uv`, torch<2.4(CPU), paho-mqtt<2. 패키지는 uv로 관리합니다.
- **커밋:** 작은 단위로 `feat:`/`docs:`/`fix:`/`test:` 접두어를 씁니다. 기능 → 문서 → 푸시 순으로 갑니다.
- **데이터 파일:** run3·run4 원본을 제외한 `JSONData/`, `DataList.xlsx`, `models/`, 개인 실험 스크립트(`experiments/`, `Dorco_*.py`), 그림 파일은 저장소에 올리지 않습니다. 새 데이터를 올릴 때는 사용자 확인을 먼저 받으세요.
- **명령 요약:** `jetson-app`, `jetson-datalist`, `jetson-replay`, `jetson-threshold`, `jetson-plot`, `jetson-studio`, `jetson-deviation-suggest`, `jetson-select-tags`, `jetson-procscores`. 자세한 옵션은 `README.md`의 명령 레퍼런스를 보세요.

## 8. 앞으로 할 일 (우선순위 순)

1. **(라벨 시각)** `batches.txt`는 DX1 시각보다 약 6초 늦으니 평가 때 맞춰 쓰고, 가능하면 라벨 기록 방식을 바로잡기.
2. 더 많은 정상 사이클로 `deviation` 알람 기준 확정, 동작 중인 축에 가해진 이상 감지 방법 검토(지금은 움직이는 구간을 점수에서 뺍니다), 더 큰 충격이 "움직임"으로 취급되어 점수 0이 되는 한계 보완.
3. 장비 정지 상태 처리 방침 결정과 반영(정지 상태 학습 또는 감시 제외).
4. 서보 오탐이 몰린 대기 구간의 기준 폭(`_floor_std`) 점검.
5. 실제 MQTT로 `--web-port` 동작 확인, 좁은 화면 레이아웃 확인.
6. Jetson 실기에서 학습 시간 측정.
7. replay/threshold/dashboard 독립 코드 리뷰.
8. studio 확장: GRU 설정 화면, 재생 검증 탭, 수집 중 이상 구간 제외, 브로커 인증 입력, 수집 시간·사이클 수 도달 시 자동 종료.
9. 자동 시작/상태 확인 명령, PLC/표시 연동.
10. studio 6단계 흐름에서 코드 리뷰 때 보류한 것: (a) 주기 추정(D)이 수집 중 멈춘 시간까지 사이클로 세서 과대 추정할 수 있고, 먼 미래의 잘못된 타임스탬프 한 건이 추정 구간을 비운다. (b) 중간에 한 번 리셋되는 카운터는 후보에서 빠진다. (c) 거의 안 바뀌는 0/1 비트(문, 알람 리셋)가 2번 바뀌면 실제 사이클 신호보다 먼저 뽑혀 잘못된 경고가 날 수 있다(적게 세는 쪽이라 안전한 방향). (d) 시험(살펴보기)의 5분 자동 종료는 화면이 떠 있을 때만 동작한다. (e) 감시 파이프라인이 `command_topic`을 구독하므로 브로커로 재학습 명령이 오면 수집본으로 학습한 모델을 덮어쓴다(기존 동작). (f) 1GB급 수집본의 "태그 분석"은 요청 하나가 끝날 때까지 수 분 걸릴 수 있다.

## 9. 처음 시작하는 법

1. `README.md`로 개요와 설정 형식을 읽습니다.
2. `uv run pytest`로 환경이 맞는지 확인합니다.
3. `USER_GUIDE.md`를 따라 `configs/nx5_servo.yaml`과 샘플 데이터로 replay → threshold → plot을 한 번 돌려 봅니다. 앱 전체 흐름이 한눈에 보입니다.
4. 충격 감지(`deviation`)는 README의 "기준값 이탈 점수" 절을 읽고 `configs/nx5_pos_deviation.yaml`로 `JSONData/run3`(학습)·`run4`(검증)를 replay해 보면 7/9 검출을 재현할 수 있습니다.
5. 코드는 `pipeline.py`에서 시작해 `snapshot_processor.py`, `inference.py` 순으로 따라가면 실시간 흐름이 이해됩니다.
