# 설비 이상 감지 앱 사용 안내서

이 문서는 **처음 쓰는 사람이 순서대로 따라 하도록** 쓴 안내서입니다. 개발자용 상세 설명은 `jetson_app/README.md`에 있습니다.

> **먼저 알아 두세요.** 지금까지의 시험은 모두 개발 PC에서, 설비에서 받은 데이터를 다시 재생하는 방식으로 했습니다. Jetson 실기에서 설치와 실행을 해 본 적은 아직 없습니다. 아래 7번의 "아직 안 되는 것"도 꼭 읽어 주세요.

---

## 1. 이 앱이 하는 일

설비가 **정상일 때의 움직임을 학습**해 두고, 이후 실시간 데이터가 정상 범위에서 벗어나면 **점수와 알람**을 내보냅니다.

```
설비 → DX1 → (MQTT) → Jetson 앱 → (MQTT) → 점수·알람
```

- **정상 데이터만 있으면 됩니다.** 이상 데이터는 필요 없습니다.
- 설비 한 대당 앱 하나를 실행합니다.
- 숫자로 된 값(위치, 속도, 토크, 켜짐/꺼짐 등)을 다룹니다.

### 사용 순서 한눈에 보기

| 단계 | 하는 일 | 대략 소요 (추정, 실제로 재 보지는 않음) |
|---|---|---|
| 1 | Jetson에 브로커와 앱을 설치합니다 (처음 한 번) | 30분 안팎 |
| 2 | DX1이 데이터를 Jetson으로 보내게 설정합니다 (처음 한 번) | 10분 |
| 3 | 설비 설정 파일을 만듭니다 (설비마다 한 번) | 30분 |
| 4 | 앱을 실행하고 설비를 **정상으로** 가동합니다 | 30~90분 |
| 5 | 학습 명령을 보냅니다 | 수십 분 |
| 6 | 점수와 알람을 확인합니다 | 상시 |

---

## 2. 용어

| 용어 | 뜻 |
|---|---|
| **태그** | DX1이 내보내는 값 하나. 예: `NX5_AxisData:AxX_Act_Pos` (`수집기이름:신호이름` 형식) |
| **토픽** | MQTT에서 메시지가 오가는 주소. 예: `dx1/AxisData` |
| **그룹** | 점수를 따로 내고 싶은 태그 묶음. 예: 서보 축 하나 |
| **점수 태그** | 이상 판단에 쓰는 태그. 그룹의 `tags`에 적습니다 |
| **입력 전용 태그** | 모델이 참고만 하고 점수에는 쓰지 않는 태그. `context_tags`나 그룹 밖의 `state_tag` |
| **상태 신호(`state_tag`)** | "지금 동작 중인가"를 알려 주는 0/1 신호. 동작 중과 대기 중의 정상 기준을 따로 만듭니다 |
| **점수** | 지금 값이 정상에서 얼마나 벗어났는지. 클수록 이상. 기본 3 이상이면 의심합니다 |
| **알람** | 점수가 기준(기본 3)을 **연속 3번** 넘으면 켜집니다 |
| `CALIBRATING` | 정상 데이터를 모으는 중인 상태 (앱을 처음 켠 상태) |
| `MONITORING` | 학습이 끝나 점수를 내는 상태 |

---

## 3. 준비물

- Jetson (JetPack 5.x) 또는 앱을 돌릴 PC
- DX1(SpeeDBee Synapse)과 같은 네트워크
- 이 저장소: `https://github.com/Hyunkeun98/JetsonProject`
- 감지할 설비의 **태그 목록**과 **토픽 이름**

---

## 4. 단계별 진행

### 단계 1. Jetson 설치 (처음 한 번)

1. 브로커(Mosquitto)를 설치합니다.
   ```bash
   sudo apt update
   sudo apt install -y mosquitto mosquitto-clients
   ```
2. 외부(DX1)에서 접속할 수 있게 설정합니다.
   ```bash
   sudo tee /etc/mosquitto/conf.d/jetson.conf > /dev/null <<'EOF'
   listener 1883 0.0.0.0
   allow_anonymous true
   EOF
   sudo systemctl restart mosquitto
   sudo systemctl enable mosquitto
   ```
   - 인증 없이 여는 설정이므로 **시험 네트워크에서만** 쓰세요.
3. 앱을 내려받고 설치합니다.
   ```bash
   git clone https://github.com/Hyunkeun98/JetsonProject.git
   cd JetsonProject/jetson_app
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   uv sync
   ```
4. Jetson의 IP를 확인해 적어 둡니다.
   ```bash
   hostname -I
   ```

**확인:** `systemctl status mosquitto`에 `active (running)`이 보이면 됩니다. 자세한 확인법과 방화벽 설정은 `jetson_app/README.md` 1장을 보세요.

### 단계 2. DX1이 데이터를 보내게 설정 (처음 한 번)

SpeeDBee Synapse에서 MQTT Emitter를 추가합니다.

- Broker Host: 단계 1에서 적어 둔 Jetson IP
- Port: `1883`
- Topic: 설비 데이터를 내보낼 토픽 (여러 개여도 됩니다)
- 데이터 형식: `{"records": [{"timestamp": "...", "수집기:신호": 값, ...}]}`

**확인:** Jetson에서 아래를 실행해 JSON이 흘러오면 됩니다 (`-C 1`은 한 건만 받고 끝냅니다).
```bash
mosquitto_sub -h localhost -t "dx1/AxisData" -C 1
```
이때 나온 JSON의 `"수집기:신호"` 글자가 곧 **태그 이름**입니다. 설정 파일에 그대로 적습니다.

### 단계 3. 설비 설정 파일 만들기 (설비마다 한 번)

`jetson_app/configs/` 아래에 새 파일(예: `my_machine.yaml`)을 만듭니다. 가장 쉬운 방법은 `configs/nx5_servo.yaml`을 복사해서 고치는 것입니다.

```yaml
equipment_id: my_machine              # 설비 이름(영문). 결과 토픽과 모델 파일 이름에 씁니다
mqtt:
  subscribe_topics:                   # DX1이 보내는 토픽을 모두 적습니다
  - dx1/AxisData
  - dx1/ProcStart
  - dx1/TaktTime                      # context_tags의 태그가 오는 토픽도 반드시 포함합니다
  publish_topic: jetson/my_machine/anomaly   # 점수와 알람이 나오는 토픽
  command_topic: jetson/my_machine/cmd       # 학습 명령을 보내는 토픽
groups:                               # 점수를 따로 내고 싶은 묶음
  axis_x:
    state_tag: "Collector_ProcStart:U0_ProcStart"   # (선택) 동작 중 신호. 점수에는 안 씁니다
    tags:                             # 이상 판단에 쓰는 태그
    - "Collector_AxisData:AxX_Act_Pos"
    - "Collector_AxisData:AxX_Act_Vel"
    - "Collector_AxisData:AxX_Act_Trq"
context_tags:                         # (선택) 모델이 참고만 하는 태그
- "Collector_TaktTime:U0_TaktTime"
resample_interval_ms: 100             # DX1 데이터 취득 주기(ms). 직접 맞춰 적습니다
max_lateness_ms: 500                  # 토픽 사이 도착 시간 차이를 기다리는 시간(ms)
window_size: 30                       # 모델이 보는 과거 길이(스텝 수)
calibration:
  max_duration: 7d                    # 정상 데이터를 보관하는 기간
  min_samples: 3000                   # 학습을 시작하려면 최소 이만큼(스텝) 필요
alarm:
  threshold: 3.0                      # 점수가 이 값을 넘으면 의심
  confirm_steps: 3                    # 연속 몇 번 넘으면 알람
training:
  max_samples: 45000                  # 학습에 쓰는 최근 데이터 수(스텝)
  epochs: 20
```

**무엇을 점수 태그로 정할까**
- **위치, 속도, 토크처럼 연속으로 변하는 값**이 이상 판단에 적합합니다.
- **켜짐/꺼짐 신호와 계단형 값**(공정 시작 신호, 사이클 시간, 센서)은 바뀌는 순간의 시각이 조금씩 달라 정상에서도 오차가 크게 나옵니다. 점수 태그로 넣으면 **오탐**이 많아지므로, `state_tag`나 `context_tags`로 쓰는 것을 권합니다.
- 어떤 태그를 점수로 볼지는 **설비를 아는 사람이 정합니다.** 앱이 자동으로 골라 주지는 않습니다.

**알아 둘 점**
- `resample_interval_ms`는 DX1의 실제 취득 주기와 같아야 합니다(100 ms 취득이면 `100`). 나중에 바꾸면 처음부터 다시 학습해야 합니다.
- `context_tags`나 `state_tag`에 적은 태그도 **토픽이 계속 들어와야** 합니다. 한 번도 안 들어오면 점수가 나오지 않습니다.
- 엑셀 목록(`DataList.xlsx`)에서 초안을 만드는 도구(`jetson-datalist`)도 있지만, 입력 전용 태그와 그룹 밖 상태 신호는 만들어 주지 않습니다. 이번 설정은 직접 쓰는 편이 빠릅니다.

**확인:** 설정 파일이 잘못되면 앱이 시작할 때 어디가 문제인지 알려 줍니다(단계 4에서 확인).

### 단계 4. 앱 실행과 정상 데이터 모으기

1. 앱을 실행합니다.
   ```bash
   cd JetsonProject/jetson_app
   uv run jetson-app --config configs/my_machine.yaml --host localhost
   ```
2. 시작하면 이런 줄이 나옵니다.
   ```
   [my_machine] 2개 토픽 구독 시작 (localhost:1883), 캘리브레이션 데이터: calibration_data, 모델 저장 위치: model_data
   ```
3. 설비를 **정상으로 가동**합니다. 앱이 데이터를 알아서 저장합니다(`calibration_data/my_machine.jsonl`).
4. 데이터가 흐르면 100스텝마다 이런 줄이 계속 나옵니다. **이 줄이 주기적으로 보이면 정상 동작 중**입니다.
   ```
   [snapshotter] 100번째 스냅샷 처리, 윈도우 30/30, 캘리브레이션 상태=CALIBRATING
   ```

**정상 데이터는 이렇게 모으세요**
- **설비가 실제로 운영되는 모든 모습**을 담습니다. 작업 투입 횟수가 다른 경우, 공정이 돌지 않는 **대기 시간**도 포함합니다.
- **이상이나 오동작이 섞이면 안 됩니다.** 에러가 났던 구간은 기준을 망가뜨립니다. 에러가 났다면 `recalibrate`(아래 5번 "운영 중 할 일")로 처음부터 다시 모으는 것이 안전합니다.
- 양은 설정의 `min_samples` 이상이어야 하고, **같은 사이클이 여러 번 반복될 만큼** 모으는 것을 권합니다. 아직 많이 시험하지는 못했지만 NX5 데모에서는 약 60분(약 39,000스텝)을 썼습니다.
- 100 ms 주기에서 스텝 수 = 시간(초) × 10 입니다. 예: 30분 = 18,000스텝.

### 단계 5. 학습

정상 데이터가 충분히 쌓였으면, **다른 터미널**에서 학습 명령을 보냅니다.

```bash
mosquitto_pub -h localhost -t "jetson/my_machine/cmd" -m '{"command": "train"}'
```

- 앱 터미널에 `학습 완료 — MONITORING 상태로 전환`이 나오면 끝입니다.
- **학습 중에는 앱이 멈춘 것처럼 조용합니다.** 정상입니다. 하트비트 줄도 나오지 않습니다.
- 걸린 시간(개발 PC CPU 기준): 약 19,000스텝 12분, 약 39,000스텝 32분. **Jetson에서는 더 걸릴 수 있고 아직 재 보지 못했습니다.**
- 명령에 `-r`(retain)을 **절대 붙이지 마세요.** 앱이 다시 연결될 때마다 학습 명령이 다시 실행됩니다.

### 단계 6. 점수와 알람 확인

학습이 끝나면 결과가 `publish_topic`으로 나옵니다.

```bash
mosquitto_sub -h localhost -t "jetson/my_machine/anomaly" -v
```

받는 메시지의 필드:

| 필드 | 뜻 |
|---|---|
| `timestamp` | 데이터 시각 |
| `jetson:anomaly_score` | 전체 점수 (그룹 점수 중 최댓값) |
| `jetson:alarm` | `true`면 알람 |
| `jetson:top_deviant_tag` | 점수를 가장 크게 올린 태그 |
| `jetson:<그룹>:score` / `:alarm` / `:top_tag` | 그룹별 점수·알람·원인 태그 (그룹이 둘 이상일 때) |

**읽는 법**
- 점수가 **0 근처**면 정상 범위입니다. **3 이상**이면 정상보다 많이 벗어난 것이고, 이 상태가 연속 3번 이어지면 알람이 켜집니다.
- 어느 그룹의 `alarm`이 `true`인지 보면 **어느 축·공정의 이상인지** 바로 알 수 있습니다.
- 점수를 받아서 화면에 띄우거나 설비를 멈추는 일은 **이 앱 밖의 일**입니다. MQTT로 이 토픽을 받는 프로그램(예: 표시 화면, PLC 연동)이 따로 필요합니다.

---

## 5. 운영 중 할 일

| 상황 | 할 일 |
|---|---|
| 앱을 다시 켜야 할 때 | 같은 설정으로 다시 실행하면 저장된 모델로 바로 `MONITORING`이 됩니다. 다시 학습하지 않아도 됩니다 |
| 설비 설정이 바뀌어 학습을 다시 해야 할 때 | `recalibrate`를 보내고 정상 데이터를 다시 모은 뒤 `train`을 보냅니다 |
| 설정 파일의 태그, 그룹, `context_tags`, `state_tag`, `resample_interval_ms`를 바꿨을 때 | 저장된 모델과 맞지 않아 앱이 `CALIBRATING`으로 돌아갑니다. 위와 똑같이 다시 학습합니다 |
| 오탐(정상인데 알람)이 많을 때 | `alarm.threshold`를 올리거나 `confirm_steps`를 늘립니다. 아래 팁을 보세요 |
| 이상을 놓칠 때 | `alarm.threshold`를 내립니다 |

`recalibrate` 명령:
```bash
mosquitto_pub -h localhost -t "jetson/my_machine/cmd" -m '{"command": "recalibrate"}'
```

**임계값 조정 팁**
- 정상 가동 중에 알람이 나면 그 시점의 점수와 `top_tag`를 기록해 두세요. 대기 중 알람이면 대기 기준이 너무 좁은 것일 수 있습니다.
- 임계값은 **정상 데이터에서 알람이 거의 안 나는 값**에서 시작해, **실제 이상을 일부러 만들어서 잡히는지** 확인하며 정하는 것이 가장 정확합니다. NX5 데모에서는 기본값 3에서 정상인데도 오탐이 많았고, 같은 정상 데이터에서 10쯤이면 오탐이 없었습니다. 다만 이상 데이터로 잡히는지는 아직 확인하지 못했습니다.

### 저장해 둔 파일로 평가하기 (임계값 추천)

설비를 다시 돌리지 않고, 이미 저장해 둔 DX1 JSON 파일(`{"records":[...]}`가 한 줄씩 든 `.json`)로 학습·평가·임계값 추천을 할 수 있습니다. 브로커와 DX1 설정은 필요 없습니다.

1. **구간 파일 `segments.yaml`을 만듭니다.** 시각에는 시간대(`+09:00`)를 꼭 적습니다.
   ```yaml
   train:        # 정상 데이터. 이 구간으로 학습합니다
     - {start: "2026-10-02T11:15:00+09:00", end: "2026-10-02T11:32:50+09:00"}
   normal_eval:  # 학습에 쓰지 않은 정상 구간. 오탐을 재는 데 씁니다
     - {start: "2026-10-02T12:11:00+09:00", end: "2026-10-02T12:40:00+09:00"}
   anomaly:      # 이상을 일부러 만든 구간(있으면). 여러 개 가능합니다
     - {name: servo_touch_1, start: "...", end: "..."}
   ```
   학습 구간과 검증/이상 구간은 겹치면 안 됩니다. 겹치면 오류가 납니다.
2. **학습합니다.** 모델이 `--model-dir`에 저장되고, 앱이 같은 폴더로 바로 `MONITORING`을 시작할 수 있습니다.
   ```bash
   uv run jetson-replay train --config configs/my_machine.yaml --data-dir 데이터폴더 --segments segments.yaml --model-dir model_data
   ```
3. **점수를 냅니다.**
   ```bash
   uv run jetson-replay score --config configs/my_machine.yaml --data-dir 데이터폴더 --segments segments.yaml --model-dir model_data --out scores.csv
   ```
4. **임계값을 추천받습니다.** 점수 파일만 읽으므로 몇 번이든 바로 실행됩니다.
   ```bash
   uv run jetson-threshold --scores scores.csv --config configs/my_machine.yaml
   ```
   - 임계값별 정상 알람 수, 이상 검출 수, 검출까지 걸린 시간이 표로 나오고 추천값이 나옵니다.
   - 이상 구간이 없으면 "정상에서 오탐이 0건인 최소값 × 1.2"를 추천하고 "민감도 미검증"이라고 알려 줍니다.
   - 이상 구간이 있으면 "오탐 0건 최소값"과 "모든 이상을 잡는 최대값"의 가운데를 추천합니다. 둘이 겹치지 않으면 추천하지 않고 어느 이상을 놓치는지 보여 줍니다.
   - 추천값은 제안입니다. 표를 보고 정해서 설정의 `alarm.threshold`에 적고 앱을 다시 시작하세요.

---

## 6. 문제가 생겼을 때

| 증상 | 확인할 것 |
|---|---|
| 시작할 때 오류가 난다 | 메시지에 설정의 어느 부분이 문제인지 나옵니다. 태그 중복, `context_tags`와 점수 태그 중복, 빈 목록 등을 확인합니다 |
| 구독 확인 줄만 나오고 하트비트가 안 나온다 | `mosquitto_sub -h localhost -t "<토픽>" -v`로 브로커에 메시지가 오는지 봅니다. 안 오면 DX1 설정이나 네트워크 문제입니다 |
| 하트비트는 나오는데 점수가 안 나온다 | 학습이 끝났는지(`MONITORING`) 확인합니다. 끝났는데도 안 나오면 `context_tags`나 `state_tag`에 적은 태그의 토픽이 안 들어오고 있을 가능성이 큽니다 |
| 학습 중에 연결이 끊겼다가 붙는다 | 현재 구조상 정상입니다. 학습 중 들어온 데이터는 유실됩니다 |
| `MQTT 연결 실패 (rc=...)` | 브로커 주소, 포트, 인증 설정을 확인합니다 |
| `[resampler: …] 누적 N건`이 계속 나온다 | 늦게 도착한 데이터가 버려지고 있습니다. `max_lateness_ms`를 늘려 보세요 |

---

## 7. 아직 안 되는 것 (알려 드립니다)

- **Jetson 실기 검증을 아직 못 했습니다.** PC에서만 시험했습니다.
- **GPU를 쓰지 않습니다.** CPU로 학습하고 추론합니다. Jetson에서 학습이 얼마나 걸리는지 아직 재 보지 못했습니다.
- **수집해 둔 JSON 파일로 학습하는 도구(`jetson-replay`)는 있지만, 앱에 실시간으로 넣는 방식은 아닙니다.** 파일 학습은 5번의 "저장해 둔 파일로 평가하기"를 보세요. 실기(Jetson)에서의 시간은 재 보지 못했습니다.
- **부팅할 때 앱이 자동으로 시작되지 않습니다.** 지금은 터미널에서 직접 실행합니다.
- **지금 상태(CALIBRATING인지, 몇 스텝 모았는지)를 묻는 명령이 없습니다.** 앱 터미널의 하트비트 줄로만 알 수 있습니다.
- **학습 중에는 앱이 멈춥니다.** 그 사이 들어온 데이터는 유실됩니다.
- **임계값이 확정되지 않았습니다.** `jetson-threshold`가 추천값을 내 주지만, 실제 이상 데이터로 잡히는지 확인하는 작업이 남아 있습니다.
- **어떤 태그를 점수로 볼지 자동으로 추천해 주지 않습니다.**
- 점수와 알람을 받아서 **화면에 보여 주거나 설비에 연결하는 부분은 이 앱에 없습니다.**

---

## 8. 명령 빠른 참조

```bash
# 앱 실행
uv run jetson-app --config configs/my_machine.yaml --host localhost

# 학습 / 재학습 시작 (retain 금지)
mosquitto_pub -h localhost -t "jetson/my_machine/cmd" -m '{"command": "train"}'
mosquitto_pub -h localhost -t "jetson/my_machine/cmd" -m '{"command": "recalibrate"}'

# 저장해 둔 파일로 학습/점수/임계값 추천
uv run jetson-replay train --config C --data-dir D --segments S --model-dir M
uv run jetson-replay score --config C --data-dir D --segments S --model-dir M --out scores.csv
uv run jetson-threshold --scores scores.csv --config C

# 결과 보기
mosquitto_sub -h localhost -t "jetson/my_machine/anomaly" -v

# DX1 데이터가 오는지 확인 (태그 이름 확인)
mosquitto_sub -h localhost -t "dx1/AxisData" -C 1
```

저장 위치: 정상 데이터 `calibration_data/<설비이름>.jsonl`, 학습된 모델 `model_data/<설비이름>.pt`.
