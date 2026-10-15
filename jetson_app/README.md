# jetson_app

Jetson 쪽 실시간 이상탐지 프레임워크의 통신 + 데이터 파이프라인 레이어. DX1(SpeeDBee Synapse)이 MQTT로 Publish하는 설비 태그 데이터를 여러 토픽에서 구독해 각 record의 `timestamp`(이벤트 시각)를 기준으로 `resample_interval_ms` 격자에 배치(이벤트 시간 리샘플러)하고, 확정된 칸을 슬라이딩 윈도우와 캘리브레이션 버퍼에 쌓는다. MQTT 명령으로 학습(train)/재캘리브레이션(recalibrate) 상태 전이를 제어한다.

**처음 쓰는 사람은 [`../docs/USER_GUIDE.md`](../docs/USER_GUIDE.md)(사용 안내서)부터 보세요.** 아래는 개발자용 상세 설명입니다.

전체 설계 배경은 [`../docs/superpowers/specs/2026-07-31-jetson-dx1-anomaly-framework-design.md`](../docs/superpowers/specs/2026-07-31-jetson-dx1-anomaly-framework-design.md), 이 통신 레이어의 구현 계획은 [`../docs/superpowers/plans/2026-08-03-jetson-mqtt-communication-layer.md`](../docs/superpowers/plans/2026-08-03-jetson-mqtt-communication-layer.md) 참고.

현재 범위: 설비 config 로더(다중 토픽) + MQTT 파싱/구독자 + 이벤트 시간 리샘플러/슬라이딩 윈도우 + 스냅샷 처리 스레드 + 캘리브레이션 저장/상태머신 + MQTT train/recalibrate 명령 구독자 + 설비 통합 PyTorch GRU 모델 학습(태그 타입별 손실, 정규화/오차 통계 저장/로드) + 실시간 이상 점수 계산/디바운스/Result Publisher + CLI 진입점(코드, 유닛테스트 완료).

## 필요 환경

- **Jetson**: JetPack 5.x (Ubuntu 20.04, aarch64)
- **개발 PC**: uv 설치된 환경 (이 저장소는 여기서 코드를 작성/테스트함)

아래 가이드는 실제 Jetson 하드웨어와 DX1이 있어야 실행 가능한, 사람이 직접 수행하는 단계다.

## 1. Jetson에 MQTT 브로커(Mosquitto) 설치

Jetson 실기의 터미널에서 직접 실행한다.

### 1-1. 설치

```bash
sudo apt update
sudo apt install -y mosquitto mosquitto-clients
```

### 1-2. 외부(DX1) 접속을 허용하는 리스너 설정

기본 Mosquitto는 익명 접속을 막아둘 수 있다. 테스트 단계이므로 우선 무인증으로 열고, 실제 운영 전환 시 별도로 인증을 추가한다.

```bash
sudo tee /etc/mosquitto/conf.d/jetson.conf > /dev/null <<'EOF'
listener 1883 0.0.0.0
allow_anonymous true
EOF
```

### 1-3. 재시작 및 부팅 시 자동 시작

```bash
sudo systemctl restart mosquitto
sudo systemctl enable mosquitto
sudo systemctl status mosquitto --no-pager
```

`active (running)`이 나오면 정상.

### 1-4. 방화벽이 활성화되어 있다면 1883 포트 개방

```bash
sudo ufw status
```

`Status: active`인 경우에만:

```bash
sudo ufw allow 1883/tcp
```

### 1-5. 로컬 루프백으로 브로커 동작 검증

터미널 두 개를 열고, 하나에는 구독:

```bash
mosquitto_sub -h localhost -t "test/ping"
```

다른 하나에는 발행:

```bash
mosquitto_pub -h localhost -t "test/ping" -m "hello"
```

구독 터미널에 `hello`가 출력되면 정상.

### 1-6. Jetson의 IP 주소 확인 (아래 DX1 설정과 CLI 실행에 필요)

```bash
hostname -I
```

## 2. Jetson에 jetson_app 배포

### 2-1. 저장소 clone/pull

```bash
git clone https://github.com/Hyunkeun98/JetsonProject.git
cd JetsonProject/jetson_app
```

이미 clone되어 있다면:

```bash
cd JetsonProject
git pull
cd jetson_app
```

### 2-2. Jetson에 uv 설치 (없는 경우)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2-3. 의존성 설치

```bash
uv sync
```

### 2-4. 설비 config 준비

```bash
cp configs/test_dx1.example.yaml configs/test_dx1.yaml
```

`configs/test_dx1.yaml`의 `tags` 목록을, 아래에서 DX1에 등록할 실제 태그명과 동일하게 수정한다.

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

## 3. DX1의 SpeeDBee Synapse에서 MQTT Emitter 설정

SpeeDBee Synapse 관리 화면에서:

1. 기존에 구성된 Collector(PLC/센서 등)의 출력을 JSON Serializer에 연결한다. 출력 포맷은 `{"records": [{"timestamp": ISO8601, "태그명": 값, ...}]}` 이어야 한다.
2. MQTT Emitter 컴포넌트를 추가한다:
   - Broker Host: 위 1-6에서 확인한 Jetson IP
   - Port: `1883`
   - Topic: `dx1/test_dx1/telemetry` (이 Emitter가 발행할 토픽 하나가 `configs/test_dx1.yaml`의 `mqtt.subscribe_topics`(복수) 목록에 정확히 포함되어 있어야 함)
   - Publish 트리거 활성화

## 4. 구독자 실행 및 End-to-End 검증

Jetson에서:

```bash
uv run jetson-app --config configs/test_dx1.yaml --host localhost
```

(또는 동일하게 `uv run python -m jetson_app.subscriber_cli --config configs/test_dx1.yaml --host localhost`)

시작 시 `[test_dx1] 1개 토픽 구독 시작 (localhost:1883), 캘리브레이션 데이터: calibration_data, 모델 저장 위치: model_data` 형태의 구독 확인 줄이 출력된다. 이후에는 메시지마다 출력되지 않고, 데이터가 실제로 흐르고 있으면 스냅샷 100개마다(격자가 50ms면 약 5초에 한 번) 다음과 같은 하트비트 줄이 찍힌다:

```
[snapshotter] 100번째 스냅샷 처리, 윈도우 10/10, 캘리브레이션 상태=CALIBRATING
```

즉 **하트비트 줄이 주기적으로 보이면 정상 동작 중**이고, 구독 확인 줄만 나오고 하트비트가 전혀 안 나오면 아직 태그 값이 하나도 안 들어온 것이다(또는 `max_lateness_ms`만큼 기다리는 중이다)(아래 점검 순서 참고). `Ctrl+C`로 종료할 수 있다.

### 캘리브레이션 데이터 위치와 train/recalibrate 명령

- `--calibration-dir` 플래그로 캘리브레이션 버퍼 저장 위치를 지정한다(기본: `calibration_data`). 설비마다 `<calibration-dir>/<equipment_id>.jsonl` 파일에 스냅샷이 JSON Lines로 쌓인다.
- 기동 직후 상태는 `CALIBRATING`이며, 정상 데이터가 충분히 쌓이면(`calibration.min_samples` 이상) 아래처럼 `train` 명령을 보내 `MONITORING`으로 전환한다. 다시 캘리브레이션부터 시작하려면 `recalibrate`를 보낸다.

```bash
mosquitto_pub -h localhost -t "jetson/test_dx1/cmd" -m '{"command": "train"}'
mosquitto_pub -h localhost -t "jetson/test_dx1/cmd" -m '{"command": "recalibrate"}'
```

- `--model-dir` 플래그로 학습된 모델을 저장할 위치를 지정한다(기본: `model_data`). `train` 명령이 성공하면 `<model-dir>/<equipment_id>.pt`에 GRU 가중치 + 정규화 통계 + 태그 타입 + 오차 분포 통계가 함께 저장된다.
- `train` 명령은 이제 실제로 PyTorch GRU 모델을 학습한다. **학습은 수 분 단위로 걸린다** — 예제 config 규모(`min_samples: 10000`, `window_size: 100`, 기본 `hidden_size=64`, `num_layers=2`, `epochs=20`)에서 x86 개발 PC CPU 기준 약 **9분**이 걸렸고, Jetson CPU에서는 더 느리다. 캘리브레이션 버퍼가 아무리 커도 학습에는 가장 최근 20,000 샘플만 사용한다(`train_model`/`make_train_fn`의 `max_training_samples`) — 7일치 버퍼를 통째로 윈도잉하면 메모리가 GB 단위로 튀어 Jetson에서 OOM이 나기 때문이다.

  이 학습은 현재 **MQTT 네트워크 스레드 위에서 동기로** 돈다. 그래서 학습이 도는 수 분 동안:

  - `MqttRecordSubscriber`가 `loop_forever()`를 돌리던 그 스레드가 막히므로, paho 기본 keepalive(60초)를 넘겨 브로커가 연결을 끊는다. 학습이 끝나면 자동으로 재접속 + 재구독되어 스스로 복구되지만, 끊긴 동안 DX1이 발행한 텔레메트리는 QoS 0이라 그대로 유실된다.
  - `CalibrationManager.handle_train_command`가 학습 내내 락을 잡고 있어서 백그라운드 스냅샷 스레드도 함께 멈춘다 — 하트비트 줄도, 윈도우 갱신도 그동안 나오지 않는다.

  즉 학습 중 콘솔이 조용하고 잠깐 연결이 끊겼다 붙는 것은 **현재 구조상 정상 동작**이다. 학습을 네트워크 스레드 밖(별도 워커 스레드/프로세스)으로 빼는 것은 다음 계획의 범위다.

> 주의: 이 명령들에는 절대 `-r`(retain)을 붙이지 않는다. retain된 명령 메시지는 앱이 재접속할 때마다 다시 전달되어 의도치 않게 재실행된다.

### 취득 주기와 `resample_interval_ms`, `max_lateness_ms`

- DX1은 Collector 주기(예: 100ms)로 취득한 값을 여러 개 묶어 한 MQTT 메시지로 보낼 수 있다. Jetson은 도착 시각이 아니라 **각 record의 `timestamp`** 로 값을 배치하므로, 메시지에 record가 몇 개 들어 있든 시점이 보존된다.
- `resample_interval_ms`는 모델이 보는 시간 간격이다. **DX1 취득 주기에 맞춰 config에 직접 적는다**(예: 취득이 100ms면 `100`, 더 줄이면 같이 줄인다). 격자보다 빠른 신호는 한 칸 안의 마지막 값을, 느린 신호는 직전 값을 유지(ffill)한다.
- `max_lateness_ms`(기본 2000)는 토픽 간 도착 시차를 기다리는 시간이다. **토픽 중 가장 긴 배치 간격보다 크게** 잡는다. 너무 작으면 늦게 도착한 record가 `[resampler: …] 누적 N건` 로그와 함께 폐기되고, 클수록 결과 발행이 그만큼 늦어진다.
- 학습 모델에 격자 값이 함께 저장된다. **`resample_interval_ms`를 바꾸면** 저장된 모델과 호환되지 않아 재시작 시 `CALIBRATING`으로 폴백한다. 캘리브레이션 버퍼도 이전 간격의 데이터와 섞이므로, 격자를 바꿀 때는 반드시 `recalibrate` 명령을 보내 버퍼를 비우고 다시 시작한다.
- DX1/PLC의 시계가 크게 앞서 가면(윈도우 길이보다 긴 공백) 슬라이딩 윈도우를 비우고 새 시점부터 다시 쌓는다. 반대로 시계가 뒤로 보정되어 이전 시각보다 크게 뒤처진 record가 10개 연속으로 오면, 점프한 시간만큼 데이터가 끊기지 않도록 새 시각을 기준으로 재동기화한다(한두 개의 잘못된 타임스탬프는 그냥 폐기한다).
- 캘리브레이션 버퍼의 오래된 데이터 정리(`calibration.max_duration`)는 Jetson 시계가 아니라 DX1 이벤트 시각을 기준으로 하므로, 두 시계가 달라도 버퍼가 지워지지 않는다.

### 그룹과 상태별 기준 (공정별 점수)

config에서 `tags:` 대신 `groups:`를 쓰면 태그를 그룹으로 묶을 수 있다(둘을 함께 쓸 수는 없다). 그룹마다 이상 점수와 알람이 따로 계산되므로 어느 부분의 이상인지 바로 알 수 있다.

```yaml
groups:
  process_4:
    state_tag: "NX5_ProcStart:U4_ProcStart"   # 선택: 이 그룹의 동작 중 신호(0/1)
    tags: ["NX5_ProcStart:U4_ProcStart", "NX5_AxisData:AxZ_Act_Trq", "NX5_TaktTime:U4_TaktTime"]
  general:                                       # 상태 태그가 없는 그룹
    tags: ["NX5_SenData:ConvSensor0"]
alarm: {threshold: 3.0, confirm_steps: 3}        # 선택(기본값)
training: {max_samples: 60000, epochs: 20}       # 선택(기본 20000 / 20)
```

- `state_tag`가 있으면 그 그룹의 태그는 **동작 중 / 대기 중 상태별로 따로 학습한 정상 오차 기준**으로 판정한다. 그래서 공정이 쉬는 동안 서보에 생기는 작은 충격도 대기 상태의 잔잔한 기준과 비교되어 이상으로 드러난다. 한 태그는 하나의 그룹의 점수 대상(`tags`)에만 속한다. `state_tag`는 그 그룹의 `tags` 안(점수에도 반영)이나 밖(입력 전용, 점수 제외)에 둘 수 있다. 자세한 내용은 아래 "점수 대상 태그와 입력 전용 태그"를 본다.
- **정상 데이터에 동작 중과 대기 중이 모두 충분히 들어 있어야 한다**(각 상태 100스텝 이상). 부족한 상태는 전체 기준으로 대체되고 학습 로그에 `[train] 상태별 기준 표본 부족 -> 전체 기준으로 대체: …`가 출력된다.
- 발행 결과(`publish_topic`)에는 기존 필드(`jetson:anomaly_score`, `jetson:alarm`, `jetson:top_deviant_tag`)와 함께, 그룹이 둘 이상이면 그룹마다 `jetson:<그룹>:score`, `jetson:<그룹>:alarm`, `jetson:<그룹>:top_tag`가 들어간다. 전체 점수는 그룹 점수의 최댓값, 전체 알람은 그룹 알람 중 하나라도 참일 때다. 그룹 알람은 그룹마다 독립적으로 `alarm.confirm_steps`번 연속 `alarm.threshold`를 넘어야 확정된다.
- 학습 후 **그룹 구성(이름, 상태 태그, 태그)을 바꾸면** 저장된 모델과 맞지 않아 재시작 시 `CALIBRATING`으로 폴백한다. 이때는 `recalibrate`로 버퍼를 비우고 다시 학습한다.
- `training.max_samples`는 학습에 쓰는 가장 최근 스텝 수의 상한이다. 90분(100ms 격자 약 54,000스텝)을 모두 쓰려면 이 값을 올린다. 학습 시간과 메모리도 함께 늘어난다.

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

### 새 설비를 붙이는 순서

1. **받을 데이터를 엑셀에 적는다.** 열은 `Variables`, `MQTT Topic`(필수)과 `Description`, `Tag`, `Group`, `StateTag`(선택)이다. `Tag`는 DX1이 내보내는 정확한 태그 이름이고, 없으면 `<접두사>_<토픽 마지막 마디>:<변수명의 '.'를 '_'로>`로 만든다. `Group`이 비어 있으면 `general`, `StateTag`에 `Y`를 적으면 그 행이 그룹의 상태 태그다. `Group` 열이 없으면 `--group-regex`로 `Description`에서 그룹을 찾을 수 있다.
2. **설정 초안을 만든다**(개발 PC에서, 엑셀을 읽으려면 `uv sync --extra tools`가 필요하다):

   ```bash
   uv run jetson-datalist DataList.xlsx --equipment-id nx5 --collector-prefix NX5 \
     --group-regex '공정(\d+)' --group-template 'process_{}' --state-pattern 'U{}_ProcStart' \
     --samples ../JSONData --output configs/nx5.yaml
   ```

   - **태그는 엑셀에 적은 것만 사용한다.** 도구가 태그를 찾아 넣거나 빼지 않는다.
   - `--samples`(샘플 JSON 파일/폴더)를 주면 목록의 태그가 샘플에 실제로 있는지 확인하고(없으면 오류, `--allow-missing`이면 경고), 샘플 간격으로 `resample_interval_ms`를 권장해 기본값으로 넣고, 값이 변하지 않은 태그를 경고한다. 샘플에만 있는 태그는 참고로 출력만 한다.
3. **생성된 YAML을 확인하고 필요하면 고친다**(`resample_interval_ms`, `window_size`, `max_lateness_ms`, `alarm`, `training`).
4. **충격·외력 감지가 목적이면** 속도·토크를 GRU로 보는 것보다 위치 신호를 `deviation`으로 보는 쪽이 맞다. 아래 "기준값 이탈 점수" 절의 순서(`jetson-deviation-suggest`로 태그 확인 → 설정 → 학습 → 이상 데이터로 확인)를 따른다.
5. 앱을 실행하고(`CALIBRATING`), 정상 가동 데이터가 충분히 쌓이면 `train` 명령을 보내 `MONITORING`으로 전환한다. 정상 데이터는 그 장비의 실제 운영 방식(동작/대기 상태 포함)을 대표해야 한다.

### 문제 발생 시 점검 순서

1. **Jetson 쪽**: `mosquitto_sub -h localhost -t "dx1/test_dx1/telemetry" -v`로 브로커에 메시지가 도달하는지 먼저 확인한다.
   - 도달하면 → Jetson 코드/config 문제 (topic 오탈자, tags 불일치 등)
   - 도달 안 하면 → 네트워크/DX1 설정 문제
2. **DX1 쪽**: SpeeDBee의 MQTT Emitter 연결 상태/에러 로그, topic 오탈자를 확인한다.
3. **네트워크**: DX1과 Jetson이 같은 서브넷에 있는지, Jetson 방화벽(1-4)이 막고 있지 않은지 확인한다.
4. **인증 실패**: 브로커가 연결을 거부하면(`rc != 0`) 콘솔에 `MQTT 연결 실패 (rc=...)` 메시지가 출력된다 — 브로커 쪽 인증/ACL 설정(1-2)을 확인한다.

## jetson-studio: 화면에서 설정·수집·학습·감시

터미널과 YAML 편집 없이 브라우저 화면 하나로 "설정 → 정상 데이터 수집 → 학습 → 감시"를 끝낸다. 위치 신호의 충격을 감지하는 기준값 이탈 점수(`deviation`) 전용이다(GRU 설정은 기존 `jetson-app`을 쓴다).

```bash
uv run jetson-studio                      # 브라우저가 자동으로 열린다 (http://localhost:8800)
dist/jetson-studio/jetson-studio.exe     # Windows 패키지 (아래 "exe 만들기")
```

| 옵션 | 설명 |
|---|---|
| `--workspace PATH` | 설정, 수집 데이터, 모델을 저장할 폴더. 기본은 exe 옆 또는 현재 폴더의 `workspace/` |
| `--web-host` | 화면을 열 주소. 기본 `127.0.0.1`(이 장비에서만) |
| `--web-port` | 포트(기본 8800) |
| `--token` | 다른 PC에서 조작할 때 필요한 비밀번호 |
| `--no-browser` | 브라우저를 자동으로 열지 않는다 |

화면은 6단계다. 현재 상태에 맞는 단계가 강조되고, 앱을 껐다 켜도 저장된 상태에서 이어진다(학습한 모델이 있으면 6단계, 수집본만 있으면 4단계, 연결 정보만 있으면 2단계).

1. **연결**: 장비 이름, 브로커 주소·포트, 구독 토픽(쉼표로 여러 개)을 저장한다. 사이클 시작 신호나 카운터가 다른 토픽에 있으면 그 토픽도 같이 적는다(예: `dx1/AxisData, dx1/ProcStart`).
2. **살펴보기**: "시험 시작"으로 토픽을 임시로 구독해 데이터가 들어오는지 본다. 수신 건수, 초당 건수, 마지막 수신 시각, 마지막 메시지 원문, 숫자 태그 목록을 보여 주고 아무것도 저장하지 않는다(5분 뒤 자동 종료).
3. **전체 수집**: 정상으로 운전하는 동안 들어오는 데이터를 **원본 그대로** `workspace/collections/<장비>/current/`에 모은다("수집 종료"를 누를 때까지, 기본 상한 1GB). 태그를 고르기 전에 모으므로 나중에 태그를 바꿔도 다시 모을 필요가 없다. 수집 중 축이 움직이는 사이클이 들어가야 추천이 나온다(멈춘 상태만 모으면 추천이 비어 나온다). 새로 수집하면 이전 수집본은 `archive/`로 옮겨지고, 이전 학습 결과는 더 이상 감시에 쓰이지 않는다. 이미 저장해 둔 DX1 JSON 폴더를 "폴더 불러오기"로 수집본 대신 쓸 수도 있다.
4. **태그 추천**: 수집본으로 추천 태그를 이유와 함께 표로 보여 준다(`jetson-deviation-suggest`와 같은 분석, 추천은 미리 체크됨). 사이클 수도 여기서 확인한다.
5. **학습**: 고른 태그로 신호별 대기 잡음 σ, 알람 기준, 움직임 기준을 계산한다. 사이클이 **10회보다 적으면 경고만** 하고 학습은 막지 않는다. 값이 전혀 안 변하는 데이터(장비 정지)로는 학습이 실패하고 이유가 표시되며 수집본은 남는다.
6. **감시**: 점수 그래프와 알람을 본다. "다시 학습"은 같은 수집본으로 4단계로 돌아간다.

**사이클 세는 법:** 태그 이름을 가정하지 않고 수집본에서 찾는다. (A) 0/1 신호가 0→1로 바뀐 횟수, (B) 정수 카운터의 증가 횟수는 정확하고, 둘 다 없으면 (D) 연속 신호의 움직임이 되풀이되는 주기로 추정한다(여러 태그가 서로 맞아야 믿고, "추정"으로 표시한다). 그것도 안 되면 사이클 수를 알 수 없다고 하고 수집 시간만 보여 준다. 후보가 여러 개면 4단계에서 기준 신호를 고를 수 있다.

**다른 PC에서 조작(Jetson):**

```bash
uv run jetson-studio --web-host 0.0.0.0 --token 비밀번호 --no-browser
# 다른 PC 브라우저: http://<Jetson IP>:8800/?token=비밀번호
```

루프백(127.0.0.1)이 아닌 주소로 열 때 `--token`이 없으면 조작 기능이 꺼지고 점수만 볼 수 있다(읽기 전용). 모든 조작 요청은 토큰과 출처(Host/Origin)를 확인하고, 파일은 작업 폴더 안에만 쓴다. 부팅 때 자동으로 띄우려면 systemd 서비스로 등록한다.

```ini
[Service]
WorkingDirectory=/home/jetson/jetson_app
ExecStart=/home/jetson/.local/bin/uv run jetson-studio --web-host 0.0.0.0 --token CHANGE_ME --no-browser
Restart=on-failure
```

**exe 만들기(Windows):** `powershell -ExecutionPolicy Bypass -File packaging/build_exe.ps1` → `dist/jetson-studio/jetson-studio.exe`(약 15MB, torch 미포함). 폴더째 복사해서 쓴다.

**지금의 한계:** `deviation` 전용이다. 수집 중 이상 구간을 시간으로 골라 빼는 기능은 없어서 섞였다면 지우고 다시 모은다. 브로커 아이디·비밀번호 입력칸이 없다. 주기 추정(D)은 불규칙한 사이클에서 틀릴 수 있다. 재생으로 확인했을 때(run3 수집 → run4 감시) 기존 실험과 같은 9건 중 7건 검출이었고, 약한 충격 2건은 놓친다.

## 저장해 둔 파일로 학습·점수·임계값 추천

MQTT 브로커 없이, 수집해 둔 DX1 JSON 파일(한 줄이 `{"records":[...]}`인 `.json`/`.jsonl`)로 앱과 같은 부품(리샘플러, 윈도우, `InferenceEngine`, 학습 함수)을 돌린다. 운영 경로는 바뀌지 않는다. 단계별 설명은 [사용 안내서](../docs/USER_GUIDE.md)의 "저장해 둔 파일로 평가하기"를 본다. 설계는 `../docs/superpowers/specs/2026-10-07-replay-and-threshold-design.md`.

```bash
uv run jetson-replay train --config configs/nx5_servo.yaml --data-dir ../JSONData/run1 --segments segments.yaml --model-dir model_data
uv run jetson-replay score --config configs/nx5_servo.yaml --data-dir ../JSONData/run1 --segments segments.yaml --model-dir model_data --out scores.csv
uv run jetson-threshold --scores scores.csv --config configs/nx5_servo.yaml --confirm 3,5,10
```

- **구간 파일**: `train` / `normal_eval` / `anomaly`(이름 선택) 목록. 시각은 시간대가 필수(`2026-10-02T11:15:00+09:00`)이고 `[start, end)`이다. 서로 겹치면 오류다(학습과 검증이 겹치면 오탐이 적게 나와 임계값이 낮게 추천된다).
- **재생 규칙**: 구간마다 리샘플러와 윈도우를 새로 시작한다. 데이터 폴더의 모든 `.json`/`.jsonl`을 읽고 토픽은 보지 않으며 설정의 태그만 쓴다. `train`은 모델과 함께 상태 마커를 `MONITORING`으로 써서 앱이 같은 `--model-dir`로 바로 이어 쓸 수 있다.
- **점수 CSV**: 스텝마다 그룹별 점수/원인 태그와 윈도우 리셋 표시. 앱 처리기와 같은 점수를 낸다(테스트 및 run1 10,166행 일치 확인).
- **임계값 계산**: 알람은 그룹마다 "점수 ≥ 임계값이 `confirm`번 연속", 전체는 OR이다. 길이 `confirm`인 창의 최솟값 중 최댓값(임계 점수 c)을 쓰면 임계값 T에서 알람이 나는 조건이 `T ≤ c`이다. 정상 구간의 c로 오탐 0건 최소값(T0), 이상 구간의 c로 모두 검출하는 최대값(T1)을 얻고, 후보 표는 앱의 디바운서로 시뮬레이션한다. 추천: 이상 없음 → `T0×1.2`(`--margin`, 또는 `--max-alarms-per-day N`), 이상 있음 → `T0 ≤ T1`이면 가운데, 아니면 추천 없음. 최소 추천값은 2.0, 이상 구간이 3개 미만이면 과적합 경고를 낸다.

### 점수에 쓸 태그 조합 고르기 (`score_exclude_tags`)

그룹 점수는 태그별 점수(z)의 최댓값이다. 한 태그가 정상에서도 크게 튀면(예: 대기 중 Pos의 미세한 흔들림) 그 값이 그룹 점수를 끌어올린다. 이때 **모델을 다시 학습하지 않고** 설정에 `score_exclude_tags: [태그, ...]`를 적어 그 태그를 점수에서만 뺄 수 있다(모델 입력과 학습은 그대로이므로 저장된 모델이 그대로 맞는다. 그룹마다 점수 태그가 하나는 남아야 한다).

- `jetson-replay score`가 점수 CSV에 `z:<태그>` 열(점수에서 뺀 태그 포함)을 같이 쓴다.
- `uv run jetson-select-tags --scores scores.csv --config cfg.yaml [--segments seg.yaml] [--confirm 3]`: 그룹마다 태그 조합(공집합 제외)을 모두 평가해, 정상 구간 태그별 단독 최대 점수와 상위 조합을 보여 준다. 이상 구간이 있으면 "정상 최대보다 높게 나온 이상 수 → 간격(이상 최소 − 정상 최대) → 태그 수" 순으로 고르고 `score_exclude_tags` 값을 출력한다. 어떤 조합도 이상을 못 가르면 제안하지 않는다.
- 구간 파일의 이상 구간에 `groups: [axis_x]`를 적으면 그 그룹의 점수로만 평가한다(없으면 모든 그룹에 배정). 정상 데이터만 있을 때는 단독 최대 점수 표를 보고 사람이 정한다.
- 이상 구간이 3개 미만이면 과적합 경고를 낸다. 고른 조합은 다음 수집 데이터로 다시 확인한다.

### 사이클·공정별 점수 (`jetson-procscores`)

`uv run jetson-procscores --scores scores.csv --data-dir DATA [--confirm 3] [--pad-s 0] [--processes 11] [--proc-pattern "NX5_ProcStart:U{}_ProcStart"] [--out process_scores.csv]`. 공정 0의 시작 신호가 켜질 때마다 새 사이클로 보고, 공정 n의 신호가 켜졌다 꺼진 구간의 점수를 사이클 x 공정 표(전체/그룹별 최대 점수)와 CSV(그룹별 최대·c·원인 태그, 겹친 이상 구간)로 낸다. 이상 구간과 겹친 칸에는 `*`가 붙고, 구간에 점수 행이 없으면 `-`이다.

### 기준값 이탈 점수 (`deviation`): 멈춘 축에 가해진 충격·외력 감지

GRU(다음 값 예측)와 **별개의** 점수 방식이다. 평소엔 거의 일정한 신호(예: 서 있는 서보 축의 위치 `Pos`)가 외력으로 잠깐 벗어나는 것을 잡는다. 점수는 `|현재 값 − 직전 2초 값의 중앙값| ÷ 대기 잡음`이다. GRU 없이 이 방식만 쓸 수도 있고(`tags`/`groups` 없이 `deviation:`만), GRU 그룹과 같이 쓸 수도 있다. 학습은 정상 데이터에서 몇 초 만에 끝나고 torch가 필요 없다.

**NX5 데모(서보 3축, 충격 9건)에서의 결과**: run3(정상 10사이클)로 학습하고 run4로 검증해 충격 7건을 검출하고 정상 구간 알람은 0건이었다. 못 찾은 2건은 Pos에 흔적이 없었다(약하게 친 것으로 보임). 같은 데이터에서 GRU(Vel/Trq)는 0건이었다. 충격은 Pos의 급한 이탈로 가장 뚜렷하게 나타난다. 설정 예시는 `configs/nx5_pos_deviation.yaml`.

#### 다른 장비에 쓰는 순서 (Claude 없이)

1. **정상 운전 데이터를 모은다.** 충격·이상을 주지 않은 정상 가동 데이터(DX1 JSON)를 폴더에 모은다. 장비가 하는 모든 동작(이동, 대기)이 들어 있어야 한다. 10사이클 이상을 권한다(적을수록 알람 기준의 여유가 얇다).
2. **어느 태그를 쓸지 도구로 확인한다.**

   ```bash
   uv run jetson-deviation-suggest --data-dir 정상데이터폴더
   ```

   모든 숫자 태그의 대기 잡음, 움직이는 비율, 값 범위를 재서 맞는 태그를 `추천`으로 표시하고, 추천하지 않은 이유와 붙여 넣을 설정 조각을 출력한다. 이상 구간이 섞인 폴더라면 `--segments`로 학습 구간만 쓴다.
   - **맞는 신호**: 멈춰 있을 때 잡음이 아주 작고 값의 전체 범위는 넓은 신호(서보·실린더의 **위치**, 엔코더 값 등). 기준은 `범위/잡음 ≥ 10^5`.
   - **맞지 않는 신호**: 속도·토크처럼 평소에도 크게 흔들리는 신호, 0/1 신호, 값이 계단처럼 가끔만 바뀌는 신호(카운터, 타이머, 거친 양자화).
   - 도구가 추천하는 태그와 직접 고른 태그가 다르면 직접 고른 태그도 설정에 적을 수 있다. 다만 실제로 충격을 잡는지는 4번처럼 이상 데이터로 확인해야 한다.
3. **설정에 적는다.** 추천 조각을 설비 config에 붙인다. 이름(왼쪽)은 점수/알람 그룹 이름이 되고 GRU 그룹 이름과 겹치면 안 된다.

   ```yaml
   deviation:
     signals:
       axis_x: "NX5_AxisData:AxX_Act_Pos"
       axis_cv: {tag: "NX5_AxisData:AxCV_Act_Pos", move_range: 0.05}   # move_range는 선택(아래 표)
     # baseline_seconds: 2     # 기준값을 잡는 직전 구간(초)
     # settle_seconds: 1       # 판정을 늦추는 시간(초)
     # margin: 1.2             # 알람 기준 = 정상 최대 점수 x margin
     # confirm_steps: 1        # 알람이 확정되려면 연속으로 몇 스텝 넘어야 하나(충격은 짧아서 기본 1)
   ```

   | 옵션 | 기본 | 의미 |
   |---|---|---|
   | `baseline_seconds` | 2 | 현재 값을 비교할 기준(직전 몇 초 값의 중앙값). 느리게 흘러가는(드리프트) 신호는 짧게 |
   | `settle_seconds` | 1 | 판정을 늦추는 시간. 이 안에 움직임이 이어지면 충격이 아니라 움직임의 시작으로 본다 |
   | `margin` | 1.2 | 알람 기준 = 학습한 정상 최대 점수 × margin. 오탐이 나면 올리고, 놓치면 내린다(1.0 아래로는 정상에서 알람이 난다) |
   | `confirm_steps` | 1 | 연속 확정 횟수. 정상에서 단발 오탐이 나면 2로 올릴 수 있다(짧은 충격은 놓칠 수 있다) |
   | `move_range` | 자동 | 이 값보다 크게 움직이면 "축이 움직이는 중"으로 보고 점수를 내지 않는 기준. 보통 자동으로 충분하다 |

4. **정상 데이터로 학습하고, 이상 데이터로 확인한다.** 앱을 실행해 `train` 명령을 보내거나(`CALIBRATING`→`MONITORING`), 저장해 둔 파일로 `jetson-replay`를 쓴다.

   ```bash
   uv run jetson-replay train --config cfg.yaml --data-dir 정상폴더 --segments train.yaml --model-dir model_data
   uv run jetson-replay score --config cfg.yaml --data-dir 이상폴더 --segments eval.yaml --model-dir model_data --out scores.csv
   ```

   학습 로그에 신호별 `대기 잡음`, `알람 기준 z`, `움직임 기준`이 나온다. 학습 결과는 `model_data/<equipment_id>.deviation.json`에 저장되고 설정(격자 간격, baseline/settle, 신호 목록)이 바뀌면 재학습하라고 안내한다. 점수 CSV의 `<그룹이름>_score`가 임계선(기본 3.0)을 넘은 곳이 알람이다.

#### 점수를 읽는 법

- **점수 3.0(= `alarm.threshold`)이 알람 기준**이다. 내부적으로 z를 `알람 기준 z`로 나눠 임계값 배율로 환산한 값이라(`z ≥ 알람 기준 z` ⇒ 점수 ≥ 3.0), GRU 그룹과 같은 임계선·그래프(`jetson-plot`, 웹 화면)로 볼 수 있다.
- **0은 "움직이는 중"이거나 점수를 낼 수 없는 구간**이다. 축이 움직이는 동안은 값이 크게 바뀌는 것이 정상이라 점수를 내지 않는다.
- **점수는 `settle_seconds`(기본 1초) 늦게 나온다.** 알람 시각은 충격 시각보다 약 1초 뒤다. CSV의 시각은 점수가 나온 시각이다.

#### 알려진 한계 (쓰기 전에 읽을 것)

- **축이 움직이는 동안 생기는 이상은 이 방식으로 못 본다.** 움직이는 구간(NX5 데모에서 데이터의 약 4%)은 점수를 내지 않는다.
- **너무 큰 충격은 "움직임"으로 취급되어 점수가 0이 될 수 있다.** 3초 동안 값의 범위가 `move_range`를 넘으면 움직임으로 보기 때문이다. NX5에서는 충격이 `move_range`의 40% 이하였다(CV축: 충격 0.011, 기준 0.028). 더 센 충격을 다룬다면 `move_range`를 직접 더 크게 지정하되, 너무 크면 움직임이 시작되는 순간이 오탐이 될 수 있으니 정상 데이터로 `jetson-replay score`를 다시 돌려 확인한다.
- **알람 기준은 정상 데이터의 최대 점수에서 정해진다.** 정상 데이터가 적거나 대표성이 없으면(예: 측정 환경이 바뀌면) 정상에서 알람이 나거나 충격을 놓칠 수 있다. 정상 데이터를 더 모아 다시 학습한다.
- **약한 충격은 못 잡는다.** 충격이 정상 대기 잡음과 구별되지 않으면 어떤 설정으로도 잡을 수 없다.
- **이상 데이터의 라벨 시각이 DX1 시각과 어긋날 수 있다.** NX5 라벨은 DX1 시각보다 약 6초 늦었다(PC 시계 차이). 구간 파일의 이상 구간이 실제 충격과 어긋나면 검출이 안 된 것처럼 보이니, 점수 그래프(`jetson-plot`)나 원본 신호를 보고 시각을 맞춘다.
- **GRU와 함께 쓸 때** 알람은 그룹마다 따로 판단하고, 전체 점수는 모든 그룹 점수의 최댓값이다. 점수 CSV에는 GRU 그룹과 deviation 신호가 같은 줄에 나란히 기록된다.

#### 문제 해결

| 증상 | 확인할 것 |
|---|---|
| 학습할 때 "태그 ... 값을 한 번도 받지 못했습니다" | 설정의 태그 이름 오탈자, 구독 토픽에 그 태그가 있는지 |
| "학습 데이터 내내 움직이는 중" | 그 태그는 대기 중에도 계속 변한다(속도·토크 같은 신호). 이 방식에 맞지 않거나 `move_range`를 직접 지정 |
| "전혀 변하지 않아 잡음 크기를 알 수 없습니다" | 정상 데이터에서 그 신호가 전혀 안 변했다. 이 방식에 쓸 수 없거나 정상 데이터에 동작이 빠졌다 |
| 정상인데 알람이 난다 | 학습에 없던 정상 동작이 있다. 정상 데이터를 더 모아 재학습, 또는 `margin`↑ / `confirm_steps` 2 |
| 충격을 줬는데 점수가 안 오른다 | 라벨 시각 어긋남을 먼저 확인, 그다음 신호가 맞는지(`jetson-deviation-suggest`), 충격이 너무 큰지(위 한계) |

## 점수 그래프 화면

그룹별 이상 점수와 임계값을 선 그래프로 본다(읽기 전용, 외부 라이브러리/CDN 없음). 설계: `../docs/superpowers/specs/2026-10-07-score-dashboard-design.md`.

- **실시간**: `jetson-app ... --web-port 8080`이면 `http://127.0.0.1:8080/`에 화면이 열린다. 옵션이 없으면 이력도 서버도 만들지 않는다(`build_pipeline(history_seconds=0)`). 기본 바인딩은 `127.0.0.1`이고, `--web-host 0.0.0.0`이면 로그인이 없다는 경고를 출력한다. 포트를 열지 못해도 앱은 웹 없이 계속 돈다.
  - 점수 이력은 `ScoreHistory`(최근 3600초 링 버퍼, 재시작하면 사라짐)에 `HistoryRecordingPublisher`가 기록한다. 원래 MQTT 발행을 먼저 하고 이력 기록 실패는 삼킨다.
  - API(GET): `/`, `/healthz`, `/api/meta`, `/api/scores?seconds=S&max_points=N`. 긴 범위는 구간마다 그룹별 최댓값만 남겨 짧은 튐이 사라지지 않는다.
- **저장된 점수**: `jetson-plot --scores scores.csv [--config C] [--threshold T] [--confirm N] --out plot.html`. 데이터를 HTML에 내장해 서버 없이 열린다. 임계값/연속 횟수 슬라이더로 알람 구간과 요약(정상 알람 건수, 이상 검출)을 즉시 다시 계산한다. 이 계산(`dashboard.html`의 `BEGIN-CORE`~`END-CORE`)은 앱의 `Debouncer`와 같은 규칙이며, 테스트가 Node로 실행해 파이썬 결과와 비교한다.

