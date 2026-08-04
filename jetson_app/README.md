# jetson_app

Jetson 쪽 실시간 이상탐지 프레임워크의 통신 + 데이터 파이프라인 레이어. DX1(SpeeDBee Synapse)이 MQTT로 Publish하는 설비 태그 데이터를 여러 토픽에서 구독해 Tag Buffer에 모으고, 주기적으로 스냅샷을 떠서 슬라이딩 윈도우와 캘리브레이션 버퍼에 쌓는다. MQTT 명령으로 학습(train)/재캘리브레이션(recalibrate) 상태 전이를 제어한다.

전체 설계 배경은 [`../docs/superpowers/specs/2026-07-31-jetson-dx1-anomaly-framework-design.md`](../docs/superpowers/specs/2026-07-31-jetson-dx1-anomaly-framework-design.md), 이 통신 레이어의 구현 계획은 [`../docs/superpowers/plans/2026-08-03-jetson-mqtt-communication-layer.md`](../docs/superpowers/plans/2026-08-03-jetson-mqtt-communication-layer.md) 참고.

현재 범위: 설비 config 로더(다중 토픽) + MQTT 파싱/구독자 + Tag Buffer/슬라이딩 윈도우 + 주기 스냅샷 스케줄러 + 캘리브레이션 저장/상태머신 + MQTT train/recalibrate 명령 구독자 + CLI 진입점(코드, 유닛테스트 완료). 실제 모델 학습/Inference/Result Publisher는 이후 별도 계획(현재 `train_fn`은 자리표시자).

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

시작 시 `[test_dx1] 1개 토픽 구독 시작 (localhost:1883), 캘리브레이션 데이터: calibration_data` 형태의 구독 확인 줄이 출력된다. 이후에는 메시지마다 출력되지 않고, 데이터가 실제로 흐르고 있으면 약 5초에 한 번씩 다음과 같은 하트비트 줄이 찍힌다:

```
[snapshotter] 100번째 스냅샷 처리, 윈도우 10/10, 캘리브레이션 상태=CALIBRATING
```

즉 **하트비트 줄이 주기적으로 보이면 정상 동작 중**이고, 구독 확인 줄만 나오고 하트비트가 전혀 안 나오면 아직 태그 값이 하나도 안 들어온 것이다(아래 점검 순서 참고). `Ctrl+C`로 종료할 수 있다.

### 캘리브레이션 데이터 위치와 train/recalibrate 명령

- `--calibration-dir` 플래그로 캘리브레이션 버퍼 저장 위치를 지정한다(기본: `calibration_data`). 설비마다 `<calibration-dir>/<equipment_id>.jsonl` 파일에 스냅샷이 JSON Lines로 쌓인다.
- 기동 직후 상태는 `CALIBRATING`이며, 정상 데이터가 충분히 쌓이면(`calibration.min_samples` 이상) 아래처럼 `train` 명령을 보내 `MONITORING`으로 전환한다. 다시 캘리브레이션부터 시작하려면 `recalibrate`를 보낸다.

```bash
mosquitto_pub -h localhost -t "jetson/test_dx1/cmd" -m '{"command": "train"}'
mosquitto_pub -h localhost -t "jetson/test_dx1/cmd" -m '{"command": "recalibrate"}'
```

> 주의: 이 명령들에는 절대 `-r`(retain)을 붙이지 않는다. retain된 명령 메시지는 앱이 재접속할 때마다 다시 전달되어 의도치 않게 재실행된다.

### 문제 발생 시 점검 순서

1. **Jetson 쪽**: `mosquitto_sub -h localhost -t "dx1/test_dx1/telemetry" -v`로 브로커에 메시지가 도달하는지 먼저 확인한다.
   - 도달하면 → Jetson 코드/config 문제 (topic 오탈자, tags 불일치 등)
   - 도달 안 하면 → 네트워크/DX1 설정 문제
2. **DX1 쪽**: SpeeDBee의 MQTT Emitter 연결 상태/에러 로그, topic 오탈자를 확인한다.
3. **네트워크**: DX1과 Jetson이 같은 서브넷에 있는지, Jetson 방화벽(1-4)이 막고 있지 않은지 확인한다.
4. **인증 실패**: 브로커가 연결을 거부하면(`rc != 0`) 콘솔에 `MQTT 연결 실패 (rc=...)` 메시지가 출력된다 — 브로커 쪽 인증/ACL 설정(1-2)을 확인한다.
