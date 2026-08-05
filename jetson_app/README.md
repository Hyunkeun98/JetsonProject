# jetson_app

Jetson 쪽 실시간 이상탐지 프레임워크의 통신 + 데이터 파이프라인 레이어. DX1(SpeeDBee Synapse)이 MQTT로 Publish하는 설비 태그 데이터를 여러 토픽에서 구독해 Tag Buffer에 모으고, 주기적으로 스냅샷을 떠서 슬라이딩 윈도우와 캘리브레이션 버퍼에 쌓는다. MQTT 명령으로 학습(train)/재캘리브레이션(recalibrate) 상태 전이를 제어한다.

전체 설계 배경은 [`../docs/superpowers/specs/2026-07-31-jetson-dx1-anomaly-framework-design.md`](../docs/superpowers/specs/2026-07-31-jetson-dx1-anomaly-framework-design.md), 이 통신 레이어의 구현 계획은 [`../docs/superpowers/plans/2026-08-03-jetson-mqtt-communication-layer.md`](../docs/superpowers/plans/2026-08-03-jetson-mqtt-communication-layer.md) 참고.

현재 범위: 설비 config 로더(다중 토픽) + MQTT 파싱/구독자 + Tag Buffer/슬라이딩 윈도우 + 주기 스냅샷 스케줄러 + 캘리브레이션 저장/상태머신 + MQTT train/recalibrate 명령 구독자 + 설비 통합 PyTorch GRU 모델 학습(태그 타입별 손실, 정규화/오차 통계 저장/로드) + CLI 진입점(코드, 유닛테스트 완료). 실시간 이상 점수 계산/디바운스/Result Publisher는 이후 별도 계획.

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

시작 시 `[test_dx1] 1개 토픽 구독 시작 (localhost:1883), 캘리브레이션 데이터: calibration_data, 모델 저장 위치: model_data` 형태의 구독 확인 줄이 출력된다. 이후에는 메시지마다 출력되지 않고, 데이터가 실제로 흐르고 있으면 약 5초에 한 번씩 다음과 같은 하트비트 줄이 찍힌다:

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

- `--model-dir` 플래그로 학습된 모델을 저장할 위치를 지정한다(기본: `model_data`). `train` 명령이 성공하면 `<model-dir>/<equipment_id>.pt`에 GRU 가중치 + 정규화 통계 + 태그 타입 + 오차 분포 통계가 함께 저장된다.
- `train` 명령은 이제 실제로 PyTorch GRU 모델을 학습한다. **학습은 수 분 단위로 걸린다** — 예제 config 규모(`min_samples: 10000`, `window_size: 100`, 기본 `hidden_size=64`, `num_layers=2`, `epochs=20`)에서 x86 개발 PC CPU 기준 약 **9분**이 걸렸고, Jetson CPU에서는 더 느리다. 캘리브레이션 버퍼가 아무리 커도 학습에는 가장 최근 20,000 샘플만 사용한다(`train_model`/`make_train_fn`의 `max_training_samples`) — 7일치 버퍼를 통째로 윈도잉하면 메모리가 GB 단위로 튀어 Jetson에서 OOM이 나기 때문이다.

  이 학습은 현재 **MQTT 네트워크 스레드 위에서 동기로** 돈다. 그래서 학습이 도는 수 분 동안:

  - `MqttRecordSubscriber`가 `loop_forever()`를 돌리던 그 스레드가 막히므로, paho 기본 keepalive(60초)를 넘겨 브로커가 연결을 끊는다. 학습이 끝나면 자동으로 재접속 + 재구독되어 스스로 복구되지만, 끊긴 동안 DX1이 발행한 텔레메트리는 QoS 0이라 그대로 유실된다.
  - `CalibrationManager.handle_train_command`가 학습 내내 락을 잡고 있어서 백그라운드 스냅샷 스레드도 함께 멈춘다 — 하트비트 줄도, 윈도우 갱신도 그동안 나오지 않는다.

  즉 학습 중 콘솔이 조용하고 잠깐 연결이 끊겼다 붙는 것은 **현재 구조상 정상 동작**이다. 학습을 네트워크 스레드 밖(별도 워커 스레드/프로세스)으로 빼는 것은 다음 계획의 범위다.

> 주의: 이 명령들에는 절대 `-r`(retain)을 붙이지 않는다. retain된 명령 메시지는 앱이 재접속할 때마다 다시 전달되어 의도치 않게 재실행된다.

### 문제 발생 시 점검 순서

1. **Jetson 쪽**: `mosquitto_sub -h localhost -t "dx1/test_dx1/telemetry" -v`로 브로커에 메시지가 도달하는지 먼저 확인한다.
   - 도달하면 → Jetson 코드/config 문제 (topic 오탈자, tags 불일치 등)
   - 도달 안 하면 → 네트워크/DX1 설정 문제
2. **DX1 쪽**: SpeeDBee의 MQTT Emitter 연결 상태/에러 로그, topic 오탈자를 확인한다.
3. **네트워크**: DX1과 Jetson이 같은 서브넷에 있는지, Jetson 방화벽(1-4)이 막고 있지 않은지 확인한다.
4. **인증 실패**: 브로커가 연결을 거부하면(`rc != 0`) 콘솔에 `MQTT 연결 실패 (rc=...)` 메시지가 출력된다 — 브로커 쪽 인증/ACL 설정(1-2)을 확인한다.
