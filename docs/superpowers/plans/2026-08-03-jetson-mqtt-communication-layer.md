# Jetson MQTT 통신 레이어 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DX1(SpeeDBee Synapse)이 MQTT로 Publish한 설비 태그 데이터를 Jetson이 Subscribe해서 콘솔에 출력하는, 설계 스펙 1절(전체 아키텍처)의 1번 컴포넌트(MQTT Subscriber)까지를 실기(DX1 + Jetson 하드웨어)에서 end-to-end로 검증한다.

**Architecture:** Jetson 자체에 Mosquitto MQTT 브로커를 설치해 DX1과 Jetson이 같은 브로커를 공유한다. DX1의 SpeeDBee MQTT Emitter가 Jetson IP:1883으로 Publish → Jetson의 Python 구독자(`jetson_app`)가 같은 브로커를 Subscribe해서 설비 config(YAML)에 등록된 태그만 필터링해 출력한다. Tag Buffer/Calibration/Inference/Result Publisher(스펙 1절의 2~5번 컴포넌트)는 이번 계획 범위 밖이며, 통신 검증 후 별도 계획으로 진행한다.

**Tech Stack:** Python 3.8(JetPack 5.x 기본 Python), uv(패키지 관리), paho-mqtt(MQTT 클라이언트), PyYAML(config 파싱), pytest(테스트), Mosquitto(MQTT 브로커, Ubuntu 20.04 apt 패키지)

## Global Constraints

- Python 패키지 관리는 반드시 `uv` 사용 (사용자 전역 CLAUDE.md 규칙)
- 대상 Jetson OS: JetPack 5.x / Ubuntu 20.04 (aarch64)
- `paho-mqtt`는 2.x의 콜백 API 변경(Callback API version 필수화)을 피하기 위해 `<2` 버전대 고정
- 설비 config 스키마는 설계 스펙 3절(`docs/superpowers/specs/2026-07-31-jetson-dx1-anomaly-framework-design.md`)의 YAML 구조를 그대로 따른다: `equipment_id`, `mqtt.subscribe_topic`, `mqtt.publish_topic`, `tags`
- 수신 JSON 스키마도 스펙 3절 그대로: `{"records": [{"timestamp": ISO8601, "태그명": 값, ...}]}`
- 이 저장소는 public GitHub repo(`Hyunkeun98/JetsonProject`)이므로 실제 고객사 설비 태그명이나 민감정보는 코드/예시 config에 넣지 않는다 — 예시 config는 `test_dx1` 같은 placeholder 태그명만 사용
- Jetson 실기에 대한 원격 접속 수단이 없으므로, Jetson에서 직접 실행해야 하는 단계(브로커 설치, 실배포, 실기 검증)는 사용자가 가이드를 보고 Jetson 콘솔에서 직접 실행한다. 이 세션에서 자동 실행되는 단계는 Windows 개발 PC에서의 코드 작성/유닛테스트뿐이다.

---

## File Structure

```
jetson_app/
  pyproject.toml              # uv 프로젝트 정의, paho-mqtt/pyyaml 의존성
  configs/
    test_dx1.example.yaml     # 예시 설비 config (placeholder 태그명)
  src/
    jetson_app/
      __init__.py
      config.py                # EquipmentConfig 로더 (YAML → dataclass)
      mqtt_subscriber.py        # JSON 파싱/태그 필터링 순수 함수 + MqttRecordSubscriber 클래스
      subscriber_cli.py         # 실행 진입점 (콘솔에 수신 레코드 출력)
  tests/
    test_config.py
    test_mqtt_subscriber.py
```

---

### Task 1: Jetson에 Mosquitto MQTT 브로커 설치 (Jetson 실기에서 직접 실행)

**Files:** 없음 (인프라 설정, Jetson OS 레벨)

**Interfaces:**
- Produces: Jetson의 `1883` 포트에서 리스닝하는 MQTT 브로커, DX1과 Jetson 양쪽에서 접속 가능

> 아래 명령은 전부 **Jetson 실기**의 터미널(직접 콘솔 또는 모니터/키보드 연결)에서 실행합니다. 이 세션은 Jetson에 원격 접속할 수 없으므로, 이 Task는 자동 실행되지 않고 가이드로만 제공됩니다.

- [ ] **Step 1: Mosquitto 설치**

```bash
sudo apt update
sudo apt install -y mosquitto mosquitto-clients
```

- [ ] **Step 2: 외부(DX1) 접속을 허용하는 리스너 설정**

Ubuntu 20.04 기본 Mosquitto는 익명 접속을 막아둘 수 있습니다. 테스트 단계이므로 우선 무인증으로 열고, 실제 운영 전환 시 별도로 인증을 추가합니다.

```bash
sudo tee /etc/mosquitto/conf.d/jetson.conf > /dev/null <<'EOF'
listener 1883 0.0.0.0
allow_anonymous true
EOF
```

- [ ] **Step 3: 브로커 재시작 및 부팅 시 자동 시작 등록**

```bash
sudo systemctl restart mosquitto
sudo systemctl enable mosquitto
sudo systemctl status mosquitto --no-pager
```

Expected: `active (running)` 표시

- [ ] **Step 4: 방화벽이 활성화되어 있다면 1883 포트 개방**

```bash
sudo ufw status
```

`Status: active`인 경우에만:

```bash
sudo ufw allow 1883/tcp
```

- [ ] **Step 5: 로컬 루프백으로 브로커 동작 검증**

터미널 두 개를 열고, 하나에는 구독:

```bash
mosquitto_sub -h localhost -t "test/ping"
```

다른 하나에는 발행:

```bash
mosquitto_pub -h localhost -t "test/ping" -m "hello"
```

Expected: 구독 터미널에 `hello` 출력

- [ ] **Step 6: Jetson의 IP 주소 확인 (다음 Task의 DX1 설정에 필요)**

```bash
hostname -I
```

이 IP를 메모해둡니다 — DX1의 SpeeDBee MQTT Emitter 설정과 Task 6의 CLI 실행에 사용합니다.

---

### Task 2: uv 기반 `jetson_app` 프로젝트 스캐폴딩

**Files:**
- Create: `jetson_app/pyproject.toml`
- Create: `jetson_app/src/jetson_app/__init__.py`

**Interfaces:**
- Produces: `uv run pytest`, `uv run python -m jetson_app.subscriber_cli`가 동작하는 프로젝트 골격

> 이 Task는 Windows 개발 PC(`C:\WORK\10. Jetson`)에서 실행합니다.

- [ ] **Step 1: uv 프로젝트 초기화**

```bash
cd "C:\WORK\10. Jetson"
uv init --package jetson_app
```

- [ ] **Step 2: 의존성 추가**

```bash
cd jetson_app
uv add "paho-mqtt<2" pyyaml
uv add --dev pytest
```

- [ ] **Step 3: 골격 확인**

```bash
uv run pytest
```

Expected: 테스트가 아직 없으므로 `no tests ran` 류의 성공 종료 (에러 없음)

- [ ] **Step 4: Commit**

```bash
git add jetson_app/pyproject.toml jetson_app/uv.lock jetson_app/src
git commit -m "chore: scaffold jetson_app uv project"
```

---

### Task 3: 설비 Config 로더 (`config.py`)

**Files:**
- Create: `jetson_app/src/jetson_app/config.py`
- Test: `jetson_app/tests/test_config.py`
- Create: `jetson_app/configs/test_dx1.example.yaml`

**Interfaces:**
- Produces:
  - `class ConfigError(ValueError)`
  - `@dataclass(frozen=True) class EquipmentConfig: equipment_id: str; subscribe_topic: str; publish_topic: str; tags: tuple[str, ...]`
  - `def load_equipment_config(path: str | Path) -> EquipmentConfig`

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_config.py`:

```python
import pytest

from jetson_app.config import ConfigError, load_equipment_config


SAMPLE_YAML = """\
equipment_id: "test_dx1"
mqtt:
  subscribe_topic: "dx1/test_dx1/telemetry"
  publish_topic: "jetson/test_dx1/anomaly"
tags:
  - "servo1:torque"
  - "sensor:A_L_01"
"""


def test_load_equipment_config_parses_valid_yaml(tmp_path):
    config_path = tmp_path / "test_dx1.yaml"
    config_path.write_text(SAMPLE_YAML, encoding="utf-8")

    config = load_equipment_config(config_path)

    assert config.equipment_id == "test_dx1"
    assert config.subscribe_topic == "dx1/test_dx1/telemetry"
    assert config.publish_topic == "jetson/test_dx1/anomaly"
    assert config.tags == ("servo1:torque", "sensor:A_L_01")


def test_load_equipment_config_missing_tags_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        'equipment_id: "x"\nmqtt:\n  subscribe_topic: "a"\n  publish_topic: "b"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_missing_mqtt_section_raises(tmp_path):
    config_path = tmp_path / "bad2.yaml"
    config_path.write_text(
        'equipment_id: "x"\ntags:\n  - "a"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd jetson_app
uv run pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jetson_app.config'`

- [ ] **Step 3: 구현 작성**

`jetson_app/src/jetson_app/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EquipmentConfig:
    equipment_id: str
    subscribe_topic: str
    publish_topic: str
    tags: tuple[str, ...]


def load_equipment_config(path: str | Path) -> EquipmentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    for field in ("equipment_id", "mqtt", "tags"):
        if field not in data:
            raise ConfigError(f"missing required field: {field}")

    mqtt_section = data["mqtt"]
    for field in ("subscribe_topic", "publish_topic"):
        if field not in mqtt_section:
            raise ConfigError(f"missing required field: mqtt.{field}")

    tags = data["tags"]
    if not isinstance(tags, list) or not tags:
        raise ConfigError("tags must be a non-empty list")

    return EquipmentConfig(
        equipment_id=data["equipment_id"],
        subscribe_topic=mqtt_section["subscribe_topic"],
        publish_topic=mqtt_section["publish_topic"],
        tags=tuple(tags),
    )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_config.py -v
```

Expected: PASS (3 passed)

- [ ] **Step 5: 예시 config 파일 작성**

`jetson_app/configs/test_dx1.example.yaml`:

```yaml
equipment_id: "test_dx1"
mqtt:
  subscribe_topic: "dx1/test_dx1/telemetry"
  publish_topic: "jetson/test_dx1/anomaly"
tags:
  - "servo1:torque"
  - "servo1:position"
  - "sensor:A_L_01"
  # DX1 SpeeDBee JSON Serializer가 실제로 내보내는 "컴포넌트명:데이터명" 형식의
  # 태그명으로 교체하세요 (Task 6에서 DX1 설정과 맞춥니다).
```

- [ ] **Step 6: Commit**

```bash
git add jetson_app/src/jetson_app/config.py jetson_app/tests/test_config.py jetson_app/configs/test_dx1.example.yaml
git commit -m "feat: add equipment config loader"
```

---

### Task 4: MQTT 메시지 파싱/필터링 + 구독자 (`mqtt_subscriber.py`)

**Files:**
- Create: `jetson_app/src/jetson_app/mqtt_subscriber.py`
- Test: `jetson_app/tests/test_mqtt_subscriber.py`

**Interfaces:**
- Consumes: `EquipmentConfig`(Task 3의 `equipment_id`, `subscribe_topic`, `tags`)
- Produces:
  - `@dataclass(frozen=True) class Record: timestamp: str; values: dict[str, float]`
  - `def parse_and_filter_records(payload: bytes, tags: tuple[str, ...]) -> list[Record]`
  - `class MqttRecordSubscriber: def __init__(self, config: EquipmentConfig, on_record: Callable[[Record], None]); def connect(self, host: str, port: int = 1883) -> None; def loop_forever(self) -> None`

- [ ] **Step 1: 실패하는 테스트 작성 (순수 파싱 함수만 — 실제 브로커 연결 없이 검증 가능)**

`jetson_app/tests/test_mqtt_subscriber.py`:

```python
from jetson_app.mqtt_subscriber import Record, parse_and_filter_records


def test_parse_and_filter_records_keeps_only_configured_tags():
    payload = (
        b'{"records": [{"timestamp": "2026-08-03T00:00:00Z", '
        b'"servo1:torque": 12.3, "servo1:unused": 99.9, "sensor:A_L_01": 110.2}]}'
    )

    records = parse_and_filter_records(payload, tags=("servo1:torque", "sensor:A_L_01"))

    assert records == [
        Record(
            timestamp="2026-08-03T00:00:00Z",
            values={"servo1:torque": 12.3, "sensor:A_L_01": 110.2},
        )
    ]


def test_parse_and_filter_records_skips_record_with_no_matching_tags():
    payload = b'{"records": [{"timestamp": "2026-08-03T00:00:00Z", "unrelated:tag": 1.0}]}'

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_handles_multiple_records():
    payload = (
        b'{"records": ['
        b'{"timestamp": "t1", "servo1:torque": 1.0}, '
        b'{"timestamp": "t2", "servo1:torque": 2.0}'
        b']}'
    )

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert [r.timestamp for r in records] == ["t1", "t2"]
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_mqtt_subscriber.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jetson_app.mqtt_subscriber'`

- [ ] **Step 3: 구현 작성**

`jetson_app/src/jetson_app/mqtt_subscriber.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import paho.mqtt.client as mqtt

from .config import EquipmentConfig


@dataclass(frozen=True)
class Record:
    timestamp: str
    values: dict[str, float]


def parse_and_filter_records(payload: bytes, tags: tuple[str, ...]) -> list[Record]:
    tag_set = set(tags)
    data = json.loads(payload)
    records = data.get("records", [])

    result = []
    for raw in records:
        timestamp = raw.get("timestamp")
        values = {k: v for k, v in raw.items() if k in tag_set}
        if timestamp is not None and values:
            result.append(Record(timestamp=timestamp, values=values))
    return result


class MqttRecordSubscriber:
    def __init__(
        self,
        config: EquipmentConfig,
        on_record: Callable[[Record], None],
    ) -> None:
        self._config = config
        self._on_record = on_record
        self._client = mqtt.Client()
        self._client.on_connect = self._handle_connect
        self._client.on_message = self._handle_message

    def connect(self, host: str, port: int = 1883) -> None:
        self._client.connect(host, port)

    def loop_forever(self) -> None:
        self._client.loop_forever()

    def _handle_connect(self, client, userdata, flags, rc):
        client.subscribe(self._config.subscribe_topic)

    def _handle_message(self, client, userdata, msg):
        for record in parse_and_filter_records(msg.payload, self._config.tags):
            self._on_record(record)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_mqtt_subscriber.py -v
```

Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add jetson_app/src/jetson_app/mqtt_subscriber.py jetson_app/tests/test_mqtt_subscriber.py
git commit -m "feat: add MQTT record parsing and subscriber"
```

---

### Task 5: CLI 진입점 (`subscriber_cli.py`)

**Files:**
- Create: `jetson_app/src/jetson_app/subscriber_cli.py`

**Interfaces:**
- Consumes: `load_equipment_config`(Task 3), `MqttRecordSubscriber`, `Record`(Task 4)
- Produces: `python -m jetson_app.subscriber_cli --config <path> --host <host> [--port 1883]` 실행형 진입점

이 컴포넌트는 실제 MQTT 브로커 연결이 전제라 유닛테스트 대상이 아니며, Task 6에서 Jetson 실기로 수동 검증합니다.

- [ ] **Step 1: 구현 작성**

`jetson_app/src/jetson_app/subscriber_cli.py`:

```python
from __future__ import annotations

import argparse

from .config import load_equipment_config
from .mqtt_subscriber import MqttRecordSubscriber, Record


def _print_record(record: Record) -> None:
    print(f"[{record.timestamp}] {record.values}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DX1 -> Jetson MQTT 수신 확인용 구독자")
    parser.add_argument("--config", required=True, help="설비 config YAML 경로")
    parser.add_argument("--host", required=True, help="MQTT 브로커 호스트 (Jetson 자신의 IP 또는 localhost)")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    config = load_equipment_config(args.config)
    subscriber = MqttRecordSubscriber(config, on_record=_print_record)
    subscriber.connect(args.host, args.port)

    print(f"[{config.equipment_id}] '{config.subscribe_topic}' 구독 시작 ({args.host}:{args.port})")
    subscriber.loop_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 로컬(Windows)에서 import 에러 없이 로드되는지만 확인 (실제 연결은 하지 않음)**

```bash
uv run python -c "from jetson_app.subscriber_cli import main; print('ok')"
```

Expected: `ok` 출력, 에러 없음

- [ ] **Step 3: Commit**

```bash
git add jetson_app/src/jetson_app/subscriber_cli.py
git commit -m "feat: add subscriber CLI entrypoint"
```

---

### Task 6: Jetson 배포 + DX1 SpeeDBee 설정 + End-to-End 검증 (Jetson 실기 + DX1에서 직접 실행)

**Files:** 없음 (배포/설정 가이드)

> Task 1과 마찬가지로, 이 Task는 사용자가 Jetson 실기와 DX1 SpeeDBee 설정 화면에서 직접 수행합니다.

- [ ] **Step 1: (이 세션에서) GitHub에 푸시하기 전에 사용자 확인**

Task 2~5에서 만든 `jetson_app/`은 민감정보 없는 순수 코드이므로 기존 public repo에 푸시하는 것이 합리적이지만, 실제 푸시 전에는 항상 먼저 확인받습니다 (이전 Data 폴더 제외 결정과 동일한 원칙).

```bash
git push origin main
```

- [ ] **Step 2: Jetson에서 저장소 clone/pull**

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

- [ ] **Step 3: Jetson에 uv 설치 (없는 경우)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

- [ ] **Step 4: 의존성 설치**

```bash
uv sync
```

- [ ] **Step 5: 테스트 설정 파일 준비**

```bash
cp configs/test_dx1.example.yaml configs/test_dx1.yaml
```

`configs/test_dx1.yaml`을 열어 `tags` 목록을, DX1에서 실제로 내보낼 태그명(다음 Step에서 SpeeDBee에 등록할 이름)과 동일하게 수정합니다.

- [ ] **Step 6: DX1의 SpeeDBee Synapse에서 MQTT Emitter 설정**

SpeeDBee Synapse 관리 화면에서:
1. 기존에 구성된 Collector(PLC/센서 등)의 출력을 JSON Serializer에 연결 (스펙 3절 포맷: `{"records": [{"timestamp": ..., "태그명": 값, ...}]}`)
2. MQTT Emitter 컴포넌트 추가:
   - Broker Host: Task 1 Step 6에서 확인한 Jetson IP
   - Port: `1883`
   - Topic: `dx1/test_dx1/telemetry` (Task 5의 config 파일과 정확히 일치해야 함)
   - Publish 트리거를 활성화

- [ ] **Step 7: Jetson에서 구독자 실행**

```bash
uv run python -m jetson_app.subscriber_cli --config configs/test_dx1.yaml --host localhost
```

- [ ] **Step 8: DX1이 Publish하는 실제 데이터가 콘솔에 찍히는지 확인**

Expected: `[test_dx1] 'dx1/test_dx1/telemetry' 구독 시작 (localhost:1883)` 출력 후, DX1이 Publish할 때마다 `[타임스탬프] {태그: 값, ...}` 형태로 실시간 출력

문제 발생 시 점검 순서:
- Jetson 쪽: `mosquitto_sub -h localhost -t "dx1/test_dx1/telemetry" -v` 로 브로커에 메시지가 도달하는지 먼저 확인 (도달하면 Jetson 코드 문제, 도달 안 하면 네트워크/DX1 설정 문제)
- DX1 쪽: SpeeDBee의 MQTT Emitter 연결 상태/에러 로그 확인, topic 오탈자 확인
- 네트워크: DX1과 Jetson이 같은 서브넷에 있는지, Jetson 방화벽(Task 1 Step 4)이 막고 있지 않은지 확인

---

## Self-Review 결과

**스펙 커버리지**: 설계 스펙 1절의 "MQTT Emitter → MQTT Subscriber" 구간과 3절의 config/데이터 계약 스키마를 Task 1~6이 실기로 검증한다. Tag Buffer/Calibration Manager/Inference Engine/Result Publisher(스펙 1절 2~5번)는 의도적으로 범위 밖 — 통신 레이어 검증 완료 후 별도 계획(Phase 1 나머지)으로 이어간다.

**Placeholder 스캔**: 모든 코드 스텝에 실제 동작하는 코드/커맨드 포함, "TODO"/"적절히 처리" 류 표현 없음.

**타입 일관성**: `EquipmentConfig`(Task 3) → `MqttRecordSubscriber.__init__`(Task 4) → `subscriber_cli.main`(Task 5)까지 필드명(`equipment_id`, `subscribe_topic`, `publish_topic`, `tags`)과 `Record` 필드명(`timestamp`, `values`)이 각 Task의 Interfaces 블록과 실제 코드에서 일치함을 확인.
