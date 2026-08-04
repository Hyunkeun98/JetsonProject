# Jetson 데이터 파이프라인 (Config/Buffer/Calibration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DX1의 여러 MQTT 토픽에서 들어오는 태그 데이터를 주기적으로 스냅샷/윈도우로 버퍼링하고, 사람이 `train`/`recalibrate` 명령을 MQTT로 보내면 캘리브레이션 데이터를 디스크에서 읽어 학습을 트리거하는 상태 머신까지 구현한다. 실제 ML 모델(GRU) 구현은 이 계획의 범위 밖이며, `train_fn`이라는 콜백 인터페이스로 자리만 잡아둔다(다음 계획에서 채움).

**Architecture:** MQTT 수신 스레드가 여러 토픽을 구독해 태그 최신값을 스레드세이프 캐시(`TagBuffer`)에 갱신하고, 별도 주기 스레드(`PeriodicSnapshotter`)가 `resample_interval_ms`마다 스냅샷을 찍어 슬라이딩 윈도우(`SlidingWindow`)에 밀어넣는다. `CALIBRATING` 상태 동안은 같은 스냅샷을 디스크 JSONL 파일에도 append한다(`CalibrationBufferWriter`). 별도 MQTT 콜백(`CommandSubscriber`)이 `train`/`recalibrate` 명령을 받아 `CalibrationManager`의 상태를 전이시킨다.

**Tech Stack:** Python 3.8, `uv`, `paho-mqtt<2`(기존 의존성 재사용), `pyyaml`(기존), 표준 라이브러리(`threading`, `collections.deque`, `json`, `datetime`, `enum`, `re`) — 새 외부 의존성 없음.

## Global Constraints

- Python 패키지 관리는 반드시 `uv` 사용
- `requires-python = ">=3.8"` 유지 — 새 코드도 `from __future__ import annotations` 사용, 3.9+ 전용 문법 금지
- `paho-mqtt<2` 유지 (콜백 API 1.x 시그니처: `on_connect(client, userdata, flags, rc)`, `on_message(client, userdata, msg)`)
- 상위 스펙 문서: [`docs/superpowers/specs/2026-08-04-jetson-anomaly-inference-pipeline-design.md`](../specs/2026-08-04-jetson-anomaly-inference-pipeline-design.md) — 이 계획은 그 문서의 1~4절(Config, Tag Buffer, 캘리브레이션 데이터 저장, 캘리브레이션 트리거)을 구현한다. 5~6절(모델, 실시간 이상 점수)은 범위 밖.
- **Breaking change 승인됨**: 기존 `EquipmentConfig.subscribe_topic`(단수)을 `subscribe_topics`(복수)로 변경 — Task 2~5에서 이미 구현된 `config.py`/`mqtt_subscriber.py`/`subscriber_cli.py`와 그 테스트를 이 계획에서 함께 수정한다
- 이 저장소는 public GitHub repo — 예시 config에는 실제 고객사 태그명이 아닌 플레이스홀더만 사용 (기존 `configs/test_dx1.example.yaml` 관례 유지, 이번엔 실제 확인된 DX1 태그명 패턴을 예시로 써도 무방 — 고객 식별정보 아님)
- 캘리브레이션 버퍼 타임스탬프는 DX1 원본 레코드의 타임스탬프가 아니라, **스냅샷을 찍는 시점에 우리가 직접 생성**한 `datetime.now(timezone.utc).isoformat()` 값이다 (마이크로초 정밀도, `+00:00` 오프셋 — Python 3.8의 `datetime.fromisoformat()`으로 안전하게 파싱 가능). DX1 원본 타임스탬프(나노초 정밀도, `+0000` 형식)를 그대로 저장/파싱하지 않는다 — 파싱 호환성 문제를 피하기 위한 의도적 설계.

---

## File Structure

```
jetson_app/
  configs/
    test_dx1.example.yaml       # 수정: 새 스키마(다중 토픽, resample/window, calibration)
  src/jetson_app/
    config.py                    # 수정: EquipmentConfig v2, CalibrationConfig, parse_duration()
    buffer.py                    # 신규: Snapshot, TagBuffer, SlidingWindow
    calibration.py                # 신규: CalibrationBufferWriter, CalibrationManager, CalibrationState
    scheduler.py                  # 신규: PeriodicSnapshotter
    command_subscriber.py         # 신규: parse_command(), CommandSubscriber
    mqtt_subscriber.py            # 수정: 다중 토픽 구독 + command_topic 구독 + client property
    pipeline.py                   # 신규: build_pipeline() — 위 컴포넌트를 전부 연결
    subscriber_cli.py             # 수정: build_pipeline() 사용하도록 재작성
  tests/
    test_config.py                # 수정: 새 스키마 반영
    test_buffer.py                # 신규
    test_calibration.py           # 신규
    test_scheduler.py             # 신규
    test_command_subscriber.py    # 신규
    test_mqtt_subscriber.py       # 수정: 다중 토픽 구독 테스트 추가
```

---

### Task 1: Config 스키마 v2 (다중 토픽 + 파이프라인 파라미터)

**Files:**
- Modify: `jetson_app/src/jetson_app/config.py`
- Modify: `jetson_app/tests/test_config.py`
- Modify: `jetson_app/configs/test_dx1.example.yaml`

**Interfaces:**
- Produces:
  - `class ConfigError(ValueError)` (기존 유지)
  - `def parse_duration(text: str) -> datetime.timedelta` — `"7d"`/`"12h"`/`"30m"`/`"45s"` 형식 파싱, 실패 시 `ConfigError`
  - `@dataclass(frozen=True) class CalibrationConfig: max_duration: timedelta; min_samples: int`
  - `@dataclass(frozen=True) class EquipmentConfig: equipment_id: str; subscribe_topics: tuple[str, ...]; publish_topic: str; command_topic: str; tags: tuple[str, ...]; resample_interval_ms: int; window_size: int; calibration: CalibrationConfig`
  - `def load_equipment_config(path: str | Path) -> EquipmentConfig`

- [ ] **Step 1: 실패하는 테스트 작성 (전체 교체)**

`jetson_app/tests/test_config.py` 전체를 아래로 교체:

```python
import pytest

from jetson_app.config import ConfigError, load_equipment_config, parse_duration


SAMPLE_YAML = """\
equipment_id: "test_dx1"
mqtt:
  subscribe_topics:
    - "dx1/test_dx1/actuator_1"
    - "dx1/test_dx1/axis_status_3"
  publish_topic: "jetson/test_dx1/anomaly"
  command_topic: "jetson/test_dx1/cmd"
tags:
  - "PLC_Collector_Actuator_1:AirBlower.Cmd[0]"
  - "PLC_Collector_Axis_Status_3:AxCV_VelDemVal"
resample_interval_ms: 50
window_size: 100
calibration:
  max_duration: "7d"
  min_samples: 10000
"""


def test_load_equipment_config_parses_valid_yaml(tmp_path):
    config_path = tmp_path / "test_dx1.yaml"
    config_path.write_text(SAMPLE_YAML, encoding="utf-8")

    config = load_equipment_config(config_path)

    assert config.equipment_id == "test_dx1"
    assert config.subscribe_topics == (
        "dx1/test_dx1/actuator_1",
        "dx1/test_dx1/axis_status_3",
    )
    assert config.publish_topic == "jetson/test_dx1/anomaly"
    assert config.command_topic == "jetson/test_dx1/cmd"
    assert config.tags == (
        "PLC_Collector_Actuator_1:AirBlower.Cmd[0]",
        "PLC_Collector_Axis_Status_3:AxCV_VelDemVal",
    )
    assert config.resample_interval_ms == 50
    assert config.window_size == 100
    assert config.calibration.min_samples == 10000
    from datetime import timedelta

    assert config.calibration.max_duration == timedelta(days=7)


def test_load_equipment_config_missing_subscribe_topics_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        'equipment_id: "x"\n'
        'mqtt:\n  publish_topic: "b"\n  command_topic: "c"\n'
        'tags:\n  - "a"\n'
        "resample_interval_ms: 50\nwindow_size: 100\n"
        'calibration:\n  max_duration: "7d"\n  min_samples: 10\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_empty_subscribe_topics_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        'equipment_id: "x"\n'
        'mqtt:\n  subscribe_topics: []\n  publish_topic: "b"\n  command_topic: "c"\n'
        'tags:\n  - "a"\n'
        "resample_interval_ms: 50\nwindow_size: 100\n"
        'calibration:\n  max_duration: "7d"\n  min_samples: 10\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_missing_calibration_section_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        'equipment_id: "x"\n'
        'mqtt:\n  subscribe_topics: ["t"]\n  publish_topic: "b"\n  command_topic: "c"\n'
        'tags:\n  - "a"\n'
        "resample_interval_ms: 50\nwindow_size: 100\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_invalid_min_samples_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        'equipment_id: "x"\n'
        'mqtt:\n  subscribe_topics: ["t"]\n  publish_topic: "b"\n  command_topic: "c"\n'
        'tags:\n  - "a"\n'
        "resample_interval_ms: 50\nwindow_size: 100\n"
        'calibration:\n  max_duration: "7d"\n  min_samples: "not-a-number"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_invalid_duration_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        'equipment_id: "x"\n'
        'mqtt:\n  subscribe_topics: ["t"]\n  publish_topic: "b"\n  command_topic: "c"\n'
        'tags:\n  - "a"\n'
        "resample_interval_ms: 50\nwindow_size: 100\n"
        'calibration:\n  max_duration: "banana"\n  min_samples: 10\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_empty_yaml_raises(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_malformed_yaml_raises(tmp_path):
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("equipment_id: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_equipment_config(config_path)


def test_load_equipment_config_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_equipment_config(tmp_path / "does-not-exist.yaml")


def test_load_equipment_config_loads_shipped_example_config():
    from pathlib import Path

    example_path = Path(__file__).parent.parent / "configs" / "test_dx1.example.yaml"
    config = load_equipment_config(example_path)

    assert len(config.subscribe_topics) >= 1
    assert len(config.tags) > 0


def test_parse_duration_parses_days():
    from datetime import timedelta

    assert parse_duration("7d") == timedelta(days=7)


def test_parse_duration_parses_hours_minutes_seconds():
    from datetime import timedelta

    assert parse_duration("12h") == timedelta(hours=12)
    assert parse_duration("30m") == timedelta(minutes=30)
    assert parse_duration("45s") == timedelta(seconds=45)


def test_parse_duration_invalid_format_raises():
    with pytest.raises(ConfigError):
        parse_duration("banana")
    with pytest.raises(ConfigError):
        parse_duration("7")
    with pytest.raises(ConfigError):
        parse_duration("7x")
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd jetson_app
uv run pytest tests/test_config.py -v
```

Expected: FAIL — `ImportError: cannot import name 'parse_duration'` 등 (구현 전이라 모듈이 새 이름을 export 안 함)

- [ ] **Step 3: 구현 작성 (전체 교체)**

`jetson_app/src/jetson_app/config.py` 전체를 아래로 교체:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


_DURATION_PATTERN = re.compile(r"^(\d+)([smhd])$")
_DURATION_UNIT_KEYWORDS = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
}


def parse_duration(text: str) -> timedelta:
    match = _DURATION_PATTERN.match(text.strip())
    if not match:
        raise ConfigError(
            f"invalid duration '{text}': expected format like '7d', '12h', '30m', '45s'"
        )
    amount = int(match.group(1))
    unit_keyword = _DURATION_UNIT_KEYWORDS[match.group(2)]
    return timedelta(**{unit_keyword: amount})


@dataclass(frozen=True)
class CalibrationConfig:
    max_duration: timedelta
    min_samples: int


@dataclass(frozen=True)
class EquipmentConfig:
    equipment_id: str
    subscribe_topics: tuple[str, ...]
    publish_topic: str
    command_topic: str
    tags: tuple[str, ...]
    resample_interval_ms: int
    window_size: int
    calibration: CalibrationConfig


def load_equipment_config(path: str | Path) -> EquipmentConfig:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read config file {path}: {e}") from e

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError("config file must contain a YAML mapping")

    for field in ("equipment_id", "mqtt", "tags", "resample_interval_ms", "window_size", "calibration"):
        if field not in data:
            raise ConfigError(f"missing required field: {field}")

    mqtt_section = data["mqtt"]
    if not isinstance(mqtt_section, dict):
        raise ConfigError("mqtt section must be a mapping")

    for field in ("subscribe_topics", "publish_topic", "command_topic"):
        if field not in mqtt_section:
            raise ConfigError(f"missing required field: mqtt.{field}")

    subscribe_topics = mqtt_section["subscribe_topics"]
    if not isinstance(subscribe_topics, list) or not subscribe_topics:
        raise ConfigError("mqtt.subscribe_topics must be a non-empty list")

    tags = data["tags"]
    if not isinstance(tags, list) or not tags:
        raise ConfigError("tags must be a non-empty list")

    resample_interval_ms = data["resample_interval_ms"]
    if not isinstance(resample_interval_ms, int) or isinstance(resample_interval_ms, bool) or resample_interval_ms <= 0:
        raise ConfigError("resample_interval_ms must be a positive integer")

    window_size = data["window_size"]
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
        raise ConfigError("window_size must be a positive integer")

    calibration_section = data["calibration"]
    if not isinstance(calibration_section, dict):
        raise ConfigError("calibration section must be a mapping")

    for field in ("max_duration", "min_samples"):
        if field not in calibration_section:
            raise ConfigError(f"missing required field: calibration.{field}")

    max_duration = parse_duration(str(calibration_section["max_duration"]))

    min_samples = calibration_section["min_samples"]
    if not isinstance(min_samples, int) or isinstance(min_samples, bool) or min_samples <= 0:
        raise ConfigError("calibration.min_samples must be a positive integer")

    return EquipmentConfig(
        equipment_id=data["equipment_id"],
        subscribe_topics=tuple(subscribe_topics),
        publish_topic=mqtt_section["publish_topic"],
        command_topic=mqtt_section["command_topic"],
        tags=tuple(tags),
        resample_interval_ms=resample_interval_ms,
        window_size=window_size,
        calibration=CalibrationConfig(max_duration=max_duration, min_samples=min_samples),
    )
```

Note: `isinstance(x, bool)` 체크가 들어간 이유 — Python에서 `bool`은 `int`의 서브클래스라 `isinstance(True, int)`가 `True`를 반환한다. `resample_interval_ms: true` 같은 오타를 정수로 오인하지 않도록 명시적으로 배제한다.

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_config.py -v
```

Expected: PASS (14 passed)

- [ ] **Step 5: 예시 config 갱신**

`jetson_app/configs/test_dx1.example.yaml` 전체를 아래로 교체:

```yaml
equipment_id: "test_dx1"
mqtt:
  subscribe_topics:
    - "dx1/test_dx1/actuator_1"
    - "dx1/test_dx1/axis_status_3"
  publish_topic: "jetson/test_dx1/anomaly"
  command_topic: "jetson/test_dx1/cmd"
tags:
  - "PLC_Collector_Actuator_1:AirBlower.Cmd[0]"
  - "PLC_Collector_Actuator_1:Gripper.Sensor[0]"
  - "PLC_Collector_Axis_Status_3:AxCV_VelDemVal"
  - "PLC_Collector_Axis_Status_3:AxX_VelDemVal"
  # 실제 DX1이 내보내는 "Collector명:신호명" 형식의 태그명으로 채우세요.
  # PLC 내부 카운터(PosTimestamp 등)는 분석 대상이 아니므로 여기 적지 않습니다.
resample_interval_ms: 50
window_size: 100
calibration:
  max_duration: "7d"
  min_samples: 10000
```

- [ ] **Step 6: 예시 config가 새 로더로 파싱되는지 확인 (Step 4의 `test_load_equipment_config_loads_shipped_example_config`가 이미 검증하지만, 수동으로도 한 번 확인)**

```bash
uv run python -c "from jetson_app.config import load_equipment_config; c = load_equipment_config('configs/test_dx1.example.yaml'); print(c)"
```

Expected: `EquipmentConfig(...)` 출력, 에러 없음

- [ ] **Step 7: Commit**

```bash
git add jetson_app/src/jetson_app/config.py jetson_app/tests/test_config.py jetson_app/configs/test_dx1.example.yaml
git commit -m "feat: config schema v2 (multi-topic, resample/window, calibration params)"
```

---

### Task 2: Buffer 기본 요소 (TagBuffer, SlidingWindow)

**Files:**
- Create: `jetson_app/src/jetson_app/buffer.py`
- Test: `jetson_app/tests/test_buffer.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Snapshot: values: dict[str, float | int | None]`
  - `class TagBuffer: def __init__(self, tags: tuple[str, ...]); def update(self, values: dict[str, float | int]) -> None; def snapshot(self) -> Snapshot`
  - `class SlidingWindow: def __init__(self, window_size: int); def push(self, snapshot: Snapshot) -> None; def is_full(self) -> bool; def to_list(self) -> list[Snapshot]`

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_buffer.py`:

```python
from jetson_app.buffer import Snapshot, SlidingWindow, TagBuffer


def test_tag_buffer_update_and_snapshot_returns_latest_values():
    buf = TagBuffer(tags=("a", "b"))

    buf.update({"a": 1, "b": 2})

    assert buf.snapshot() == Snapshot(values={"a": 1, "b": 2})


def test_tag_buffer_ignores_untracked_tags():
    buf = TagBuffer(tags=("a",))

    buf.update({"a": 1, "unrelated": 99})

    assert buf.snapshot() == Snapshot(values={"a": 1})


def test_tag_buffer_snapshot_returns_none_for_unseen_tags():
    buf = TagBuffer(tags=("a", "b"))

    buf.update({"a": 1})

    assert buf.snapshot() == Snapshot(values={"a": 1, "b": None})


def test_tag_buffer_snapshot_keeps_last_value_until_next_update():
    buf = TagBuffer(tags=("a",))

    buf.update({"a": 1})
    first = buf.snapshot()
    second = buf.snapshot()

    assert first == second == Snapshot(values={"a": 1})


def test_sliding_window_push_and_to_list_preserves_order():
    window = SlidingWindow(window_size=3)
    s1, s2 = Snapshot(values={"a": 1}), Snapshot(values={"a": 2})

    window.push(s1)
    window.push(s2)

    assert window.to_list() == [s1, s2]


def test_sliding_window_is_full_only_when_window_size_reached():
    window = SlidingWindow(window_size=2)

    assert window.is_full() is False
    window.push(Snapshot(values={"a": 1}))
    assert window.is_full() is False
    window.push(Snapshot(values={"a": 2}))
    assert window.is_full() is True


def test_sliding_window_drops_oldest_when_over_capacity():
    window = SlidingWindow(window_size=2)
    s1, s2, s3 = (
        Snapshot(values={"a": 1}),
        Snapshot(values={"a": 2}),
        Snapshot(values={"a": 3}),
    )

    window.push(s1)
    window.push(s2)
    window.push(s3)

    assert window.to_list() == [s2, s3]
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_buffer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jetson_app.buffer'`

- [ ] **Step 3: 구현 작성**

`jetson_app/src/jetson_app/buffer.py`:

```python
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Snapshot:
    values: dict[str, float | int | None]


class TagBuffer:
    """설정된 태그들의 최신값을 스레드세이프하게 유지하는 캐시.

    한 번 값이 들어오면, 다음 update()가 오기 전까지 snapshot()은 계속
    같은 값을 반환한다 (짧은 결측에 대한 ffill은 이 특성으로 자연히 만족된다).
    """

    def __init__(self, tags: tuple[str, ...]) -> None:
        self._tags = tags
        self._lock = threading.Lock()
        self._latest: dict[str, float | int] = {}

    def update(self, values: dict[str, float | int]) -> None:
        with self._lock:
            for tag, value in values.items():
                if tag in self._tags:
                    self._latest[tag] = value

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(values={tag: self._latest.get(tag) for tag in self._tags})


class SlidingWindow:
    """고정 크기 롤링 윈도우. 용량을 넘으면 가장 오래된 항목을 버린다."""

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._window_size = window_size
        self._items: deque[Snapshot] = deque(maxlen=window_size)

    def push(self, snapshot: Snapshot) -> None:
        self._items.append(snapshot)

    def is_full(self) -> bool:
        return len(self._items) == self._window_size

    def to_list(self) -> list[Snapshot]:
        return list(self._items)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_buffer.py -v
```

Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add jetson_app/src/jetson_app/buffer.py jetson_app/tests/test_buffer.py
git commit -m "feat: add TagBuffer and SlidingWindow"
```

---

### Task 3: 캘리브레이션 저장 & 상태 머신

**Files:**
- Create: `jetson_app/src/jetson_app/calibration.py`
- Test: `jetson_app/tests/test_calibration.py`

**Interfaces:**
- Consumes: `Snapshot`(Task 2, `buffer.py`)
- Produces:
  - `class CalibrationError(ValueError)`
  - `class CalibrationState(Enum): CALIBRATING = "CALIBRATING"; MONITORING = "MONITORING"`
  - `@dataclass(frozen=True) class CalibrationSample: timestamp: str; values: dict[str, float | int | None]`
  - `class CalibrationBufferWriter: def __init__(self, path: str | Path); def append(self, snapshot: Snapshot, timestamp: str) -> None; def read_all(self) -> list[CalibrationSample]; def count(self) -> int; def clear(self) -> None; def prune_older_than(self, cutoff: datetime) -> None`
  - `TrainFn = Callable[[list[CalibrationSample]], None]`
  - `class CalibrationManager: def __init__(self, buffer_writer: CalibrationBufferWriter, min_samples: int, max_duration: timedelta, train_fn: TrainFn); self.state: CalibrationState; def record_sample(self, snapshot: Snapshot, timestamp: str) -> None; def handle_train_command(self) -> None; def handle_recalibrate_command(self) -> None`

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_calibration.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from jetson_app.buffer import Snapshot
from jetson_app.calibration import (
    CalibrationBufferWriter,
    CalibrationError,
    CalibrationManager,
    CalibrationState,
)


def test_calibration_buffer_writer_append_and_read_all(tmp_path):
    writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")

    writer.append(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")
    writer.append(Snapshot(values={"a": 2}), "2026-08-04T00:00:01+00:00")

    samples = writer.read_all()
    assert len(samples) == 2
    assert samples[0].timestamp == "2026-08-04T00:00:00+00:00"
    assert samples[0].values == {"a": 1}
    assert samples[1].values == {"a": 2}


def test_calibration_buffer_writer_read_all_on_missing_file_returns_empty(tmp_path):
    writer = CalibrationBufferWriter(tmp_path / "does-not-exist.jsonl")

    assert writer.read_all() == []
    assert writer.count() == 0


def test_calibration_buffer_writer_clear_removes_data(tmp_path):
    writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")
    writer.append(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")

    writer.clear()

    assert writer.read_all() == []


def test_calibration_buffer_writer_prune_older_than_removes_old_samples(tmp_path):
    writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")
    writer.append(Snapshot(values={"a": 1}), "2026-08-01T00:00:00+00:00")
    writer.append(Snapshot(values={"a": 2}), "2026-08-04T00:00:00+00:00")

    writer.prune_older_than(datetime(2026, 8, 3, tzinfo=timezone.utc))

    samples = writer.read_all()
    assert len(samples) == 1
    assert samples[0].values == {"a": 2}


def _make_manager(tmp_path, min_samples=2, max_duration=timedelta(days=7), train_calls=None):
    writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")
    calls = train_calls if train_calls is not None else []

    def train_fn(samples):
        calls.append(samples)

    manager = CalibrationManager(
        buffer_writer=writer,
        min_samples=min_samples,
        max_duration=max_duration,
        train_fn=train_fn,
    )
    return manager, writer, calls


def test_calibration_manager_starts_in_calibrating_state(tmp_path):
    manager, _, _ = _make_manager(tmp_path)

    assert manager.state == CalibrationState.CALIBRATING


def test_calibration_manager_record_sample_writes_while_calibrating(tmp_path):
    manager, writer, _ = _make_manager(tmp_path)

    manager.record_sample(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")

    assert writer.count() == 1


def test_calibration_manager_train_command_below_min_samples_raises(tmp_path):
    manager, writer, calls = _make_manager(tmp_path, min_samples=5)
    manager.record_sample(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")

    with pytest.raises(CalibrationError):
        manager.handle_train_command()

    assert manager.state == CalibrationState.CALIBRATING
    assert calls == []


def test_calibration_manager_train_command_transitions_and_clears_buffer(tmp_path):
    manager, writer, calls = _make_manager(tmp_path, min_samples=2)
    manager.record_sample(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")
    manager.record_sample(Snapshot(values={"a": 2}), "2026-08-04T00:00:01+00:00")

    manager.handle_train_command()

    assert manager.state == CalibrationState.MONITORING
    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert writer.count() == 0


def test_calibration_manager_train_command_while_monitoring_raises(tmp_path):
    manager, _, _ = _make_manager(tmp_path, min_samples=1)
    manager.record_sample(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")
    manager.handle_train_command()

    with pytest.raises(CalibrationError):
        manager.handle_train_command()


def test_calibration_manager_recalibrate_clears_buffer_and_resets_state(tmp_path):
    manager, writer, _ = _make_manager(tmp_path, min_samples=1)
    manager.record_sample(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")
    manager.handle_train_command()
    assert manager.state == CalibrationState.MONITORING

    manager.handle_recalibrate_command()

    assert manager.state == CalibrationState.CALIBRATING
    assert writer.count() == 0


def test_calibration_manager_record_sample_ignored_while_monitoring(tmp_path):
    manager, writer, _ = _make_manager(tmp_path, min_samples=1)
    manager.record_sample(Snapshot(values={"a": 1}), "2026-08-04T00:00:00+00:00")
    manager.handle_train_command()

    manager.record_sample(Snapshot(values={"a": 2}), "2026-08-04T00:00:02+00:00")

    assert writer.count() == 0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_calibration.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jetson_app.calibration'`

- [ ] **Step 3: 구현 작성**

`jetson_app/src/jetson_app/calibration.py`:

```python
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable

from .buffer import Snapshot

_PRUNE_CHECK_INTERVAL = 10_000  # 대략 50ms 틱 기준 ~8분마다 오래된 데이터 정리 체크


class CalibrationError(ValueError):
    pass


class CalibrationState(Enum):
    CALIBRATING = "CALIBRATING"
    MONITORING = "MONITORING"


@dataclass(frozen=True)
class CalibrationSample:
    timestamp: str
    values: dict[str, float | int | None]


class CalibrationBufferWriter:
    """캘리브레이션 스냅샷을 디스크의 JSON Lines 파일에 순차 저장한다."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def append(self, snapshot: Snapshot, timestamp: str) -> None:
        line = json.dumps({"timestamp": timestamp, "values": snapshot.values})
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> list[CalibrationSample]:
        if not self._path.exists():
            return []
        samples = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                samples.append(
                    CalibrationSample(timestamp=data["timestamp"], values=data["values"])
                )
        return samples

    def count(self) -> int:
        return len(self.read_all())

    def clear(self) -> None:
        with self._lock:
            if self._path.exists():
                self._path.unlink()

    def prune_older_than(self, cutoff: datetime) -> None:
        kept = [
            s
            for s in self.read_all()
            if datetime.fromisoformat(s.timestamp) >= cutoff
        ]
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", encoding="utf-8") as f:
                for s in kept:
                    f.write(json.dumps({"timestamp": s.timestamp, "values": s.values}) + "\n")


TrainFn = Callable[[list[CalibrationSample]], None]


class CalibrationManager:
    """CALIBRATING/MONITORING 상태 전이와 캘리브레이션 데이터 수집을 담당."""

    def __init__(
        self,
        buffer_writer: CalibrationBufferWriter,
        min_samples: int,
        max_duration: timedelta,
        train_fn: TrainFn,
    ) -> None:
        self._buffer_writer = buffer_writer
        self._min_samples = min_samples
        self._max_duration = max_duration
        self._train_fn = train_fn
        self._lock = threading.Lock()
        self._tick_count = 0
        self.state = CalibrationState.CALIBRATING

    def record_sample(self, snapshot: Snapshot, timestamp: str) -> None:
        with self._lock:
            if self.state != CalibrationState.CALIBRATING:
                return
            self._buffer_writer.append(snapshot, timestamp)
            self._tick_count += 1
            if self._tick_count % _PRUNE_CHECK_INTERVAL == 0:
                cutoff = datetime.now(timezone.utc) - self._max_duration
                self._buffer_writer.prune_older_than(cutoff)

    def handle_train_command(self) -> None:
        with self._lock:
            if self.state != CalibrationState.CALIBRATING:
                raise CalibrationError(f"cannot train while in state {self.state.value}")
            samples = self._buffer_writer.read_all()
            if len(samples) < self._min_samples:
                raise CalibrationError(
                    f"not enough calibration samples: have {len(samples)}, need {self._min_samples}"
                )
            self._train_fn(samples)
            self._buffer_writer.clear()
            self.state = CalibrationState.MONITORING

    def handle_recalibrate_command(self) -> None:
        with self._lock:
            self._buffer_writer.clear()
            self.state = CalibrationState.CALIBRATING
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_calibration.py -v
```

Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add jetson_app/src/jetson_app/calibration.py jetson_app/tests/test_calibration.py
git commit -m "feat: add calibration buffer storage and state machine"
```

---

### Task 4: 주기 스냅샷 스케줄러

**Files:**
- Create: `jetson_app/src/jetson_app/scheduler.py`
- Test: `jetson_app/tests/test_scheduler.py`

**Interfaces:**
- Consumes: `TagBuffer`, `SlidingWindow`(Task 2), `CalibrationManager`(Task 3)
- Produces: `class PeriodicSnapshotter: def __init__(self, tag_buffer: TagBuffer, sliding_window: SlidingWindow, calibration_manager: CalibrationManager, interval_ms: int); def start(self) -> None; def stop(self) -> None`

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_scheduler.py`:

```python
import time
from datetime import timedelta

from jetson_app.buffer import SlidingWindow, TagBuffer
from jetson_app.calibration import CalibrationBufferWriter, CalibrationManager
from jetson_app.scheduler import PeriodicSnapshotter


def _make_calibration_manager(tmp_path):
    writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")
    return CalibrationManager(
        buffer_writer=writer,
        min_samples=1,
        max_duration=timedelta(days=7),
        train_fn=lambda samples: None,
    ), writer


def test_periodic_snapshotter_pushes_snapshots_to_window(tmp_path):
    tag_buffer = TagBuffer(tags=("a",))
    tag_buffer.update({"a": 1})
    window = SlidingWindow(window_size=100)
    calibration_manager, _ = _make_calibration_manager(tmp_path)

    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=window,
        calibration_manager=calibration_manager,
        interval_ms=10,
    )
    snapshotter.start()
    time.sleep(0.05)
    snapshotter.stop()

    assert len(window.to_list()) >= 2


def test_periodic_snapshotter_records_samples_into_calibration_buffer(tmp_path):
    tag_buffer = TagBuffer(tags=("a",))
    tag_buffer.update({"a": 1})
    window = SlidingWindow(window_size=100)
    calibration_manager, writer = _make_calibration_manager(tmp_path)

    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=window,
        calibration_manager=calibration_manager,
        interval_ms=10,
    )
    snapshotter.start()
    time.sleep(0.05)
    snapshotter.stop()

    assert writer.count() >= 2


def test_periodic_snapshotter_skips_completely_empty_snapshot(tmp_path):
    tag_buffer = TagBuffer(tags=("a",))  # update() 없이 -> 값이 전부 None
    window = SlidingWindow(window_size=100)
    calibration_manager, writer = _make_calibration_manager(tmp_path)

    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=window,
        calibration_manager=calibration_manager,
        interval_ms=10,
    )
    snapshotter.start()
    time.sleep(0.05)
    snapshotter.stop()

    assert window.to_list() == []
    assert writer.count() == 0


def test_periodic_snapshotter_stop_stops_the_background_thread(tmp_path):
    tag_buffer = TagBuffer(tags=("a",))
    tag_buffer.update({"a": 1})
    window = SlidingWindow(window_size=100)
    calibration_manager, _ = _make_calibration_manager(tmp_path)

    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=window,
        calibration_manager=calibration_manager,
        interval_ms=10,
    )
    snapshotter.start()
    time.sleep(0.02)
    snapshotter.stop()

    assert snapshotter._thread.is_alive() is False
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_scheduler.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jetson_app.scheduler'`

- [ ] **Step 3: 구현 작성**

`jetson_app/src/jetson_app/scheduler.py`:

```python
from __future__ import annotations

import threading
from datetime import datetime, timezone

from .buffer import SlidingWindow, TagBuffer
from .calibration import CalibrationManager


class PeriodicSnapshotter:
    """`interval_ms`마다 TagBuffer 스냅샷을 SlidingWindow에 밀어넣고,
    CalibrationManager에도 전달한다 (CALIBRATING 상태일 때만 실제로 저장됨).
    모든 태그가 None인(한 번도 값을 못 받은) 스냅샷은 건너뛴다.
    """

    def __init__(
        self,
        tag_buffer: TagBuffer,
        sliding_window: SlidingWindow,
        calibration_manager: CalibrationManager,
        interval_ms: int,
    ) -> None:
        self._tag_buffer = tag_buffer
        self._sliding_window = sliding_window
        self._calibration_manager = calibration_manager
        self._interval_seconds = interval_ms / 1000
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _tick(self) -> None:
        snapshot = self._tag_buffer.snapshot()
        if all(value is None for value in snapshot.values.values()):
            return
        self._sliding_window.push(snapshot)
        timestamp = datetime.now(timezone.utc).isoformat()
        self._calibration_manager.record_sample(snapshot, timestamp)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._tick()
            self._stop_event.wait(self._interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_scheduler.py -v
```

Expected: PASS (4 passed). 타이밍 기반 테스트라 드물게 느릴 수 있음 — 실패하면 한 번 재실행해서 확인.

- [ ] **Step 5: Commit**

```bash
git add jetson_app/src/jetson_app/scheduler.py jetson_app/tests/test_scheduler.py
git commit -m "feat: add periodic snapshot scheduler"
```

---

### Task 5: MQTT 명령(train/recalibrate) 구독자

**Files:**
- Create: `jetson_app/src/jetson_app/command_subscriber.py`
- Test: `jetson_app/tests/test_command_subscriber.py`

**Interfaces:**
- Consumes: `CalibrationManager`, `CalibrationError`(Task 3)
- Produces:
  - `def parse_command(payload: bytes) -> str | None` — `{"command": "train"}` 또는 `{"command": "recalibrate"}`만 유효, 그 외 전부 `None`
  - `class CommandSubscriber: def __init__(self, command_topic: str, calibration_manager: CalibrationManager); def attach(self, client) -> None`

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_command_subscriber.py`:

```python
from types import SimpleNamespace

from jetson_app.command_subscriber import CommandSubscriber, parse_command


def test_parse_command_valid_train():
    assert parse_command(b'{"command": "train"}') == "train"


def test_parse_command_valid_recalibrate():
    assert parse_command(b'{"command": "recalibrate"}') == "recalibrate"


def test_parse_command_invalid_json_returns_none():
    assert parse_command(b"not json") is None


def test_parse_command_unknown_command_returns_none():
    assert parse_command(b'{"command": "shutdown"}') is None


def test_parse_command_missing_command_key_returns_none():
    assert parse_command(b'{"foo": "bar"}') is None


class _FakeCalibrationManager:
    def __init__(self):
        self.train_calls = 0
        self.recalibrate_calls = 0
        self.raise_on_train = None

    def handle_train_command(self):
        self.train_calls += 1
        if self.raise_on_train is not None:
            raise self.raise_on_train

    def handle_recalibrate_command(self):
        self.recalibrate_calls += 1


def test_command_subscriber_handles_train_message():
    manager = _FakeCalibrationManager()
    subscriber = CommandSubscriber(command_topic="jetson/x/cmd", calibration_manager=manager)
    msg = SimpleNamespace(payload=b'{"command": "train"}')

    subscriber._handle_command_message(None, None, msg)

    assert manager.train_calls == 1
    assert manager.recalibrate_calls == 0


def test_command_subscriber_handles_recalibrate_message():
    manager = _FakeCalibrationManager()
    subscriber = CommandSubscriber(command_topic="jetson/x/cmd", calibration_manager=manager)
    msg = SimpleNamespace(payload=b'{"command": "recalibrate"}')

    subscriber._handle_command_message(None, None, msg)

    assert manager.recalibrate_calls == 1
    assert manager.train_calls == 0


def test_command_subscriber_swallows_calibration_error():
    from jetson_app.calibration import CalibrationError

    manager = _FakeCalibrationManager()
    manager.raise_on_train = CalibrationError("not enough samples")
    subscriber = CommandSubscriber(command_topic="jetson/x/cmd", calibration_manager=manager)
    msg = SimpleNamespace(payload=b'{"command": "train"}')

    subscriber._handle_command_message(None, None, msg)  # 예외가 밖으로 나오면 테스트 실패

    assert manager.train_calls == 1


def test_command_subscriber_ignores_unparseable_payload():
    manager = _FakeCalibrationManager()
    subscriber = CommandSubscriber(command_topic="jetson/x/cmd", calibration_manager=manager)
    msg = SimpleNamespace(payload=b"garbage")

    subscriber._handle_command_message(None, None, msg)

    assert manager.train_calls == 0
    assert manager.recalibrate_calls == 0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_command_subscriber.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jetson_app.command_subscriber'`

- [ ] **Step 3: 구현 작성**

`jetson_app/src/jetson_app/command_subscriber.py`:

```python
from __future__ import annotations

import json

from .calibration import CalibrationError, CalibrationManager

_VALID_COMMANDS = {"train", "recalibrate"}


def parse_command(payload: bytes) -> str | None:
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    command = data.get("command")
    if command not in _VALID_COMMANDS:
        return None
    return command


class CommandSubscriber:
    """MQTT의 설비 command_topic에서 train/recalibrate 명령을 받아
    CalibrationManager로 전달한다."""

    def __init__(self, command_topic: str, calibration_manager: CalibrationManager) -> None:
        self._command_topic = command_topic
        self._calibration_manager = calibration_manager

    def attach(self, client) -> None:
        client.message_callback_add(self._command_topic, self._handle_command_message)

    def _handle_command_message(self, client, userdata, msg) -> None:
        command = parse_command(msg.payload)
        if command is None:
            print(f"알 수 없는 명령 페이로드 무시: {msg.payload!r}")
            return
        try:
            if command == "train":
                self._calibration_manager.handle_train_command()
                print("학습 완료 — MONITORING 상태로 전환")
            else:
                self._calibration_manager.handle_recalibrate_command()
                print("재캘리브레이션 — CALIBRATING 상태로 재시작")
        except CalibrationError as e:
            print(f"명령 처리 실패: {e}")
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_command_subscriber.py -v
```

Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add jetson_app/src/jetson_app/command_subscriber.py jetson_app/tests/test_command_subscriber.py
git commit -m "feat: add MQTT train/recalibrate command subscriber"
```

---

### Task 6: MqttRecordSubscriber 다중 토픽 지원

**Files:**
- Modify: `jetson_app/src/jetson_app/mqtt_subscriber.py`
- Modify: `jetson_app/tests/test_mqtt_subscriber.py`

**Interfaces:**
- Consumes: `EquipmentConfig.subscribe_topics`(복수, Task 1), `EquipmentConfig.command_topic`(Task 1)
- Produces: `MqttRecordSubscriber.client` (읽기전용 property, `paho.mqtt.client.Client` 반환) — Task 7에서 `CommandSubscriber.attach()`에 넘길 때 사용

- [ ] **Step 1: 실패하는 테스트 작성 (기존 파일에 추가)**

`jetson_app/tests/test_mqtt_subscriber.py`의 기존 임포트 줄 바로 아래에 추가:

```python
from unittest.mock import MagicMock

from jetson_app.config import CalibrationConfig, EquipmentConfig
from datetime import timedelta


def _make_config(subscribe_topics, command_topic="jetson/x/cmd"):
    return EquipmentConfig(
        equipment_id="x",
        subscribe_topics=subscribe_topics,
        publish_topic="jetson/x/anomaly",
        command_topic=command_topic,
        tags=("a",),
        resample_interval_ms=50,
        window_size=100,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=10),
    )
```

파일 맨 끝에 추가:

```python
def test_handle_connect_subscribes_to_all_configured_topics():
    config = _make_config(subscribe_topics=("topic/a", "topic/b"))
    subscriber = MqttRecordSubscriber(config, on_record=lambda r: None)
    fake_client = MagicMock()

    subscriber._handle_connect(fake_client, None, None, 0)

    subscribed_topics = {call.args[0] for call in fake_client.subscribe.call_args_list}
    assert subscribed_topics == {"topic/a", "topic/b", "jetson/x/cmd"}


def test_client_property_exposes_underlying_paho_client():
    config = _make_config(subscribe_topics=("topic/a",))
    subscriber = MqttRecordSubscriber(config, on_record=lambda r: None)

    assert subscriber.client is subscriber._client
```

기존의 `test_handle_connect_subscribes_on_success`/`test_handle_connect_does_not_subscribe_on_failure` 테스트가 `EquipmentConfig(..., subscribe_topic="...", ...)`(단수 필드)를 직접 생성하고 있다면, 그 부분을 위 `_make_config()` 헬퍼를 쓰도록 바꾸고 assertion도 `fake_client.subscribe.assert_called_once_with(config.subscribe_topic)` 대신 아래처럼 바꾼다:

```python
def test_handle_connect_subscribes_on_success():
    config = _make_config(subscribe_topics=("topic/a",))
    subscriber = MqttRecordSubscriber(config, on_record=lambda r: None)
    fake_client = MagicMock()

    subscriber._handle_connect(fake_client, None, None, 0)

    subscribed_topics = {call.args[0] for call in fake_client.subscribe.call_args_list}
    assert "topic/a" in subscribed_topics


def test_handle_connect_does_not_subscribe_on_failure():
    config = _make_config(subscribe_topics=("topic/a",))
    subscriber = MqttRecordSubscriber(config, on_record=lambda r: None)
    fake_client = MagicMock()

    subscriber._handle_connect(fake_client, None, None, 5)

    fake_client.subscribe.assert_not_called()
```

(이 두 테스트가 이미 이런 형태로 존재한다면 그대로 두고, 옛 `EquipmentConfig(subscribe_topic=...)` 형태로 되어 있다면 위처럼 고친다. `test_handle_message_delivers_parsed_records_to_on_record` 등 `EquipmentConfig`를 직접 생성하는 다른 기존 테스트도 전부 `_make_config()` 헬퍼를 쓰도록 동일하게 고친다.)

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_mqtt_subscriber.py -v
```

Expected: FAIL — 기존 테스트들이 `EquipmentConfig(subscribe_topic=...)` 시그니처 불일치로 `TypeError`, 새 테스트는 `AttributeError: 'MqttRecordSubscriber' object has no attribute 'client'`

- [ ] **Step 3: 구현 수정**

`jetson_app/src/jetson_app/mqtt_subscriber.py`의 `MqttRecordSubscriber` 클래스를 아래로 교체 (파일 상단의 `Record`/`parse_and_filter_records`는 그대로 유지):

```python
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

    @property
    def client(self) -> mqtt.Client:
        return self._client

    def connect(self, host: str, port: int = 1883) -> None:
        self._client.connect(host, port)

    def loop_forever(self) -> None:
        self._client.loop_forever()

    def _handle_connect(self, client, userdata, flags, rc):
        if rc != 0:
            print(
                f"[{self._config.equipment_id}] MQTT 연결 실패 (rc={rc}) "
                "— 브로커 인증/ACL 설정을 확인하세요"
            )
            return
        for topic in self._config.subscribe_topics:
            client.subscribe(topic)
        client.subscribe(self._config.command_topic)

    def _handle_message(self, client, userdata, msg):
        for record in parse_and_filter_records(msg.payload, self._config.tags):
            self._on_record(record)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_mqtt_subscriber.py -v
```

Expected: PASS (전체)

- [ ] **Step 5: Commit**

```bash
git add jetson_app/src/jetson_app/mqtt_subscriber.py jetson_app/tests/test_mqtt_subscriber.py
git commit -m "feat: multi-topic subscribe and command_topic wiring in MqttRecordSubscriber"
```

---

### Task 7: Pipeline 오케스트레이터 & CLI 통합

**Files:**
- Create: `jetson_app/src/jetson_app/pipeline.py`
- Modify: `jetson_app/src/jetson_app/subscriber_cli.py`
- Test: `jetson_app/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `EquipmentConfig`(Task 1), `TagBuffer`/`SlidingWindow`(Task 2), `CalibrationBufferWriter`/`CalibrationManager`/`TrainFn`(Task 3), `PeriodicSnapshotter`(Task 4), `CommandSubscriber`(Task 5), `MqttRecordSubscriber`(Task 6)
- Produces:
  - `@dataclass(frozen=True) class Pipeline: config: EquipmentConfig; tag_buffer: TagBuffer; sliding_window: SlidingWindow; calibration_manager: CalibrationManager; snapshotter: PeriodicSnapshotter; mqtt_subscriber: MqttRecordSubscriber; command_subscriber: CommandSubscriber`
  - `def build_pipeline(config: EquipmentConfig, calibration_dir: str | Path, train_fn: TrainFn) -> Pipeline`

- [ ] **Step 1: 실패하는 테스트 작성**

`jetson_app/tests/test_pipeline.py`:

```python
from datetime import timedelta

from jetson_app.calibration import CalibrationState
from jetson_app.config import CalibrationConfig, EquipmentConfig
from jetson_app.pipeline import build_pipeline


def _make_config(tmp_path):
    return EquipmentConfig(
        equipment_id="test_dx1",
        subscribe_topics=("dx1/test_dx1/actuator_1",),
        publish_topic="jetson/test_dx1/anomaly",
        command_topic="jetson/test_dx1/cmd",
        tags=("PLC_Collector_Actuator_1:AirBlower.Cmd[0]",),
        resample_interval_ms=50,
        window_size=10,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=10),
    )


def test_build_pipeline_wires_all_components(tmp_path):
    config = _make_config(tmp_path)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        train_fn=lambda samples: None,
    )

    assert pipeline.config is config
    assert pipeline.calibration_manager.state == CalibrationState.CALIBRATING
    assert pipeline.mqtt_subscriber.client is not None


def test_build_pipeline_command_subscriber_shares_calibration_manager(tmp_path):
    config = _make_config(tmp_path)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        train_fn=lambda samples: None,
    )

    assert pipeline.command_subscriber._calibration_manager is pipeline.calibration_manager


def test_build_pipeline_on_record_updates_tag_buffer(tmp_path):
    config = _make_config(tmp_path)
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        train_fn=lambda samples: None,
    )

    from jetson_app.mqtt_subscriber import Record

    pipeline.mqtt_subscriber._on_record(
        Record(
            timestamp="2026-08-04T00:00:00+00:00",
            values={"PLC_Collector_Actuator_1:AirBlower.Cmd[0]": 1},
        )
    )

    snapshot = pipeline.tag_buffer.snapshot()
    assert snapshot.values["PLC_Collector_Actuator_1:AirBlower.Cmd[0]"] == 1
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jetson_app.pipeline'`

- [ ] **Step 3: 구현 작성**

`jetson_app/src/jetson_app/pipeline.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .buffer import SlidingWindow, TagBuffer
from .calibration import CalibrationBufferWriter, CalibrationManager, TrainFn
from .command_subscriber import CommandSubscriber
from .config import EquipmentConfig
from .mqtt_subscriber import MqttRecordSubscriber, Record
from .scheduler import PeriodicSnapshotter


@dataclass(frozen=True)
class Pipeline:
    config: EquipmentConfig
    tag_buffer: TagBuffer
    sliding_window: SlidingWindow
    calibration_manager: CalibrationManager
    snapshotter: PeriodicSnapshotter
    mqtt_subscriber: MqttRecordSubscriber
    command_subscriber: CommandSubscriber


def build_pipeline(
    config: EquipmentConfig,
    calibration_dir: str | Path,
    train_fn: TrainFn,
) -> Pipeline:
    tag_buffer = TagBuffer(config.tags)
    sliding_window = SlidingWindow(config.window_size)

    buffer_path = Path(calibration_dir) / f"{config.equipment_id}.jsonl"
    buffer_writer = CalibrationBufferWriter(buffer_path)
    calibration_manager = CalibrationManager(
        buffer_writer=buffer_writer,
        min_samples=config.calibration.min_samples,
        max_duration=config.calibration.max_duration,
        train_fn=train_fn,
    )

    snapshotter = PeriodicSnapshotter(
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        interval_ms=config.resample_interval_ms,
    )

    def on_record(record: Record) -> None:
        tag_buffer.update(record.values)

    mqtt_subscriber = MqttRecordSubscriber(config, on_record=on_record)
    command_subscriber = CommandSubscriber(config.command_topic, calibration_manager)
    command_subscriber.attach(mqtt_subscriber.client)

    return Pipeline(
        config=config,
        tag_buffer=tag_buffer,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        snapshotter=snapshotter,
        mqtt_subscriber=mqtt_subscriber,
        command_subscriber=command_subscriber,
    )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: PASS (3 passed)

- [ ] **Step 5: CLI를 Pipeline 사용하도록 재작성**

`jetson_app/src/jetson_app/subscriber_cli.py` 전체를 아래로 교체:

```python
from __future__ import annotations

import argparse
import sys

from .calibration import CalibrationSample
from .config import ConfigError, load_equipment_config
from .pipeline import build_pipeline


def _placeholder_train_fn(samples: list[CalibrationSample]) -> None:
    print(
        f"[학습] {len(samples)}개 샘플로 학습을 실행합니다 "
        "(모델 미연결 — 다음 계획에서 실제 GRU 학습으로 교체 예정)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DX1 -> Jetson MQTT 데이터 파이프라인")
    parser.add_argument("--config", required=True, help="설비 config YAML 경로")
    parser.add_argument("--host", required=True, help="MQTT 브로커 호스트 (Jetson 자신의 IP 또는 localhost)")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument(
        "--calibration-dir",
        default="calibration_data",
        help="캘리브레이션 버퍼 파일을 저장할 디렉터리 (기본: calibration_data)",
    )
    args = parser.parse_args()

    try:
        config = load_equipment_config(args.config)
        pipeline = build_pipeline(
            config=config,
            calibration_dir=args.calibration_dir,
            train_fn=_placeholder_train_fn,
        )
        pipeline.mqtt_subscriber.connect(args.host, args.port)
    except (ConfigError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)

    pipeline.snapshotter.start()
    print(
        f"[{config.equipment_id}] {len(config.subscribe_topics)}개 토픽 구독 시작 "
        f"({args.host}:{args.port}), 캘리브레이션 데이터: {args.calibration_dir}"
    )
    try:
        pipeline.mqtt_subscriber.loop_forever()
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        pipeline.snapshotter.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 로컬(Windows)에서 import 에러 없이 로드되는지 확인**

```bash
uv run python -c "from jetson_app.subscriber_cli import main; print('ok')"
```

Expected: `ok` 출력, 에러 없음

- [ ] **Step 7: 전체 테스트 스위트 실행**

```bash
uv run pytest -v
```

Expected: 전체 PASS (Task 1~7의 모든 테스트 + 기존 Task 2~5 테스트 포함)

- [ ] **Step 8: Commit**

```bash
git add jetson_app/src/jetson_app/pipeline.py jetson_app/src/jetson_app/subscriber_cli.py jetson_app/tests/test_pipeline.py
git commit -m "feat: wire pipeline components together and update CLI"
```

---

## Self-Review 결과

**스펙 커버리지**: 상위 스펙 문서(`2026-08-04-jetson-anomaly-inference-pipeline-design.md`) 1절(Config 개정)은 Task 1, 2절(Tag Buffer)은 Task 2+4, 3절(캘리브레이션 데이터 저장)은 Task 3, 4절(캘리브레이션 트리거)은 Task 3+5가 구현한다. 5절(모델)과 6절(실시간 이상 점수)은 이 계획의 명시적 범위 밖(다음 계획에서 `TrainFn`/`CalibrationSample`을 소비하는 실제 GRU 학습 로직으로 구현). 7절(에러 처리) 중 "학습 명령 min_samples 미달 거부", "캘리브레이션 버퍼 손상"(파일 없음 → `read_all()`이 빈 리스트 반환하는 것으로 자연히 처리), "다중 토픽 중 일부만 끊김"(태그별 ffill은 `TagBuffer`가 토픽 구분 없이 태그명으로만 동작하므로 자동 처리)은 Task 3/6에서 커버됨.

**Placeholder 스캔**: 모든 코드 스텝에 실제 동작하는 코드/커맨드 포함. `_placeholder_train_fn`은 이름에 "placeholder"가 들어가지만 스펙상 의도된 자리표시자(다음 계획에서 교체 예정이라고 CLI 출력에도 명시)이지 "TODO" 같은 미완성 표시가 아님.

**타입 일관성**: `EquipmentConfig`(Task 1: `subscribe_topics`, `command_topic`, `resample_interval_ms`, `window_size`, `calibration.min_samples`, `calibration.max_duration`) → `TagBuffer`/`SlidingWindow`(Task 2: `Snapshot.values`) → `CalibrationManager`(Task 3: `record_sample(snapshot, timestamp)`, `TrainFn = Callable[[list[CalibrationSample]], None]`) → `PeriodicSnapshotter`(Task 4: 동일 시그니처로 소비) → `CommandSubscriber`(Task 5: `CalibrationManager.handle_train_command`/`handle_recalibrate_command`) → `MqttRecordSubscriber.client`(Task 6) → `build_pipeline`(Task 7: 전부를 정확한 이름으로 조립) 까지 필드명과 시그니처가 각 Task의 Interfaces 블록과 실제 코드에서 일치함을 확인.
