# 이벤트 시간 기반 리샘플러 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DX1이 여러 record를 묶어 보내는 배치 메시지와 임의의 취득 주기를, record의 `timestamp`(이벤트 시각) 기준으로 `resample_interval_ms` 격자에 정확히 배치해 파이프라인에 공급한다.

**Architecture:** 파서가 record의 timestamp를 epoch ns로 해석해 `Record`에 싣고, 새 `EventTimeResampler`가 값을 격자 칸에 배치하며 워터마크(`max_lateness_ms`)로 칸을 확정한다. 확정된 칸(`ResampledStep`)은 큐를 거쳐 새 `SnapshotProcessor` 워커가 기존 틱 로직(점수 계산 → 윈도우 → 캘리브레이션 기록)으로 처리한다. 학습 모델 artifact에 격자 값을 저장해 config와 다르면 CALIBRATING으로 폴백한다. Jetson 시계는 쓰지 않는다.

**Tech Stack:** 기존 `jetson_app` 패키지(Python 3.8, uv, paho-mqtt<2, torch<2.4, pytest 7). 새 의존성 없음.

**Spec:** [`docs/superpowers/specs/2026-10-01-event-time-resampler-design.md`](../specs/2026-10-01-event-time-resampler-design.md)

## Global Constraints

- **Python 3.8 호환**: 새/수정 파일 최상단에 `from __future__ import annotations`. 애노테이션 위치의 `X | None`, `dict[...]`, `list[...]`은 안전(지연 평가). 런타임에 평가되는 위치(`queue.Queue[...]`, 모듈 최상위 별칭 등)에는 쓰지 않는다. `datetime.fromisoformat`은 나노초와 콜론 없는 오프셋(`+0000`)을 읽지 못한다 — DX1 타임스탬프는 반드시 `parse_dx1_timestamp`로 해석한다.
- **시간 기준은 이벤트 시각**: 리샘플러·처리 스레드는 `datetime.now()`/`time.sleep`에 의존하지 않는다(테스트가 sleep 없이 결정적이어야 한다).
- `resample_interval_ms`는 config에 직접 적는다(자동 감지 없음). `max_lateness_ms`는 신규 선택 항목, 기본값 **2000**, 0 이상 정수.
- 학습 모델에 `resample_interval_ms`를 저장하고, 이 값이 config와 다르거나 없으면(0) `CALIBRATING`으로 폴백한다.
- 이 저장소 규칙: `jetson_app/src/`는 Claude가 만들어 git으로 관리하는 코드라 제자리 수정(`_revb` 복사본 불필요). 사용자가 가져온 `Dorco_*.py`와 `Data/`, `JSONData/`, `Model/`은 건드리지 않는다.
- 커밋 메시지 끝에 `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` 줄을 붙인다(아래 커밋 명령의 두 번째 `-m`). 사용자가 커밋을 원하지 않는다고 하면 커밋 단계만 건너뛴다.
- 모든 명령은 `C:\WORK\10. Jetson\jetson_app`에서 `uv run pytest ...`로 실행한다. 각 Task가 끝날 때 **전체 테스트(`uv run pytest -q`)가 통과**해야 한다.

## Review Focus

스펙이 암시하지만 정상 경로 테스트가 놓치기 쉬운 입력/상황이다. 각 항목의 테스트는 지정한 Task에 이미 들어 있다.

1. **한 칸 안에서 record가 시각 역순으로 도착** → 도착 순서가 아니라 이벤트 시각이 가장 늦은 값이 이긴다. (Task 4 `test_records_inside_one_bucket_use_latest_event_time_regardless_of_arrival_order`)
2. **QoS 1 재전송으로 같은 메시지가 두 번 도착** → 스텝이 늘거나 값이 바뀌지 않고, 칸이 확정된 뒤의 재전송은 늦은 record로 폐기된다. (Task 4 `test_duplicate_delivery_is_idempotent`)
3. **DX1/PLC 시계 점프(하루 앞/뒤)** → 앞으로 점프하면 ffill 칸을 수만 개 만들지 않고 윈도우를 리셋, 뒤로 점프한 record는 늦은 record로 폐기된다. (Task 4 `test_one_day_forward_jump_stays_bounded`, `test_record_far_in_the_past_after_emission_is_dropped`)
4. **한 토픽의 태그만 도착(다른 토픽이 아직/영영 안 옴)** → 지연 허용 안에서는 한 스냅샷으로 정렬되고, 끝내 안 오는 태그는 `None`인 스냅샷이 나간다(스텝을 버리지 않는다). 학습/추론은 `None` 포함 윈도우를 기존 정책대로 건너뛴다. (Task 4 `test_multi_topic_values_align_within_lateness`, `test_partial_snapshot_keeps_none_for_unseen_tag_and_all_none_is_skipped`)
5. **처리 스레드가 못 따라가는 폭주(작은 격자 + 긴 배치, 재접속 직후 밀린 메시지)** → 큐가 무한히 자라지 않고 오래된 스텝부터 버리며 로그를 남긴다. (Task 5 `test_submit_drops_oldest_when_queue_is_full`)

**알려진 한계(사용자와 합의, 이번 범위 밖):** (a) record 값이 숫자가 아닌 경우(문자열/`null`/`true`)는 아직 걸러내지 않는다 — 실제 토픽 샘플을 받은 뒤 보강한다. (b) `CALIBRATING` 도중 `resample_interval_ms`를 바꾸면 이전 버퍼와 섞이므로 `recalibrate`가 필요하다(README에 명시, Task 8). (c) `max_lateness_ms`는 토픽 중 가장 긴 배치 간격보다 크게 잡아야 한다(README에 명시).

## File Structure

| 파일 | 작업 | 책임 |
|---|---|---|
| `src/jetson_app/timeparse.py` | 신규 | DX1 타임스탬프 → epoch ns, epoch ns → ISO 문자열 |
| `src/jetson_app/droplog.py` | 신규 | 폐기 개수 누적 + 속도 제한 로그 |
| `src/jetson_app/mqtt_subscriber.py` | 수정 | `Record.epoch_ns`, 파서의 타임스탬프 해석/폐기 |
| `src/jetson_app/config.py` | 수정 | `max_lateness_ms` |
| `src/jetson_app/resampler.py` | 신규 | `EventTimeResampler`, `ResampledStep` (스펙 2절) |
| `src/jetson_app/snapshot_processor.py` | 신규 | `SnapshotProcessor` 큐 소비 워커 (스펙 3절) |
| `src/jetson_app/buffer.py` | 수정 | `SlidingWindow.clear()` 추가, `TagBuffer` 제거 |
| `src/jetson_app/training.py` | 수정 | artifact에 `resample_interval_ms` 저장/로드 |
| `src/jetson_app/pipeline.py` | 수정 | 리샘플러 + 처리 워커 연결, 격자 호환성 검사 |
| `src/jetson_app/subscriber_cli.py` | 수정 | `make_train_fn`에 격자 값 전달 |
| `src/jetson_app/scheduler.py` | 삭제 | 시계 기반 `PeriodicSnapshotter` (워커가 대체) |
| `README.md`, `configs/test_dx1.example.yaml`, 스펙 문서 | 수정 | 설정/운영 안내와 상태 갱신 |
| `tests/…` | 신규/수정 | 각 Task에 명시 |

> **스펙과의 차이(동작은 동일)**: 스펙은 `buffer.py`/`scheduler.py`를 수정하는 것으로 적었지만, 단일 책임 파일로 나눠 `resampler.py`/`snapshot_processor.py`를 새로 만들고 기존 `TagBuffer`/`scheduler.py`는 마지막(Task 7)에 제거한다. 이렇게 하면 Task마다 전체 테스트가 통과한다. 처리 스레드 클래스는 `PeriodicSnapshotter` → `SnapshotProcessor`로 이름이 바뀌지만 `Pipeline.snapshotter` 필드명과 `start()`/`stop()` 사용법은 그대로다.

---

### Task 1: 타임스탬프 파싱과 폐기 로그 헬퍼

**Files:**
- Create: `src/jetson_app/timeparse.py`, `src/jetson_app/droplog.py`
- Test: `tests/test_timeparse.py`, `tests/test_droplog.py`

**Interfaces:**
- Produces:
  - `parse_dx1_timestamp(text: object) -> int | None` — epoch 나노초. 형식 오류/없는 날짜/문자열이 아님 → `None`.
  - `format_epoch_ns(epoch_ns: int) -> str` — `datetime.fromisoformat`(3.8)이 읽는 UTC ISO 8601(마이크로초, `+00:00`).
  - `DropCounter(label: str, log_every: int = 100)` — `.add(n: int = 1) -> None`, `.total: int`. 첫 발생과 이후 `log_every` 경계를 넘을 때만 `print`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_timeparse.py`:

```python
import pytest

from jetson_app.timeparse import format_epoch_ns, parse_dx1_timestamp


def test_parses_dx1_nanosecond_timestamp_with_compact_offset():
    # 2026-10-01T00:47:01Z = 1790815621 초
    assert parse_dx1_timestamp("2026-10-01T00:47:01.718651520+0000") == 1790815621_718651520


def test_parses_z_suffix_and_colon_offset():
    base = parse_dx1_timestamp("2026-10-01T00:00:00Z")
    assert parse_dx1_timestamp("2026-10-01T00:00:00+00:00") == base
    # +09:00 이면 같은 순간이 9시간 이른 UTC 시각이다
    assert parse_dx1_timestamp("2026-10-01T09:00:00+09:00") == base
    assert parse_dx1_timestamp("2026-09-30T19:00:00-0500") == base


@pytest.mark.parametrize("fraction,expected_ns", [("", 0), (".5", 500_000_000), (".000001", 1000), (".123456789", 123456789)])
def test_fraction_digits_zero_to_nine(fraction, expected_ns):
    base = parse_dx1_timestamp("2026-10-01T00:00:00Z")
    assert parse_dx1_timestamp("2026-10-01T00:00:00%sZ" % fraction) == base + expected_ns


@pytest.mark.parametrize(
    "bad",
    ["t1", "", "2026-10-01", "2026-10-01T00:00:00", "2026-13-01T00:00:00Z", "2026-10-01T00:00:60Z", None, 123, "2026-10-01T00:00:00.1234567891Z"],
)
def test_invalid_timestamps_return_none(bad):
    assert parse_dx1_timestamp(bad) is None


def test_format_epoch_ns_round_trips_through_fromisoformat():
    from datetime import datetime

    epoch_ns = parse_dx1_timestamp("2026-10-01T00:47:01.718651520+0000")
    text = format_epoch_ns(epoch_ns)
    assert text == "2026-10-01T00:47:01.718651+00:00"
    assert datetime.fromisoformat(text).tzinfo is not None
    assert format_epoch_ns(parse_dx1_timestamp("2026-10-01T00:00:00Z")) == "2026-10-01T00:00:00+00:00"
```

`tests/test_droplog.py`:

```python
from jetson_app.droplog import DropCounter


def test_drop_counter_logs_first_drop_then_every_n(capsys):
    counter = DropCounter("label", log_every=3)

    counter.add()  # 1: 첫 발생은 항상 출력
    counter.add()  # 2
    counter.add()  # 3: 3의 배수를 처음 넘는 순간 출력
    counter.add()  # 4

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["[label] 누적 1건", "[label] 누적 3건"]
    assert counter.total == 4


def test_drop_counter_add_many_at_once_logs_when_crossing_a_boundary(capsys):
    counter = DropCounter("label", log_every=100)

    counter.add(250)

    assert capsys.readouterr().out.strip() == "[label] 누적 250건"
    assert counter.total == 250
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_timeparse.py tests/test_droplog.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'jetson_app.timeparse'` (및 `droplog`).

- [ ] **Step 3: 구현**

`src/jetson_app/timeparse.py`:

```python
from __future__ import annotations

import re
from datetime import datetime, timezone

_TS_PATTERN = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,9}))?"
    r"(Z|[+-]\d{2}:?\d{2})$"
)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def parse_dx1_timestamp(text: object) -> int | None:
    """DX1(SpeeDBee Synapse)이 내보내는 ISO 8601 타임스탬프를 epoch 나노초(int)로 바꾼다.

    DX1 형식(`2026-10-01T00:47:01.718651520+0000`: 나노초 9자리, 콜론 없는 오프셋)은
    Python 3.8의 `datetime.fromisoformat`이 읽지 못해 직접 파싱한다. 형식이 맞지 않거나
    존재하지 않는 날짜면 None을 반환한다(호출부가 해당 record를 버린다).
    """
    if not isinstance(text, str):
        return None
    match = _TS_PATTERN.match(text.strip())
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(g) for g in match.groups()[:6])
    fraction = match.group(7) or ""
    offset_text = match.group(8)
    try:
        moment = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None
    if offset_text == "Z":
        offset_seconds = 0
    else:
        digits = offset_text[1:].replace(":", "")
        offset_seconds = int(digits[:2]) * 3600 + int(digits[2:]) * 60
        if offset_text[0] == "-":
            offset_seconds = -offset_seconds
    delta = moment - _EPOCH
    epoch_seconds = delta.days * 86400 + delta.seconds - offset_seconds
    return epoch_seconds * 1_000_000_000 + int(fraction.ljust(9, "0"))


def format_epoch_ns(epoch_ns: int) -> str:
    """epoch 나노초를 Python 3.8의 `datetime.fromisoformat`이 읽을 수 있는
    UTC ISO 8601 문자열(마이크로초 정밀도, `+00:00`)로 바꾼다."""
    seconds, nanos = divmod(epoch_ns, 1_000_000_000)
    moment = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=nanos // 1000)
    return moment.isoformat()
```

`src/jetson_app/droplog.py`:

```python
from __future__ import annotations


class DropCounter:
    """버려진 항목의 누적 개수를 세고, 메시지마다 로그가 쏟아지지 않도록
    첫 발생과 이후 `log_every`건 단위로만 한 줄씩 출력한다."""

    def __init__(self, label: str, log_every: int = 100) -> None:
        self._label = label
        self._log_every = log_every
        self.total = 0

    def add(self, n: int = 1) -> None:
        before = self.total
        self.total += n
        if before == 0 or before // self._log_every != self.total // self._log_every:
            print(f"[{self._label}] 누적 {self.total}건")
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_timeparse.py tests/test_droplog.py -q`
Expected: PASS (모두 통과)

- [ ] **Step 5: 커밋**

```bash
git add jetson_app/src/jetson_app/timeparse.py jetson_app/src/jetson_app/droplog.py jetson_app/tests/test_timeparse.py jetson_app/tests/test_droplog.py
git commit -m "feat: add DX1 timestamp parser and rate-limited drop counter" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `Record`에 이벤트 시각 추가 (파서)

**Files:**
- Modify: `src/jetson_app/mqtt_subscriber.py` (전체 교체)
- Modify: `tests/test_mqtt_subscriber.py` (전체 교체), `tests/test_pipeline.py` (`Record(...)` 한 곳)

**Interfaces:**
- Consumes: `parse_dx1_timestamp`, `DropCounter` (Task 1)
- Produces:
  - `Record(timestamp: str, values: dict[str, float], epoch_ns: int)` — `epoch_ns`는 필수 필드(기본값 없음).
  - `parse_and_filter_records(payload: bytes, tags: tuple[str, ...], on_invalid_timestamp: Callable[[], None] | None = None) -> list[Record]` — 타임스탬프를 해석할 수 없는 record는 버리고 그때마다 `on_invalid_timestamp()`를 호출.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_mqtt_subscriber.py`를 아래 내용으로 **전체 교체**한다(기존 테스트에 `epoch_ns`와 유효한 타임스탬프를 반영하고, 이벤트 시각/불량 타임스탬프/배치 테스트 3개를 추가).

```python
import types
from datetime import timedelta
from unittest.mock import MagicMock

from jetson_app.config import CalibrationConfig, EquipmentConfig
from jetson_app.mqtt_subscriber import MqttRecordSubscriber, Record, parse_and_filter_records
from jetson_app.timeparse import parse_dx1_timestamp


_EPOCH_2026_08_03_NS = parse_dx1_timestamp("2026-08-03T00:00:00Z")


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


class _FakeClient:
    def __init__(self):
        self.subscribed_topics = []

    def subscribe(self, topic):
        self.subscribed_topics.append(topic)


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
            epoch_ns=_EPOCH_2026_08_03_NS,
        )
    ]


def test_parse_and_filter_records_skips_record_with_no_matching_tags():
    payload = b'{"records": [{"timestamp": "2026-08-03T00:00:00Z", "unrelated:tag": 1.0}]}'

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_handles_multiple_records():
    payload = (
        b'{"records": ['
        b'{"timestamp": "2026-08-03T00:00:01Z", "servo1:torque": 1.0}, '
        b'{"timestamp": "2026-08-03T00:00:02Z", "servo1:torque": 2.0}'
        b']}'
    )

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert [r.timestamp for r in records] == ["2026-08-03T00:00:01Z", "2026-08-03T00:00:02Z"]


def test_parse_and_filter_records_handles_malformed_json():
    payload = b"not json"

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_handles_records_not_a_list():
    payload = b'{"records": null}'

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_skips_non_dict_items_in_records():
    payload = (
        b'{"records": ["not a dict", '
        b'{"timestamp": "2026-08-03T00:00:00Z", "servo1:torque": 1.0}]}'
    )

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == [
        Record(
            timestamp="2026-08-03T00:00:00Z",
            values={"servo1:torque": 1.0},
            epoch_ns=_EPOCH_2026_08_03_NS,
        )
    ]


def test_parse_and_filter_records_handles_non_object_top_level_json():
    payload = b'["not", "an", "object"]'

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_parse_and_filter_records_handles_non_utf8_bytes():
    payload = b"\x80\x81\x82"  # invalid UTF-8 start byte -> UnicodeDecodeError

    records = parse_and_filter_records(payload, tags=("servo1:torque",))

    assert records == []


def test_handle_connect_subscribes_on_success():
    config = _make_config(subscribe_topics=("topic/a",))
    subscriber = MqttRecordSubscriber(config, on_record=lambda record: None)
    fake_client = MagicMock()

    subscriber._handle_connect(fake_client, None, None, 0)

    subscribed_topics = {call.args[0] for call in fake_client.subscribe.call_args_list}
    assert "topic/a" in subscribed_topics


def test_handle_connect_does_not_subscribe_on_failure():
    config = _make_config(subscribe_topics=("topic/a",))
    subscriber = MqttRecordSubscriber(config, on_record=lambda record: None)
    fake_client = MagicMock()

    subscriber._handle_connect(fake_client, None, None, 5)

    fake_client.subscribe.assert_not_called()


def test_handle_message_delivers_parsed_records_to_on_record():
    config = _make_config(subscribe_topics=("topic/a",))
    received: list[Record] = []
    subscriber = MqttRecordSubscriber(config, on_record=received.append)
    fake_client = _FakeClient()
    msg = types.SimpleNamespace(
        payload=b'{"records": [{"timestamp": "2026-08-03T00:00:00Z", "a": 12.3}]}'
    )

    subscriber._handle_message(fake_client, None, msg)

    assert received == [
        Record(timestamp="2026-08-03T00:00:00Z", values={"a": 12.3}, epoch_ns=_EPOCH_2026_08_03_NS)
    ]


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


def test_parse_and_filter_records_extracts_event_time_from_dx1_timestamp():
    payload = (
        b'{"records": [{"timestamp": "2026-10-01T00:47:01.718651520+0000", '
        b'"NX5_Axis_X_Data:AxX_Act_Pos": 123.4}]}'
    )

    records = parse_and_filter_records(payload, tags=("NX5_Axis_X_Data:AxX_Act_Pos",))

    assert [r.epoch_ns for r in records] == [parse_dx1_timestamp("2026-10-01T00:47:01.718651520+0000")]


def test_parse_and_filter_records_drops_records_with_unparseable_timestamp_and_reports_them():
    payload = (
        b'{"records": ['
        b'{"timestamp": "not-a-time", "a": 1.0}, '
        b'{"timestamp": "2026-08-03T00:00:00Z", "a": 2.0}]}'
    )
    dropped = []

    records = parse_and_filter_records(payload, tags=("a",), on_invalid_timestamp=lambda: dropped.append(1))

    assert [r.values for r in records] == [{"a": 2.0}]
    assert dropped == [1]


def test_parse_and_filter_records_keeps_every_record_of_a_batch_message():
    batch = ", ".join(
        '{"timestamp": "2026-10-01T00:47:0%d.%d00000000+0000", "a": %d}' % (s, d, s * 10 + d)
        for s in (1, 2)
        for d in range(5)
    )
    payload = ('{"records": [%s]}' % batch).encode("utf-8")

    records = parse_and_filter_records(payload, tags=("a",))

    assert [r.values["a"] for r in records] == [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    assert [r.epoch_ns for r in records] == sorted(r.epoch_ns for r in records)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_mqtt_subscriber.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'epoch_ns'` 등.

- [ ] **Step 3: 구현** — `src/jetson_app/mqtt_subscriber.py`를 아래 내용으로 **전체 교체**한다.

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import paho.mqtt.client as mqtt

from .config import EquipmentConfig
from .droplog import DropCounter
from .timeparse import parse_dx1_timestamp


@dataclass(frozen=True)
class Record:
    timestamp: str
    values: dict[str, float]
    epoch_ns: int


def parse_and_filter_records(
    payload: bytes,
    tags: tuple[str, ...],
    on_invalid_timestamp: Callable[[], None] | None = None,
) -> list[Record]:
    tag_set = set(tags)
    try:
        data = json.loads(payload)
    except ValueError:
        # json.JSONDecodeError (malformed JSON) and UnicodeDecodeError
        # (non-UTF8 bytes, raised while json.loads decodes payload) are
        # both ValueError subclasses.
        return []

    if not isinstance(data, dict):
        return []

    records = data.get("records", [])
    if not isinstance(records, list):
        return []

    result = []
    for raw in records:
        if not isinstance(raw, dict):
            continue
        timestamp = raw.get("timestamp")
        values = {k: v for k, v in raw.items() if k in tag_set}
        if timestamp is None or not values:
            continue
        epoch_ns = parse_dx1_timestamp(timestamp)
        if epoch_ns is None:
            if on_invalid_timestamp is not None:
                on_invalid_timestamp()
            continue
        result.append(Record(timestamp=timestamp, values=values, epoch_ns=epoch_ns))
    return result


class MqttRecordSubscriber:
    def __init__(
        self,
        config: EquipmentConfig,
        on_record: Callable[[Record], None],
    ) -> None:
        self._config = config
        self._on_record = on_record
        self._invalid_timestamps = DropCounter(
            f"{config.equipment_id}: timestamp를 해석할 수 없어 폐기한 record"
        )
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
        for record in parse_and_filter_records(
            msg.payload, self._config.tags, on_invalid_timestamp=self._invalid_timestamps.add
        ):
            self._on_record(record)
```

- [ ] **Step 4: 기존 `tests/test_pipeline.py`의 `Record(...)` 호출 수정** — `Record`가 `epoch_ns`를 요구하므로 `test_build_pipeline_on_record_updates_tag_buffer` 안의 호출에 한 줄을 추가한다(이 파일은 Task 7에서 통째로 교체되므로 임시 수정이다).

```python
        Record(
            timestamp="2026-08-04T00:00:00+00:00",
            values={"PLC_Collector_Actuator_1:AirBlower.Cmd[0]": 1},
            epoch_ns=0,
        )
```

- [ ] **Step 5: 전체 통과 확인**

Run: `uv run pytest -q`
Expected: PASS (전체)

- [ ] **Step 6: 커밋**

```bash
git add jetson_app/src/jetson_app/mqtt_subscriber.py jetson_app/tests/test_mqtt_subscriber.py jetson_app/tests/test_pipeline.py
git commit -m "feat: parse record timestamps into event time (epoch ns)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `max_lateness_ms` 설정

**Files:**
- Modify: `src/jetson_app/config.py`
- Modify: `configs/test_dx1.example.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `EquipmentConfig.max_lateness_ms: int` (기본 `DEFAULT_MAX_LATENESS_MS = 2000`, 마지막 필드, 0 이상). 잘못된 값 → `ConfigError("max_lateness_ms must be a non-negative integer")`.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_config.py`의 `test_load_equipment_config_loads_shipped_example_config` **바로 위**에 아래 테스트들을 추가한다.

```python
def test_max_lateness_ms_defaults_to_2000_when_omitted(tmp_path):
    config_path = tmp_path / "test_dx1.yaml"
    config_path.write_text(SAMPLE_YAML, encoding="utf-8")

    assert load_equipment_config(config_path).max_lateness_ms == 2000


def test_max_lateness_ms_is_read_from_config(tmp_path):
    config_path = tmp_path / "test_dx1.yaml"
    config_path.write_text(SAMPLE_YAML + "max_lateness_ms: 500\n", encoding="utf-8")

    assert load_equipment_config(config_path).max_lateness_ms == 500


def test_max_lateness_ms_zero_is_allowed(tmp_path):
    config_path = tmp_path / "test_dx1.yaml"
    config_path.write_text(SAMPLE_YAML + "max_lateness_ms: 0\n", encoding="utf-8")

    assert load_equipment_config(config_path).max_lateness_ms == 0


@pytest.mark.parametrize("bad_value", ["-1", "1.5", '"2s"', "true", "null"])
def test_max_lateness_ms_rejects_invalid_values(tmp_path, bad_value):
    config_path = tmp_path / "test_dx1.yaml"
    config_path.write_text(SAMPLE_YAML + "max_lateness_ms: %s\n" % bad_value, encoding="utf-8")

    with pytest.raises(ConfigError, match="max_lateness_ms"):
        load_equipment_config(config_path)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'max_lateness_ms'` 등.

- [ ] **Step 3: 구현** — `src/jetson_app/config.py`에 네 군데를 수정한다.

(a) `class ConfigError` 위에 상수 추가:

```python
DEFAULT_MAX_LATENESS_MS = 2000


class ConfigError(ValueError):
    pass
```

(b) `EquipmentConfig`의 마지막에 기본값이 있는 필드 추가:

```python
    window_size: int
    calibration: CalibrationConfig
    max_lateness_ms: int = DEFAULT_MAX_LATENESS_MS
```

(c) `load_equipment_config`에서 `calibration_section = data["calibration"]` **바로 앞**에 검증 추가:

```python
    max_lateness_ms = data.get("max_lateness_ms", DEFAULT_MAX_LATENESS_MS)
    if not isinstance(max_lateness_ms, int) or isinstance(max_lateness_ms, bool) or max_lateness_ms < 0:
        raise ConfigError("max_lateness_ms must be a non-negative integer")

    calibration_section = data["calibration"]
```

(d) 함수 끝의 `return EquipmentConfig(...)`에 인자 추가:

```python
        calibration=CalibrationConfig(max_duration=max_duration, min_samples=min_samples),
        max_lateness_ms=max_lateness_ms,
    )
```

- [ ] **Step 4: 예제 config에 항목 추가** — `configs/test_dx1.example.yaml`의 `resample_interval_ms: 50` 줄을 아래 세 줄로 바꾼다.

```yaml
resample_interval_ms: 50   # 모델이 보는 시간 간격. DX1 취득 주기(Collector 주기)에 맞춰 직접 지정한다.
max_lateness_ms: 2000      # 토픽 간 도착 시차를 기다리는 시간. 토픽 중 가장 긴 배치 간격보다 크게 잡는다. 결과 발행은 이만큼 지연된다.
```

(`window_size: 100` 이하는 그대로 둔다.)

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest -q`
Expected: PASS (전체)

- [ ] **Step 6: 커밋**

```bash
git add jetson_app/src/jetson_app/config.py jetson_app/configs/test_dx1.example.yaml jetson_app/tests/test_config.py
git commit -m "feat: add max_lateness_ms config option" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 이벤트 시간 리샘플러

**Files:**
- Create: `src/jetson_app/resampler.py`
- Test: `tests/test_resampler.py`

**Interfaces:**
- Consumes: `Record` (Task 2), `Snapshot` (기존 `buffer.py`), `DropCounter` (Task 1)
- Produces:
  - `ResampledStep(epoch_ns: int, snapshot: Snapshot, reset_window: bool = False)` — frozen dataclass. `epoch_ns`는 칸 시작 시각.
  - `EventTimeResampler(tags: tuple[str, ...], interval_ms: int, max_lateness_ms: int, window_size: int)`
    - `.add(record: Record) -> list[ResampledStep]` — 이 record 때문에 새로 확정된 스텝을 시간순으로 반환(MQTT 스레드에서 호출).
    - `.late_dropped: int` — 확정된 칸보다 늦게 도착해 폐기한 record 누적 수.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_resampler.py`:

```python
from jetson_app.mqtt_subscriber import Record
from jetson_app.resampler import EventTimeResampler


def rec(ts_ms, **values):
    return Record(timestamp="", values=values, epoch_ns=ts_ms * 1_000_000)


def feed(resampler, records):
    steps = []
    for r in records:
        steps.extend(resampler.add(r))
    return steps


def make(interval_ms=100, lateness_ms=0, window_size=5, tags=("a",)):
    return EventTimeResampler(tags=tags, interval_ms=interval_ms, max_lateness_ms=lateness_ms, window_size=window_size)


def values_of(steps, tag="a"):
    return [s.snapshot.values[tag] for s in steps]


def test_batch_of_ten_records_yields_ten_distinct_steps():
    r = make()
    batch = [rec(i * 100, a=i) for i in range(10)] + [rec(1000, a=10)]
    steps = feed(r, batch)
    assert values_of(steps) == list(range(10))
    assert [s.epoch_ns for s in steps] == [i * 100_000_000 for i in range(10)]


def test_faster_than_grid_keeps_last_value_in_bucket():
    r = make()
    steps = feed(r, [rec(0, a=1), rec(20, a=2), rec(40, a=3), rec(60, a=4), rec(80, a=5), rec(100, a=6)])
    assert values_of(steps) == [5]


def test_slower_than_grid_forward_fills():
    r = make()
    steps = feed(r, [rec(0, a=1), rec(450, a=2)])
    assert values_of(steps) == [1, 1, 1, 1]
    assert [s.epoch_ns for s in steps] == [0, 100_000_000, 200_000_000, 300_000_000]


def test_fine_grid_20ms_with_100ms_source():
    r = make(interval_ms=20)
    steps = feed(r, [rec(0, a=1), rec(100, a=2), rec(200, a=3)])
    assert len(steps) == 10
    assert values_of(steps) == [1] * 5 + [2] * 5


def test_coarse_grid_1000ms_with_100ms_source():
    r = make(interval_ms=1000)
    steps = feed(r, [rec(i * 100, a=i) for i in range(21)])
    assert values_of(steps) == [9, 19]
    assert [s.epoch_ns for s in steps] == [0, 1_000_000_000]


def test_max_lateness_delays_emission_and_late_record_is_dropped():
    r = make(lateness_ms=300)
    assert feed(r, [rec(0, a=1), rec(200, a=2)]) == []
    steps = feed(r, [rec(500, a=3)])
    assert values_of(steps) == [1, 1]  # 칸 0, 1 확정 (칸 1은 ffill)
    assert r.add(rec(150, a=99)) == []  # 이미 확정된 칸 1에 속하는 늦은 record
    assert r.late_dropped == 1


def test_multi_topic_values_align_within_lateness():
    r = make(lateness_ms=200, tags=("a", "b"))
    steps = feed(
        r,
        [rec(0, a=1), rec(100, a=2), rec(200, a=3), rec(0, b=10), rec(100, b=20), rec(300, a=4)],
    )
    assert [s.snapshot.values for s in steps] == [{"a": 1, "b": 10}]
    steps = feed(r, [rec(400, a=5)])
    assert [s.snapshot.values for s in steps] == [{"a": 2, "b": 20}]


def test_partial_snapshot_keeps_none_for_unseen_tag_and_all_none_is_skipped():
    r = make(tags=("a", "b"))
    assert r.add(rec(0, zzz=1)) == []  # 추적하지 않는 태그뿐인 record
    steps = feed(r, [rec(0, a=1), rec(100, a=2)])
    assert [s.snapshot.values for s in steps] == [{"a": 1, "b": None}]


def test_records_inside_one_bucket_use_latest_event_time_regardless_of_arrival_order():
    r = make()
    steps = feed(r, [rec(80, a=2), rec(20, a=1), rec(100, a=3)])
    assert values_of(steps) == [2]


def test_duplicate_delivery_is_idempotent():
    r = make()
    steps = feed(r, [rec(0, a=1), rec(0, a=1), rec(100, a=2)])
    assert values_of(steps) == [1]
    assert r.add(rec(0, a=1)) == []  # 확정 뒤 재전송은 늦은 record로 폐기
    assert r.late_dropped == 1


def test_gap_longer_than_window_resets_window_without_flooding_steps():
    r = make(window_size=3)
    steps = feed(r, [rec(0, a=1), rec(100, a=2), rec(200, a=3), rec(10_000, a=4)])
    assert values_of(steps) == [1, 2, 3]
    assert [s.reset_window for s in steps] == [False, False, False]
    steps = feed(r, [rec(10_100, a=5)])
    assert values_of(steps) == [4]
    assert steps[0].reset_window is True
    assert steps[0].epoch_ns == 10_000 * 1_000_000


def test_one_day_forward_jump_stays_bounded():
    r = make(window_size=3)
    feed(r, [rec(0, a=1), rec(100, a=2)])
    steps = feed(r, [rec(86_400_000, a=3), rec(86_400_100, a=4)])
    assert len(steps) <= 3
    assert steps[-1].reset_window is True


def test_gap_equal_to_window_size_is_forward_filled_not_reset():
    r = make(window_size=3)
    steps = feed(r, [rec(0, a=1), rec(400, a=2), rec(500, a=3)])
    assert values_of(steps) == [1, 1, 1, 1, 2]
    assert all(s.reset_window is False for s in steps)


def test_record_far_in_the_past_after_emission_is_dropped():
    r = make()
    feed(r, [rec(0, a=1), rec(100, a=2), rec(200, a=3)])
    assert r.add(rec(-86_400_000, a=9)) == []
    assert r.late_dropped == 1
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_resampler.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'jetson_app.resampler'`.

- [ ] **Step 3: 구현** — `src/jetson_app/resampler.py`:

```python
from __future__ import annotations

import threading
from dataclasses import dataclass

from .buffer import Snapshot
from .droplog import DropCounter
from .mqtt_subscriber import Record


@dataclass(frozen=True)
class ResampledStep:
    """격자 한 칸을 확정한 결과. epoch_ns는 칸의 시작 시각.
    reset_window가 True면 직전 스텝과 윈도우 길이를 넘는 공백이 있었다는 뜻이므로,
    소비자는 슬라이딩 윈도우를 비우고 이 스텝부터 다시 쌓아야 한다."""

    epoch_ns: int
    snapshot: Snapshot
    reset_window: bool = False


class EventTimeResampler:
    """record의 이벤트 시각(timestamp)을 기준으로 값을 `interval_ms` 격자에 배치한다.

    칸 k는 [k·Δ, (k+1)·Δ). 칸 안에 같은 태그의 값이 여럿이면 이벤트 시각이 가장 늦은
    값을, 없으면 직전 값을 유지(ffill)한다. 지금까지 본 가장 늦은 이벤트 시각에서
    `max_lateness_ms`를 뺀 시각 이전의 칸은 확정(방출)한다 — 여러 토픽이 서로 다른
    시점에 도착해도 같은 시각의 값이 한 스냅샷에 모이도록 기다리는 시간이다.
    Jetson 시계는 전혀 쓰지 않는다.
    """

    def __init__(
        self,
        tags: tuple[str, ...],
        interval_ms: int,
        max_lateness_ms: int,
        window_size: int,
    ) -> None:
        self._tags = tags
        self._tag_set = set(tags)
        self._interval_ns = interval_ms * 1_000_000
        self._lateness_ns = max_lateness_ms * 1_000_000
        self._window_size = window_size
        self._lock = threading.Lock()
        # 칸 인덱스 -> {태그: (이벤트 시각 ns, 값)}
        self._pending: dict[int, dict[str, tuple[int, float | int]]] = {}
        self._last_values: dict[str, float | int] = {}
        self._max_ns: int | None = None
        self._next_bucket: int | None = None  # 다음에 방출할 칸. 첫 방출 전에는 None
        self._reset_pending = False
        self._late = DropCounter("resampler: 확정된 칸보다 늦게 도착해 폐기한 record")

    @property
    def late_dropped(self) -> int:
        return self._late.total

    def add(self, record: Record) -> list[ResampledStep]:
        """record를 반영하고, 이 record 때문에 새로 확정된 스텝들을 시간순으로 반환한다."""
        with self._lock:
            bucket = record.epoch_ns // self._interval_ns
            if self._next_bucket is not None and bucket < self._next_bucket:
                self._late.add()
                return []
            tracked = {t: v for t, v in record.values.items() if t in self._tag_set}
            if not tracked:
                return []
            slot = self._pending.setdefault(bucket, {})
            for tag, value in tracked.items():
                current = slot.get(tag)
                if current is None or record.epoch_ns >= current[0]:
                    slot[tag] = (record.epoch_ns, value)
            if self._max_ns is None or record.epoch_ns > self._max_ns:
                self._max_ns = record.epoch_ns
            return self._drain()

    def _drain(self) -> list[ResampledStep]:
        closed_below = (self._max_ns - self._lateness_ns) // self._interval_ns
        if self._next_bucket is None:
            first = min(self._pending)
            if first >= closed_below:
                return []
            self._next_bucket = first
        steps: list[ResampledStep] = []
        while True:
            next_data = min(self._pending) if self._pending else None
            if next_data is not None and next_data - self._next_bucket > self._window_size:
                # 윈도우 길이를 넘는 공백(통신 단절 등): ffill 칸을 만들지 않고 건너뛴다.
                self._next_bucket = next_data
                self._reset_pending = True
            limit = closed_below if next_data is None else min(closed_below, next_data)
            for bucket in range(self._next_bucket, limit):
                self._append_step(steps, bucket, {})
            self._next_bucket = max(self._next_bucket, limit)
            if next_data is not None and next_data < closed_below:
                self._append_step(steps, next_data, self._pending.pop(next_data))
                self._next_bucket = next_data + 1
                continue
            break
        return steps

    def _append_step(
        self,
        steps: list[ResampledStep],
        bucket: int,
        slot: dict[str, tuple[int, float | int]],
    ) -> None:
        for tag, (_, value) in slot.items():
            self._last_values[tag] = value
        values = {tag: self._last_values.get(tag) for tag in self._tags}
        if all(v is None for v in values.values()):
            return  # 아직 어떤 태그도 값을 받지 못한 칸은 방출하지 않는다
        steps.append(
            ResampledStep(
                epoch_ns=bucket * self._interval_ns,
                snapshot=Snapshot(values=values),
                reset_window=self._reset_pending,
            )
        )
        self._reset_pending = False
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_resampler.py -q` → PASS (모두 통과), 이어서 `uv run pytest -q` → PASS (전체).

- [ ] **Step 5: 커밋**

```bash
git add jetson_app/src/jetson_app/resampler.py jetson_app/tests/test_resampler.py
git commit -m "feat: add event-time resampler with watermark and gap handling" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 처리 워커 `SnapshotProcessor`

**Files:**
- Create: `src/jetson_app/snapshot_processor.py`
- Modify: `src/jetson_app/buffer.py` (`SlidingWindow.clear()` 추가)
- Test: `tests/test_snapshot_processor.py`, `tests/test_buffer.py`

**Interfaces:**
- Consumes: `ResampledStep` (Task 4), `format_epoch_ns`, `DropCounter` (Task 1), 기존 `SlidingWindow`, `CalibrationManager`, `ActiveModelHolder`, `Debouncer`, `ResultPublisher`
- Produces:
  - `SlidingWindow.clear() -> None`
  - `SnapshotProcessor(sliding_window, calibration_manager, inference_engine_holder=None, debouncer=None, result_publisher=None, max_queue_size=10_000)`
    - `.submit(steps: list[ResampledStep]) -> None` — MQTT 스레드용. 큐가 가득 차면 가장 오래된 스텝을 버리고 로그.
    - `.process(step: ResampledStep) -> None` — 스텝 하나 처리(기존 `_tick` 본문).
    - `.process_pending() -> None` — 큐를 호출 스레드에서 모두 처리.
    - `.start() / .stop()` — `stop()`은 스레드를 join한 뒤 남은 큐를 마저 처리.
    - `.dropped: int` — 큐 초과로 버린 스텝 누적 수.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_buffer.py` 맨 끝에 추가:

```python
def test_sliding_window_clear_empties_the_window():
    window = SlidingWindow(window_size=2)
    window.push(Snapshot(values={"a": 1}))
    window.push(Snapshot(values={"a": 2}))

    window.clear()

    assert window.to_list() == []
    assert window.is_full() is False
```

`tests/test_snapshot_processor.py`:

```python
from datetime import datetime, timedelta

from jetson_app.buffer import SlidingWindow, Snapshot
from jetson_app.calibration import (
    CalibrationBufferWriter,
    CalibrationManager,
    CalibrationState,
    StateStore,
)
from jetson_app.inference import AnomalyResult
from jetson_app.resampler import ResampledStep
from jetson_app.snapshot_processor import SnapshotProcessor
from jetson_app.timeparse import format_epoch_ns


def _step(i, value=None, reset_window=False):
    return ResampledStep(
        epoch_ns=i * 50_000_000,
        snapshot=Snapshot(values={"a": float(i) if value is None else value}),
        reset_window=reset_window,
    )


class _FakeCalibrationManager:
    def __init__(self, state):
        self.state = state
        self.recorded = []

    def record_sample(self, snapshot, timestamp):
        self.recorded.append((snapshot, timestamp))


class _FakeEngine:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def score(self, window, actual):
        self.calls.append((list(window), actual))
        return self._result


class _FakeHolder:
    def __init__(self, engine):
        self._engine = engine

    def get(self):
        return self._engine


class _FakeDebouncer:
    def __init__(self, alarm):
        self._alarm = alarm
        self.scores = []
        self.reset_calls = 0

    def update(self, score):
        self.scores.append(score)
        return self._alarm

    def reset(self):
        self.reset_calls += 1


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, timestamp, anomaly_score, alarm, top_deviant_tag):
        self.published.append((timestamp, anomaly_score, alarm, top_deviant_tag))


def _filled_window(size, count):
    window = SlidingWindow(size)
    for i in range(count):
        window.push(Snapshot(values={"a": float(i)}))
    return window


def _processor(window, manager, engine=None, debouncer=None, publisher=None, **kwargs):
    return SnapshotProcessor(
        sliding_window=window,
        calibration_manager=manager,
        inference_engine_holder=_FakeHolder(engine) if engine is not None else None,
        debouncer=debouncer,
        result_publisher=publisher,
        **kwargs,
    )


def test_process_does_not_score_when_calibrating():
    window = _filled_window(2, 2)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    publisher = _FakePublisher()
    processor = _processor(window, manager, engine, _FakeDebouncer(True), publisher)

    processor.process(_step(2, 99.0))

    assert engine.calls == []
    assert publisher.published == []
    assert manager.recorded  # CALIBRATING이어도 캘리브레이션 기록은 계속됨


def test_process_does_not_score_when_window_not_full():
    window = _filled_window(5, 2)
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    processor = _processor(window, manager, engine, _FakeDebouncer(False), _FakePublisher())

    processor.process(_step(2, 99.0))

    assert engine.calls == []


def test_process_scores_with_pre_push_window_and_publishes_with_event_time():
    window = _filled_window(2, 2)
    pre_push_window = window.to_list()
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    debouncer = _FakeDebouncer(alarm=True)
    publisher = _FakePublisher()
    processor = _processor(window, manager, engine, debouncer, publisher)

    step = _step(2, 99.0)
    processor.process(step)

    assert len(engine.calls) == 1
    scored_window, scored_actual = engine.calls[0]
    assert scored_window == pre_push_window  # push되기 *전* 윈도우로 채점됐는지 확인
    assert scored_actual.values == {"a": 99.0}
    assert debouncer.scores == [5.0]
    timestamp, anomaly_score, alarm, top_deviant_tag = publisher.published[0]
    assert timestamp == format_epoch_ns(step.epoch_ns)  # 현재 시각이 아니라 이벤트 시각
    assert (anomaly_score, alarm, top_deviant_tag) == (5.0, True, "a")
    assert manager.recorded[0][1] == format_epoch_ns(step.epoch_ns)


def test_process_skips_publish_when_engine_returns_none():
    window = _filled_window(2, 2)
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    publisher = _FakePublisher()
    processor = _processor(window, manager, _FakeEngine(None), _FakeDebouncer(False), publisher)

    processor.process(_step(2, 99.0))

    assert publisher.published == []


def test_process_does_not_score_when_debouncer_and_publisher_missing():
    # holder만 주입되고 debouncer/result_publisher가 None이면 채점을 조용히 건너뛰어야
    # 한다. 그냥 진행하면 AttributeError가 매 스텝 발생하고 포괄 예외 처리에 삼켜져
    # 로그만 무한히 쌓인다.
    window = _filled_window(2, 2)
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=5.0, top_deviant_tag="a"))
    processor = _processor(window, manager, engine)

    processor.process(_step(2, 99.0))  # 예외 없이 통과해야 한다

    assert engine.calls == []
    assert manager.recorded


def test_process_without_inference_collaborators_still_records_calibration():
    window = SlidingWindow(2)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.process(_step(1, 1.0))

    assert manager.recorded
    assert len(window.to_list()) == 1


def test_process_reset_window_clears_window_and_debouncer_before_pushing():
    window = _filled_window(3, 3)
    manager = _FakeCalibrationManager(CalibrationState.MONITORING)
    engine = _FakeEngine(AnomalyResult(anomaly_score=1.0, top_deviant_tag="a"))
    debouncer = _FakeDebouncer(False)
    processor = _processor(window, manager, engine, debouncer, _FakePublisher())

    step = _step(10, 7.0, reset_window=True)
    processor.process(step)

    assert debouncer.reset_calls == 1
    assert window.to_list() == [step.snapshot]  # 비워진 뒤 이 스텝만 들어 있다
    assert engine.calls == []  # 윈도우가 비었으니 채점하지 않는다


def test_submit_then_process_pending_handles_steps_in_order():
    window = SlidingWindow(10)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.submit([_step(1), _step(2), _step(3)])
    processor.process_pending()

    assert [s.values["a"] for s in window.to_list()] == [1.0, 2.0, 3.0]
    assert len(manager.recorded) == 3


def test_submit_drops_oldest_when_queue_is_full():
    window = SlidingWindow(10)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager, max_queue_size=2)

    processor.submit([_step(1), _step(2), _step(3)])
    processor.process_pending()

    assert [s.values["a"] for s in window.to_list()] == [2.0, 3.0]
    assert processor.dropped == 1


def test_stop_flushes_steps_submitted_before_stop_and_stops_thread():
    window = SlidingWindow(10)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.start()
    processor.submit([_step(1), _step(2)])
    processor.stop()

    assert len(manager.recorded) == 2
    assert processor._thread.is_alive() is False


def test_stop_without_start_still_flushes_queue():
    window = SlidingWindow(10)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.submit([_step(1)])
    processor.stop()

    assert len(manager.recorded) == 1


def test_restart_after_stop_processes_again():
    window = SlidingWindow(10)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.start()
    processor.stop()
    processor.start()
    processor.submit([_step(1)])
    processor.stop()

    assert len(manager.recorded) == 1


def test_worker_survives_exception_in_processing(capsys):
    window = SlidingWindow(10)

    class _ExplodingManager(_FakeCalibrationManager):
        def record_sample(self, snapshot, timestamp):
            if snapshot.values["a"] == 1.0:
                raise RuntimeError("boom")
            super().record_sample(snapshot, timestamp)

    manager = _ExplodingManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.submit([_step(1), _step(2)])
    processor.process_pending()

    assert len(manager.recorded) == 1
    assert "boom" in capsys.readouterr().out


def test_heartbeat_line_printed_every_100_steps(capsys):
    window = SlidingWindow(10)
    manager = _FakeCalibrationManager(CalibrationState.CALIBRATING)
    processor = _processor(window, manager)

    processor.submit([_step(i) for i in range(100)])
    processor.process_pending()

    assert "[snapshotter] 100번째 스냅샷 처리" in capsys.readouterr().out


def test_calibration_timestamps_are_parseable_by_prune(tmp_path):
    # 캘리브레이션 버퍼의 prune_older_than이 datetime.fromisoformat으로 읽는 값이므로,
    # 이벤트 시각 문자열이 Python 3.8에서 파싱 가능해야 한다.
    writer = CalibrationBufferWriter(tmp_path / "buf.jsonl")
    manager = CalibrationManager(
        buffer_writer=writer,
        min_samples=1,
        max_duration=timedelta(days=7),
        train_fn=lambda samples: None,
        state_store=StateStore(tmp_path / "state"),
    )
    processor = _processor(SlidingWindow(3), manager)

    processor.process(_step(1, 5.0))

    sample = writer.read_all()[0]
    assert datetime.fromisoformat(sample.timestamp).tzinfo is not None
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_snapshot_processor.py tests/test_buffer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'jetson_app.snapshot_processor'` 및 `AttributeError: ... 'clear'`.

- [ ] **Step 3: 구현**

`src/jetson_app/buffer.py`의 `SlidingWindow`에서 `push` 메서드 **바로 뒤**(`is_full` 앞)에 추가:

```python
    def clear(self) -> None:
        self._items.clear()
```

`src/jetson_app/snapshot_processor.py`:

```python
from __future__ import annotations

import queue
import threading

from .buffer import SlidingWindow
from .calibration import CalibrationManager, CalibrationState
from .debounce import Debouncer
from .droplog import DropCounter
from .inference import ActiveModelHolder
from .publisher import ResultPublisher
from .resampler import ResampledStep
from .timeparse import format_epoch_ns

_HEARTBEAT_STEP_INTERVAL = 100  # 처리한 스텝 100개마다 한 줄 (50ms 격자면 약 5초)
_DEFAULT_MAX_QUEUE_SIZE = 10_000


class SnapshotProcessor:
    """리샘플러가 확정한 스텝(`ResampledStep`)을 큐로 받아 처리하는 워커.

    MQTT 스레드는 `submit()`으로 큐에 넣기만 하고, 점수 계산(GRU 추론)은 별도 스레드가
    한다. 스텝마다 하는 일: 새 스냅샷을 윈도우에 넣기 *전에* 그 시점까지의 윈도우로 다음
    값을 예측해 이상 점수를 계산·발행하고(MONITORING이고 모델이 있을 때만), 슬라이딩
    윈도우와 캘리브레이션 버퍼에 스냅샷을 쌓는다. 스냅샷을 먼저 넣으면 "미래"를 보고
    예측하는 꼴이 되어 스코어링이 무의미해진다.
    """

    def __init__(
        self,
        sliding_window: SlidingWindow,
        calibration_manager: CalibrationManager,
        inference_engine_holder: ActiveModelHolder | None = None,
        debouncer: Debouncer | None = None,
        result_publisher: ResultPublisher | None = None,
        max_queue_size: int = _DEFAULT_MAX_QUEUE_SIZE,
    ) -> None:
        self._sliding_window = sliding_window
        self._calibration_manager = calibration_manager
        self._inference_engine_holder = inference_engine_holder
        self._debouncer = debouncer
        self._result_publisher = result_publisher
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._dropped = DropCounter("snapshot_processor: 처리가 밀려 폐기한 스냅샷")
        self._stop_event = threading.Event()
        self._thread = None
        self._step_count = 0

    @property
    def dropped(self) -> int:
        return self._dropped.total

    def submit(self, steps: list[ResampledStep]) -> None:
        """MQTT 스레드에서 호출. 큐가 가득 차면 가장 오래된 스텝을 버리고 로그를 남긴다."""
        for step in steps:
            while True:
                try:
                    self._queue.put_nowait(step)
                    break
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._dropped.add()

    def process_pending(self) -> None:
        """큐에 쌓인 스텝을 호출한 스레드에서 모두 처리한다(stop()과 테스트에서 사용)."""
        while True:
            try:
                step = self._queue.get_nowait()
            except queue.Empty:
                return
            self._process_safely(step)

    def process(self, step: ResampledStep) -> None:
        if step.reset_window:
            # 통신 단절 등으로 윈도우 길이를 넘는 공백이 있었다: 공백 앞뒤를 이어 붙인
            # 윈도우로 학습/추론하지 않도록 비우고, 연속 초과 카운터도 되돌린다.
            self._sliding_window.clear()
            if self._debouncer is not None:
                self._debouncer.reset()
        snapshot = step.snapshot
        timestamp = format_epoch_ns(step.epoch_ns)

        self._score_and_publish(timestamp, snapshot)

        self._sliding_window.push(snapshot)
        self._calibration_manager.record_sample(snapshot, timestamp)
        self._step_count += 1
        if self._step_count % _HEARTBEAT_STEP_INTERVAL == 0:
            print(
                f"[snapshotter] {self._step_count}번째 스냅샷 처리, "
                f"윈도우 {len(self._sliding_window.to_list())}/{self._sliding_window.window_size}, "
                f"캘리브레이션 상태={self._calibration_manager.state.value}"
            )

    def _score_and_publish(self, timestamp: str, snapshot) -> None:
        # 세 협력자는 함께 있어야만 채점이 성립한다. 하나라도 없는 상태로 진행하면
        # 뒤쪽 update()/publish()에서 AttributeError가 나고, _process_safely의 포괄 예외
        # 처리에 삼켜져 매 스텝 로그만 쏟아진다.
        if (
            self._inference_engine_holder is None
            or self._debouncer is None
            or self._result_publisher is None
        ):
            return
        if self._calibration_manager.state != CalibrationState.MONITORING:
            return
        if not self._sliding_window.is_full():
            return
        engine = self._inference_engine_holder.get()
        if engine is None:
            return
        result = engine.score(self._sliding_window.to_list(), snapshot)
        if result is None:
            return
        alarm = self._debouncer.update(result.anomaly_score)
        self._result_publisher.publish(
            timestamp, result.anomaly_score, alarm, result.top_deviant_tag
        )

    def _process_safely(self, step: ResampledStep) -> None:
        try:
            self.process(step)
        except Exception as e:
            # 백그라운드 스레드가 죽으면 데이터 수집이 조용히 멈추므로
            # 어떤 예외도 로그만 남기고 계속 진행한다.
            print(f"[SnapshotProcessor] 스텝 처리 중 오류 발생, 계속 진행: {e}")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                step = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._process_safely(step)

    def start(self) -> None:
        # stop()이 set()한 이벤트를 지우지 않으면, stop() 이후 재시작된 스레드는
        # 루프 조건이 이미 참(정지)인 채로 시작해 아무것도 처리하지 않는다.
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        # 정지 직전까지 제출된 스텝이 큐에 남지 않도록 마저 처리한다.
        self.process_pending()
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_snapshot_processor.py tests/test_buffer.py -q` → PASS, 이어서 `uv run pytest -q` → PASS (전체).

- [ ] **Step 5: 커밋**

```bash
git add jetson_app/src/jetson_app/snapshot_processor.py jetson_app/src/jetson_app/buffer.py jetson_app/tests/test_snapshot_processor.py jetson_app/tests/test_buffer.py
git commit -m "feat: add queue-driven SnapshotProcessor worker" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: 학습 모델에 격자 값 저장

**Files:**
- Modify: `src/jetson_app/training.py`, `src/jetson_app/subscriber_cli.py`
- Modify: `tests/test_training.py`, `tests/test_pipeline.py` (`make_train_fn` 호출 6곳)

**Interfaces:**
- Produces:
  - `ModelArtifact.resample_interval_ms: int = 0` (마지막 필드. 0 = 격자 정보 없는 기존 artifact)
  - `train_model(..., window_size, resample_interval_ms: int = 0, epochs=...)` — artifact에 그대로 기록
  - `make_train_fn(tags, window_size, model_path, resample_interval_ms: int, epochs=...)` — **필수** 인자
  - `save_artifact`/`load_artifact`가 이 값을 저장/복원(없으면 0)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_training.py`에서 `test_make_train_fn_trains_and_saves` **바로 위**에 아래 세 테스트를 추가하고, 그 테스트 본문은 아래처럼 고친다(`resample_interval_ms=50` 인자와 마지막 assert 추가).

```python
def test_train_model_records_resample_interval_in_artifact():
    artifact = train_model(
        _make_samples(30),
        tags=("a", "b"),
        window_size=3,
        resample_interval_ms=100,
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )
    assert artifact.resample_interval_ms == 100


def test_save_and_load_artifact_preserves_resample_interval(tmp_path: Path):
    artifact = train_model(
        _make_samples(30),
        tags=("a", "b"),
        window_size=3,
        resample_interval_ms=100,
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )
    path = tmp_path / "model.pt"
    save_artifact(path, artifact)

    assert load_artifact(path).resample_interval_ms == 100


def test_load_artifact_without_grid_info_defaults_to_zero(tmp_path: Path):
    # 격자 정보가 생기기 전에 저장된 기존 artifact: pipeline이 항상 불일치로 처리하도록 0
    artifact = train_model(
        _make_samples(30),
        tags=("a", "b"),
        window_size=3,
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )
    path = tmp_path / "legacy.pt"
    save_artifact(path, artifact)
    import torch as _torch

    data = _torch.load(path, weights_only=False)
    del data["resample_interval_ms"]
    _torch.save(data, path)

    assert load_artifact(path).resample_interval_ms == 0
```

`test_make_train_fn_trains_and_saves` 수정 (`make_train_fn(...)` 호출에 `resample_interval_ms=50,` 추가, 마지막에 assert 추가):

```python
    train_fn = make_train_fn(
        tags=("a", "b"),
        window_size=3,
        model_path=model_path,
        resample_interval_ms=50,
        epochs=2,
        hidden_size=4,
        num_layers=1,
    )
    train_fn(samples)
    assert model_path.exists()
    loaded = load_artifact(model_path)
    assert loaded.tags == ("a", "b")
    assert loaded.resample_interval_ms == 50
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_training.py -q`
Expected: FAIL — `TypeError: train_model() got an unexpected keyword argument 'resample_interval_ms'` 등.

- [ ] **Step 3: 구현** — `src/jetson_app/training.py`를 여섯 군데 수정한다.

(a) `ModelArtifact`의 `state_dict: dict` 뒤에 필드 추가:

```python
    state_dict: dict
    # 학습 때의 리샘플 격자 간격(ms). 0은 격자 정보가 없는 기존 artifact를 뜻하며,
    # pipeline의 호환성 검사에서 항상 불일치로 처리되어 재학습을 유도한다.
    resample_interval_ms: int = 0
```

(b) `train_model` 시그니처에서 `window_size: int,` 다음 줄에 추가:

```python
    window_size: int,
    resample_interval_ms: int = 0,
    epochs: int = DEFAULT_EPOCHS,
```

(c) `train_model`이 반환하는 `ModelArtifact(...)`에 마지막 인자 추가:

```python
        state_dict=model.state_dict(),
        resample_interval_ms=resample_interval_ms,
    )
```

(d) `save_artifact`의 저장 딕셔너리 끝에 추가:

```python
            "state_dict": artifact.state_dict,
            "resample_interval_ms": artifact.resample_interval_ms,
        },
```

(e) `load_artifact`의 `ModelArtifact(...)` 끝에 추가:

```python
        state_dict=data["state_dict"],
        resample_interval_ms=data.get("resample_interval_ms", 0),
    )
```

(f) `make_train_fn` 시그니처의 `model_path: str | Path,` 다음 줄에 **필수** 인자를 추가하고, 내부 `train_model(...)` 호출에 전달:

```python
    model_path: str | Path,
    resample_interval_ms: int,
    epochs: int = DEFAULT_EPOCHS,
```

```python
            window_size=window_size,
            resample_interval_ms=resample_interval_ms,
            epochs=epochs,
```

`src/jetson_app/subscriber_cli.py`의 `make_train_fn(...)` 호출에 한 줄 추가:

```python
        train_fn = make_train_fn(
            tags=config.tags,
            window_size=config.window_size,
            model_path=model_path,
            resample_interval_ms=config.resample_interval_ms,
        )
```

- [ ] **Step 4: 기존 `tests/test_pipeline.py`의 `make_train_fn(` 호출 6곳 수정** — 필수 인자가 생겼으므로 임시로 맞춘다(이 파일은 Task 7에서 통째로 교체된다). 아래 스크립트를 `jetson_app` 디렉터리에서 실행한다.

```bash
uv run python - <<'EOF'
import re
p = "tests/test_pipeline.py"
s = open(p, encoding="utf-8").read()
new, n = re.subn(
    r"(model_path=[^\n]+,\n)(\s+epochs=1,)",
    lambda m: m.group(1) + "        resample_interval_ms=config.resample_interval_ms,\n" + m.group(2),
    s,
)
assert n == 6, n
open(p, "w", encoding="utf-8").write(new)
EOF
```

- [ ] **Step 5: 전체 통과 확인**

Run: `uv run pytest -q`
Expected: PASS (전체)

- [ ] **Step 6: 커밋**

```bash
git add jetson_app/src/jetson_app/training.py jetson_app/src/jetson_app/subscriber_cli.py jetson_app/tests/test_training.py jetson_app/tests/test_pipeline.py
git commit -m "feat: store resample interval in model artifact" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 파이프라인 연결과 구 스케줄러 제거

**Files:**
- Modify: `src/jetson_app/pipeline.py` (전체 교체), `src/jetson_app/buffer.py` (전체 교체)
- Delete: `src/jetson_app/scheduler.py`, `tests/test_scheduler.py`
- Modify: `tests/test_pipeline.py` (전체 교체), `tests/test_buffer.py` (전체 교체)

**Interfaces:**
- Consumes: Task 1~6의 모든 산출물
- Produces:
  - `Pipeline` 필드: `config`, `resampler: EventTimeResampler`, `sliding_window`, `calibration_manager`, `inference_engine_holder`, `snapshotter: SnapshotProcessor`, `mqtt_subscriber`, `command_subscriber` (`tag_buffer` 필드는 `resampler`로 대체)
  - `build_pipeline(config, calibration_dir, model_dir, train_fn) -> Pipeline` — 시그니처 불변. `on_record`가 `snapshotter.submit(resampler.add(record))`를 호출. 저장된 모델의 `window_size`/`tags`/`resample_interval_ms`가 config와 다르면 `CALIBRATING` 폴백.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_pipeline.py`를 아래 내용으로 **전체 교체**한다(`time.sleep`/스레드 의존을 `process_pending()`으로 대체하고, 배치 메시지·격자 불일치·격자 정보 없는 artifact·워커 스레드 테스트를 추가).

```python
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from jetson_app.calibration import CalibrationState, StateStore
from jetson_app.config import CalibrationConfig, EquipmentConfig
from jetson_app.model import AnomalyGRU
from jetson_app.mqtt_subscriber import Record
from jetson_app.pipeline import build_pipeline
from jetson_app.training import ModelArtifact, load_artifact, make_train_fn, save_artifact


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


def _e2e_config(equipment_id, tags=("tag_a", "tag_b"), window_size=3, min_samples=10):
    # 격자 5ms, 지연 허용 0ms: 테스트에서 칸이 다음 record가 도착하는 즉시 확정된다.
    return EquipmentConfig(
        equipment_id=equipment_id,
        subscribe_topics=(f"dx1/{equipment_id}/data",),
        publish_topic=f"jetson/{equipment_id}/anomaly",
        command_topic=f"jetson/{equipment_id}/cmd",
        tags=tags,
        resample_interval_ms=5,
        window_size=window_size,
        calibration=CalibrationConfig(max_duration=timedelta(days=7), min_samples=min_samples),
        max_lateness_ms=0,
    )


def _ts(i, step_ms=5):
    """i번째 격자 칸의 DX1 형식 타임스탬프(나노초 9자리, +0000)."""
    moment = datetime(2026, 8, 4, tzinfo=timezone.utc) + timedelta(milliseconds=i * step_ms)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000+0000"


def _send(pipeline, i, tags):
    """i번째 시점의 record 하나를 담은 MQTT 메시지를 구독자에 직접 전달한다.
    첫 태그는 연속값(float(i)), 나머지는 0/1(i % 2)."""
    record = {"timestamp": _ts(i)}
    for j, tag in enumerate(tags):
        record[tag] = float(i) if j == 0 else i % 2
    payload = json.dumps({"records": [record]}).encode("utf-8")
    pipeline.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))


def _feed(pipeline, tags, start, stop):
    for i in range(start, stop):
        _send(pipeline, i, tags)
    pipeline.snapshotter.process_pending()


def _train_fn_for(config, model_dir):
    return make_train_fn(
        tags=config.tags,
        window_size=config.window_size,
        model_path=model_dir / f"{config.equipment_id}.pt",
        resample_interval_ms=config.resample_interval_ms,
        epochs=1,
        hidden_size=4,
        num_layers=1,
    )


def _send_train(pipeline):
    pipeline.command_subscriber._handle_command_message(
        None, None, SimpleNamespace(payload=b'{"command": "train"}')
    )


def test_build_pipeline_wires_all_components(tmp_path):
    config = _make_config(tmp_path)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=tmp_path / "model_data",
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
        model_dir=tmp_path / "model_data",
        train_fn=lambda samples: None,
    )

    assert pipeline.command_subscriber._calibration_manager is pipeline.calibration_manager


def test_build_pipeline_on_record_feeds_resampler_and_processor(tmp_path):
    config = replace(_make_config(tmp_path), max_lateness_ms=0)
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=tmp_path / "model_data",
        train_fn=lambda samples: None,
    )
    tag = "PLC_Collector_Actuator_1:AirBlower.Cmd[0]"

    pipeline.mqtt_subscriber._on_record(
        Record(timestamp="2026-08-04T00:00:00+00:00", values={tag: 1}, epoch_ns=0)
    )
    pipeline.mqtt_subscriber._on_record(
        Record(timestamp="2026-08-04T00:00:00.050+00:00", values={tag: 0}, epoch_ns=50_000_000)
    )
    pipeline.snapshotter.process_pending()

    # 두 번째 record가 첫 칸(0~50ms)을 확정시켰고, 그 칸의 값은 1이다.
    assert [s.values[tag] for s in pipeline.sliding_window.to_list()] == [1]


def test_pipeline_end_to_end_message_through_training(tmp_path):
    config = _e2e_config("e2e_test", tags=("tag_a",), min_samples=3)
    train_calls = []
    model_dir = tmp_path / "model_data"

    def _fake_train_fn(samples):
        # 실제 학습 대신, wrapped_train_fn(pipeline.py)이 학습 직후 곧바로
        # load_artifact로 다시 불러올 수 있도록 최소한의 유효한 아티팩트만 저장한다.
        train_calls.append(samples)
        model = AnomalyGRU(
            num_tags=1, continuous_indices=[0], binary_indices=[], hidden_size=2, num_layers=1
        )
        artifact = ModelArtifact(
            tags=config.tags,
            tag_types={"tag_a": "continuous"},
            norm_stats={"tag_a": (0.0, 1.0)},
            error_stats={"tag_a": (0.0, 1.0)},
            window_size=config.window_size,
            hidden_size=2,
            num_layers=1,
            state_dict=model.state_dict(),
            resample_interval_ms=config.resample_interval_ms,
        )
        save_artifact(model_dir / f"{config.equipment_id}.pt", artifact)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=_fake_train_fn,
    )

    # 한 메시지에 시점이 다른 record 5개(배치) -> 마지막 칸을 뺀 4칸이 확정된다
    payload = json.dumps(
        {"records": [{"timestamp": _ts(i), "tag_a": 42} for i in range(5)]}
    ).encode("utf-8")
    pipeline.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))
    pipeline.snapshotter.process_pending()

    buffer_path = tmp_path / "calibration_data" / "e2e_test.jsonl"
    assert buffer_path.exists()
    recorded_lines = buffer_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(recorded_lines) == 4

    _send_train(pipeline)

    assert len(train_calls) == 1
    assert pipeline.calibration_manager.state == CalibrationState.MONITORING
    assert not buffer_path.exists()


def test_pipeline_batch_message_keeps_every_record_at_its_own_timestamp(tmp_path):
    config = replace(
        _e2e_config("e2e_batch", tags=("tag_a",), min_samples=1000),
        resample_interval_ms=100,
    )
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=tmp_path / "model_data",
        train_fn=lambda samples: None,
    )
    # DX1가 100ms 주기로 취득한 값 10개를 한 메시지로 묶어 보내고, 다음 메시지가 이어진다
    first = {"records": [{"timestamp": _ts(i, 100), "tag_a": i} for i in range(10)]}
    second = {"records": [{"timestamp": _ts(10, 100), "tag_a": 10}]}
    for message in (first, second):
        payload = json.dumps(message).encode("utf-8")
        pipeline.mqtt_subscriber._handle_message(None, None, SimpleNamespace(payload=payload))
    pipeline.snapshotter.process_pending()

    lines = (tmp_path / "calibration_data" / "e2e_batch.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["values"]["tag_a"] for row in rows] == list(range(10))
    stamps = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    assert [b - a for a, b in zip(stamps, stamps[1:])] == [timedelta(milliseconds=100)] * 9


def test_pipeline_worker_thread_processes_messages_end_to_end(tmp_path):
    config = _e2e_config("e2e_thread", tags=("tag_a",))
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=tmp_path / "model_data",
        train_fn=lambda samples: None,
    )

    pipeline.snapshotter.start()
    for i in range(6):
        _send(pipeline, i, config.tags)
    pipeline.snapshotter.stop()

    lines = (tmp_path / "calibration_data" / "e2e_thread.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 5


def test_pipeline_end_to_end_with_real_train_fn(tmp_path, capsys):
    """subscriber_cli가 하는 것과 동일한 방식으로 make_train_fn을 배선해,
    MQTT 메시지 → 스냅샷 → train 명령 → 모델 파일 저장까지 실제로 도는지 확인한다.
    (command_subscriber가 예외를 삼키므로 배선 버그는 이 경로 없이는 안 잡힌다.)"""
    config = _e2e_config("e2e_train")
    model_dir = tmp_path / "model_data"
    # wrapped_train_fn(pipeline.py)이 학습 직후 model_artifact_path(model_dir, equipment_id)에서
    # 즉시 다시 읽어들이므로, 여기 model_path도 그 규칙과 일치해야 한다.
    model_path = model_dir / f"{config.equipment_id}.pt"
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=_train_fn_for(config, model_dir),
    )

    _feed(pipeline, config.tags, 0, 20)

    buffer_path = tmp_path / "calibration_data" / "e2e_train.jsonl"
    recorded_lines = buffer_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(recorded_lines) >= config.calibration.min_samples

    _send_train(pipeline)

    # 학습이 실패하면 command_subscriber가 예외를 삼키므로 원인을 출력에서 보여준다
    assert model_path.exists(), capsys.readouterr().out
    assert pipeline.calibration_manager.state == CalibrationState.MONITORING
    artifact = load_artifact(model_path)
    assert artifact.tags == config.tags
    assert artifact.resample_interval_ms == config.resample_interval_ms


def test_pipeline_scores_and_publishes_after_training(tmp_path):
    config = _e2e_config("e2e_score")
    model_dir = tmp_path / "model_data"
    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=_train_fn_for(config, model_dir),
    )

    _feed(pipeline, config.tags, 0, 20)
    _send_train(pipeline)
    assert pipeline.calibration_manager.state == CalibrationState.MONITORING
    assert pipeline.inference_engine_holder.get() is not None

    published = []

    def _fake_publish(topic, payload):
        published.append((topic, payload))
        return SimpleNamespace(rc=0)  # ResultPublisher가 rc를 확인한다

    pipeline.mqtt_subscriber.client.publish = _fake_publish

    _feed(pipeline, config.tags, 20, 30)

    assert published, "MONITORING 진입 후에도 이상 점수가 발행되지 않았다"
    topic, payload = published[0]
    assert topic == config.publish_topic
    record = json.loads(payload)["records"][0]
    assert "jetson:anomaly_score" in record
    assert "jetson:alarm" in record
    assert record["jetson:top_deviant_tag"] in config.tags


def test_pipeline_resumes_monitoring_after_restart(tmp_path):
    config = _e2e_config("e2e_resume")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed(first, config.tags, 0, 20)
    _send_train(first)
    assert first.calibration_manager.state == CalibrationState.MONITORING

    # "재시작": 같은 calibration_dir/model_dir로 파이프라인을 새로 만든다
    second = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    assert second.calibration_manager.state == CalibrationState.MONITORING
    assert second.inference_engine_holder.get() is not None


def _feed_and_train(pipeline, tags, count=20):
    """MQTT 메시지를 흘려 캘리브레이션 버퍼를 채운 뒤 train 명령까지 보낸다."""
    _feed(pipeline, tags, 0, count)
    _send_train(pipeline)


def test_pipeline_falls_back_to_calibrating_when_config_window_size_changed(tmp_path):
    """모델 학습 후 config의 window_size가 바뀌면, 모델은 정상 로드되지만
    InferenceEngine.score()가 매 스텝 None을 반환해 아무 것도 발행하지 않는다.
    손상된 모델 파일과 동일하게 CALIBRATING으로 폴백해야 한다."""
    config = _e2e_config("e2e_wsmismatch")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed_and_train(first, config.tags)
    assert first.calibration_manager.state == CalibrationState.MONITORING

    # "재시작": config YAML의 window_size만 바뀐 채로 같은 model_dir을 다시 연다
    changed = replace(config, window_size=5)
    second = build_pipeline(
        config=changed,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )

    assert second.calibration_manager.state == CalibrationState.CALIBRATING
    assert second.inference_engine_holder.get() is None


def test_pipeline_falls_back_to_calibrating_when_config_tags_changed(tmp_path):
    config = _e2e_config("e2e_tagmismatch")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed_and_train(first, config.tags)
    assert first.calibration_manager.state == CalibrationState.MONITORING

    # "재시작": tag_b가 tag_c로 교체된 config
    changed = replace(config, tags=("tag_a", "tag_c"))
    second = build_pipeline(
        config=changed,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )

    assert second.calibration_manager.state == CalibrationState.CALIBRATING
    assert second.inference_engine_holder.get() is None


def test_pipeline_falls_back_to_calibrating_when_resample_interval_changed(tmp_path):
    """학습 때와 다른 격자 간격으로 윈도우를 만들면 모델 입력의 시간 간격이 달라져
    점수가 무의미해진다. window_size/tags가 바뀐 경우와 마찬가지로 CALIBRATING 폴백."""
    config = _e2e_config("e2e_gridmismatch")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed_and_train(first, config.tags)
    assert first.calibration_manager.state == CalibrationState.MONITORING

    changed = replace(config, resample_interval_ms=10)
    second = build_pipeline(
        config=changed,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )

    assert second.calibration_manager.state == CalibrationState.CALIBRATING
    assert second.inference_engine_holder.get() is None


def test_pipeline_falls_back_to_calibrating_for_artifact_without_grid_info(tmp_path):
    """격자 정보가 없는 기존 artifact(resample_interval_ms=0)는 어떤 config와도 호환되지
    않는 것으로 보고 재학습하게 한다."""
    config = _e2e_config("e2e_legacy_artifact", tags=("tag_a",))
    model_dir = tmp_path / "model_data"
    model = AnomalyGRU(
        num_tags=1, continuous_indices=[0], binary_indices=[], hidden_size=2, num_layers=1
    )
    save_artifact(
        model_dir / f"{config.equipment_id}.pt",
        ModelArtifact(
            tags=config.tags,
            tag_types={"tag_a": "continuous"},
            norm_stats={"tag_a": (0.0, 1.0)},
            error_stats={"tag_a": (0.0, 1.0)},
            window_size=config.window_size,
            hidden_size=2,
            num_layers=1,
            state_dict=model.state_dict(),
        ),
    )
    StateStore(model_dir / f"{config.equipment_id}.state").write(CalibrationState.MONITORING)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=lambda samples: None,
    )

    assert pipeline.calibration_manager.state == CalibrationState.CALIBRATING
    assert pipeline.inference_engine_holder.get() is None


def test_pipeline_stays_calibrating_after_recalibrate_even_though_model_file_remains(tmp_path):
    """recalibrate는 의도적으로 모델 파일을 지우지 않고 상태 마커만 되돌린다.
    그 직후 재시작하면 남아있는 모델로 MONITORING을 재개하는 게 아니라
    CALIBRATING에 머물러야 한다."""
    config = _e2e_config("e2e_recal_restart")
    model_dir = tmp_path / "model_data"
    train_fn = _train_fn_for(config, model_dir)

    first = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    _feed_and_train(first, config.tags)
    assert first.calibration_manager.state == CalibrationState.MONITORING

    first.command_subscriber._handle_command_message(
        None, None, SimpleNamespace(payload=b'{"command": "recalibrate"}')
    )
    assert first.calibration_manager.state == CalibrationState.CALIBRATING

    model_path = model_dir / f"{config.equipment_id}.pt"
    assert model_path.exists()  # 모델 파일은 의도적으로 남아있어야 한다

    second = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=train_fn,
    )
    assert second.calibration_manager.state == CalibrationState.CALIBRATING
    assert second.inference_engine_holder.get() is None


def test_pipeline_falls_back_to_calibrating_when_model_file_corrupted(tmp_path):
    config = _make_config(tmp_path)
    model_dir = tmp_path / "model_data"
    model_dir.mkdir(parents=True)
    (model_dir / f"{config.equipment_id}.pt").write_bytes(b"not a real torch file")
    StateStore(model_dir / f"{config.equipment_id}.state").write(CalibrationState.MONITORING)

    pipeline = build_pipeline(
        config=config,
        calibration_dir=tmp_path / "calibration_data",
        model_dir=model_dir,
        train_fn=lambda samples: None,
    )

    assert pipeline.calibration_manager.state == CalibrationState.CALIBRATING
    assert pipeline.inference_engine_holder.get() is None
```

`tests/test_buffer.py`를 아래 내용으로 **전체 교체**한다(`TagBuffer` 테스트 제거, `clear` 테스트 유지).

```python
from jetson_app.buffer import Snapshot, SlidingWindow


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


def test_sliding_window_clear_empties_the_window():
    window = SlidingWindow(window_size=2)
    window.push(Snapshot(values={"a": 1}))
    window.push(Snapshot(values={"a": 2}))

    window.clear()

    assert window.to_list() == []
    assert window.is_full() is False
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_pipeline.py tests/test_buffer.py -q`
Expected: FAIL — `AttributeError: 'Pipeline' object has no attribute 'resampler'` / 파이프라인이 아직 `TagBuffer`를 사용.

- [ ] **Step 3: 구현**

`src/jetson_app/pipeline.py`를 아래 내용으로 **전체 교체**:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .buffer import SlidingWindow
from .calibration import (
    CalibrationBufferWriter,
    CalibrationManager,
    CalibrationSample,
    CalibrationState,
    StateStore,
    TrainFn,
)
from .command_subscriber import CommandSubscriber
from .config import EquipmentConfig
from .debounce import Debouncer
from .inference import ActiveModelHolder, InferenceEngine
from .mqtt_subscriber import MqttRecordSubscriber, Record
from .publisher import ResultPublisher
from .resampler import EventTimeResampler
from .snapshot_processor import SnapshotProcessor
from .training import ModelArtifact, load_artifact, model_artifact_path, state_marker_path


@dataclass(frozen=True)
class Pipeline:
    config: EquipmentConfig
    resampler: EventTimeResampler
    sliding_window: SlidingWindow
    calibration_manager: CalibrationManager
    inference_engine_holder: ActiveModelHolder
    snapshotter: SnapshotProcessor
    mqtt_subscriber: MqttRecordSubscriber
    command_subscriber: CommandSubscriber


def build_pipeline(
    config: EquipmentConfig,
    calibration_dir: str | Path,
    model_dir: str | Path,
    train_fn: TrainFn,
) -> Pipeline:
    resampler = EventTimeResampler(
        tags=config.tags,
        interval_ms=config.resample_interval_ms,
        max_lateness_ms=config.max_lateness_ms,
        window_size=config.window_size,
    )
    sliding_window = SlidingWindow(config.window_size)

    def on_record(record: Record) -> None:
        # 리샘플러가 이 record로 새로 확정한 격자 칸들을 처리 스레드의 큐로 넘긴다.
        # MQTT 스레드는 큐에 넣기만 하고, 점수 계산은 처리 스레드가 한다.
        snapshotter.submit(resampler.add(record))

    mqtt_subscriber = MqttRecordSubscriber(config, on_record=on_record)
    result_publisher = ResultPublisher(
        client=mqtt_subscriber.client, publish_topic=config.publish_topic
    )

    buffer_path = Path(calibration_dir) / f"{config.equipment_id}.jsonl"
    # 잘못된 --calibration-dir이 백그라운드 스레드가 아니라 기동 시점에 드러나도록
    # 디렉터리를 미리 만든다 (CLI의 OSError 처리에 걸린다).
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    buffer_writer = CalibrationBufferWriter(buffer_path)

    model_path = model_artifact_path(model_dir, config.equipment_id)
    state_path = state_marker_path(model_dir, config.equipment_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_store = StateStore(state_path)

    inference_engine_holder = ActiveModelHolder()
    debouncer = Debouncer()

    def _check_artifact_matches_config(artifact: ModelArtifact) -> None:
        """설정 YAML의 window_size/tags/resample_interval_ms가 학습 이후 바뀌면 모델은
        정상적으로 로드되지만 InferenceEngine.score()가 매 스텝 None을 반환하거나
        (window_size/tags), 학습 때와 다른 시간 간격의 윈도우로 엉뚱한 점수를 낸다
        (resample_interval_ms). 겉보기 상태는 정상 MONITORING이므로, 손상된 모델 파일과
        동일하게 취급한다. 격자 정보가 없는 기존 artifact(resample_interval_ms=0)도
        여기서 걸러져 재학습을 유도한다."""
        if (
            artifact.window_size != config.window_size
            or set(artifact.tags) != set(config.tags)
            or artifact.resample_interval_ms != config.resample_interval_ms
        ):
            raise ValueError(
                f"model artifact incompatible with current config: "
                f"window_size {artifact.window_size} vs {config.window_size}, "
                f"tags {artifact.tags} vs {config.tags}, "
                f"resample_interval_ms {artifact.resample_interval_ms} vs "
                f"{config.resample_interval_ms}"
            )

    def wrapped_train_fn(samples: list[CalibrationSample]) -> None:
        train_fn(samples)
        # 학습이 방금 성공적으로 저장한 모델을 즉시 메모리에 올려, 다음 스텝부터
        # 바로 채점을 시작할 수 있게 한다 (재시작을 기다릴 필요 없음).
        artifact = load_artifact(model_path)
        _check_artifact_matches_config(artifact)
        inference_engine_holder.set(InferenceEngine(artifact))
        # 새 모델은 오차 통계가 완전히 다르므로, 이전 모델 점수로 쌓인 연속 초과
        # 카운터를 물려받아 첫 스텝부터 알람이 확정되는 일이 없도록 리셋한다.
        debouncer.reset()

    calibration_manager = CalibrationManager(
        buffer_writer=buffer_writer,
        min_samples=config.calibration.min_samples,
        max_duration=config.calibration.max_duration,
        train_fn=wrapped_train_fn,
        state_store=state_store,
    )

    if calibration_manager.state == CalibrationState.MONITORING:
        try:
            artifact = load_artifact(model_path)
            _check_artifact_matches_config(artifact)
            inference_engine_holder.set(InferenceEngine(artifact))
            print(f"[build_pipeline] 저장된 모델을 불러와 MONITORING으로 재개: {model_path}")
        except Exception as e:
            # 모델 파일 손상/누락, 또는 config와 불일치 — MONITORING 진입을 막고
            # CALIBRATING으로 폴백해 사람이 재학습을 판단하도록 한다
            # (상위 문서 5절 에러 처리 원칙).
            print(f"[build_pipeline] 모델 로드 실패, CALIBRATING으로 폴백: {e}")
            calibration_manager.handle_recalibrate_command()

    snapshotter = SnapshotProcessor(
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        inference_engine_holder=inference_engine_holder,
        debouncer=debouncer,
        result_publisher=result_publisher,
    )

    command_subscriber = CommandSubscriber(config.command_topic, calibration_manager)
    command_subscriber.attach(mqtt_subscriber.client)

    return Pipeline(
        config=config,
        resampler=resampler,
        sliding_window=sliding_window,
        calibration_manager=calibration_manager,
        inference_engine_holder=inference_engine_holder,
        snapshotter=snapshotter,
        mqtt_subscriber=mqtt_subscriber,
        command_subscriber=command_subscriber,
    )
```

`src/jetson_app/buffer.py`를 아래 내용으로 **전체 교체**(`TagBuffer`와 `import threading` 제거):

```python
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Snapshot:
    values: dict[str, float | int | None]


class SlidingWindow:
    """고정 크기 롤링 윈도우. 용량을 넘으면 가장 오래된 항목을 버린다."""

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._window_size = window_size
        self._items: deque[Snapshot] = deque(maxlen=window_size)

    @property
    def window_size(self) -> int:
        return self._window_size

    def push(self, snapshot: Snapshot) -> None:
        self._items.append(snapshot)

    def clear(self) -> None:
        self._items.clear()

    def is_full(self) -> bool:
        return len(self._items) == self._window_size

    def to_list(self) -> list[Snapshot]:
        return list(self._items)
```

구 스케줄러 제거:

```bash
git rm jetson_app/src/jetson_app/scheduler.py jetson_app/tests/test_scheduler.py
```

- [ ] **Step 4: 전체 통과 확인**

Run: `uv run pytest -q`
Expected: PASS (전체). 남은 참조가 없는지도 확인: `uv run python -c "import jetson_app.pipeline, jetson_app.subscriber_cli"` 가 오류 없이 끝나야 한다.

- [ ] **Step 5: 커밋**

```bash
git add jetson_app/src/jetson_app/pipeline.py jetson_app/src/jetson_app/buffer.py jetson_app/tests/test_pipeline.py jetson_app/tests/test_buffer.py
git commit -m "feat: drive pipeline from event-time resampler and snapshot worker" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: 문서 갱신과 최종 검증

**Files:**
- Modify: `README.md`, `docs/superpowers/specs/2026-10-01-event-time-resampler-design.md`

- [ ] **Step 1: README 수정** — `jetson_app/README.md`를 네 군데 수정한다.

(a) 3행: `…여러 토픽에서 구독해 Tag Buffer에 모으고, 주기적으로 스냅샷을 떠서 슬라이딩 윈도우와 캘리브레이션 버퍼에 쌓는다.` 를 다음으로 교체:

```
…여러 토픽에서 구독해, 각 record의 `timestamp`(이벤트 시각)를 기준으로 `resample_interval_ms` 격자에 배치(이벤트 시간 리샘플러)하고, 확정된 칸을 슬라이딩 윈도우와 캘리브레이션 버퍼에 쌓는다.
```

(b) 7행 "현재 범위" 문장에서 `Tag Buffer/슬라이딩 윈도우 + 주기 스냅샷 스케줄러` 를 `이벤트 시간 리샘플러/슬라이딩 윈도우 + 스냅샷 처리 스레드` 로, 마지막 문장 `실시간 이상 점수 계산/디바운스/Result Publisher는 이후 별도 계획.` 을 `+ 실시간 이상 점수 계산/디바운스/Result Publisher.` 로 바꾼다.

(c) 하트비트 설명 `약 5초에 한 번씩` 을 `스냅샷 100개마다(격자가 50ms면 약 5초에 한 번)` 로 바꾸고, 같은 문단의 `하트비트가 전혀 안 나오면 아직 태그 값이 하나도 안 들어온 것이다` 뒤에 `(또는 `max_lateness_ms`만큼 기다리는 중이다)` 를 덧붙인다.

(d) `### 문제 발생 시 점검 순서` 바로 앞에 새 절을 추가:

````markdown
### 취득 주기와 `resample_interval_ms`, `max_lateness_ms`

- DX1은 Collector 주기(예: 100ms)로 취득한 값을 여러 개 묶어 한 MQTT 메시지로 보낼 수 있다. Jetson은 도착 시각이 아니라 **각 record의 `timestamp`** 로 값을 배치하므로, 메시지에 record가 몇 개 들어 있든 시점이 보존된다.
- `resample_interval_ms`는 모델이 보는 시간 간격이다. **DX1 취득 주기에 맞춰 config에 직접 적는다**(예: 취득이 100ms면 `100`, 더 줄이면 같이 줄인다). 격자보다 빠른 신호는 한 칸 안의 마지막 값을, 느린 신호는 직전 값을 유지(ffill)한다.
- `max_lateness_ms`(기본 2000)는 토픽 간 도착 시차를 기다리는 시간이다. **토픽 중 가장 긴 배치 간격보다 크게** 잡는다. 너무 작으면 늦게 도착한 record가 `[resampler: …] 누적 N건` 로그와 함께 폐기되고, 클수록 결과 발행이 그만큼 늦어진다.
- 학습 모델에 격자 값이 함께 저장된다. **`resample_interval_ms`를 바꾸면** 저장된 모델과 호환되지 않아 재시작 시 `CALIBRATING`으로 폴백한다. 캘리브레이션 버퍼도 이전 간격의 데이터와 섞이므로, 격자를 바꿀 때는 반드시 `recalibrate` 명령을 보내 버퍼를 비우고 다시 시작한다.
- DX1/PLC의 시계가 크게 앞서 가면(윈도우 길이보다 긴 공백) 슬라이딩 윈도우를 비우고 새 시점부터 다시 쌓는다.
````

- [ ] **Step 2: 스펙 문서 상태 갱신** — `docs/superpowers/specs/2026-10-01-event-time-resampler-design.md` 상단의 `- Status: Approved (design, 2026-10-01 대화에서 승인), 구현 계획 작성 전` 을 `- Status: Implemented (2026-10-01), 구현 계획: ../plans/2026-10-01-event-time-resampler.md` 로 바꾸고, 문서 끝에 아래 절을 추가한다.

```markdown

## 구현 노트

- 스펙의 `buffer.py`/`scheduler.py` 수정 대신 `resampler.py`(`EventTimeResampler`, `ResampledStep`)와 `snapshot_processor.py`(`SnapshotProcessor`)를 새로 만들고, 기존 `TagBuffer`와 `scheduler.py`는 제거했다. 동작은 스펙과 같다.
- 처리 스레드는 큐를 `get(timeout=0.1)`로 소비하고, `stop()`은 스레드를 join한 뒤 남은 큐를 호출 스레드에서 마저 처리한다. 테스트는 `process_pending()`으로 스레드 없이 결정적으로 검증한다.
- 격자 정보가 없는 기존 artifact는 `resample_interval_ms = 0`으로 읽혀 항상 불일치로 처리된다(재학습 유도).
- **미해결(추후 보강):** record 값의 타입 검증(문자열/`null`/`true` 등 비숫자 값)은 아직 하지 않는다. 실제 토픽 샘플을 확보한 뒤 보강한다.
```

- [ ] **Step 3: 전체 테스트와 import 확인**

Run: `uv run pytest -q`
Expected: PASS (전체). 또한 `uv run python -c "import jetson_app.subscriber_cli"` 가 오류 없이 끝나야 한다.

- [ ] **Step 4: 수동 스모크 테스트(로컬 Mosquitto, 선택)** — 실제 배치 메시지로 end-to-end를 한 번 확인한다. 터미널 A에서 앱을 띄운다(임시 config: `tags`에는 `PLC_Collector_Actuator_1:AirBlower.Cmd[0]` 하나만 두고 `resample_interval_ms: 100`, `window_size: 10`, `min_samples: 20`, `max_lateness_ms: 0`으로 설정).

```bash
uv run jetson-app --config configs/test_dx1.yaml --host localhost
```

터미널 B에서 100ms 간격 record 10개짜리 메시지를 두 번 보낸다(두 번째 메시지가 첫 메시지의 마지막 칸을 확정시킨다).

```bash
mosquitto_pub -h localhost -t "dx1/test_dx1/actuator_1" -m '{"records":[{"timestamp":"2026-10-01T00:47:01.000000000+0000","PLC_Collector_Actuator_1:AirBlower.Cmd[0]":0},{"timestamp":"2026-10-01T00:47:01.100000000+0000","PLC_Collector_Actuator_1:AirBlower.Cmd[0]":1},{"timestamp":"2026-10-01T00:47:01.200000000+0000","PLC_Collector_Actuator_1:AirBlower.Cmd[0]":0},{"timestamp":"2026-10-01T00:47:01.300000000+0000","PLC_Collector_Actuator_1:AirBlower.Cmd[0]":1}]}'
mosquitto_pub -h localhost -t "dx1/test_dx1/actuator_1" -m '{"records":[{"timestamp":"2026-10-01T00:47:01.400000000+0000","PLC_Collector_Actuator_1:AirBlower.Cmd[0]":0}]}'
```

Expected: `calibration_data/test_dx1.jsonl`에 4줄이 쌓이고, 각 줄의 `timestamp`가 `2026-10-01T00:47:01.000000+00:00`, `…01.100000+00:00`, `…01.200000+00:00`, `…01.300000+00:00`(100ms 간격)이며 값이 `0, 1, 0, 1` 순서다. 확인 후 `calibration_data/`와 `model_data/`의 테스트 산출물은 삭제한다.

- [ ] **Step 5: 커밋**

```bash
git add jetson_app/README.md docs/superpowers/specs/2026-10-01-event-time-resampler-design.md
git commit -m "docs: document event-time resampling and mark spec implemented" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-Review 기록

- **스펙 커버리지:** 1절(타임스탬프 파싱) → Task 1·2, 2절(리샘플러: 칸 배치/워터마크/늦은 record/긴 공백/큐 출력) → Task 4·5, 3절(처리 스레드/이벤트 시각 발행) → Task 5·7, 4절(config/artifact 격자/제약) → Task 3·6·7·8, 5절(에러 처리 표) → Task 2(타임스탬프 폐기), 4(늦은 record, 공백), 5(큐 초과), 7(artifact 불일치), 6절(테스트 계획) → 각 Task의 테스트, 7절(영향 파일) → File Structure.
- **타입/이름 일관성:** `ResampledStep(epoch_ns, snapshot, reset_window)`, `EventTimeResampler.add/late_dropped`, `SnapshotProcessor.submit/process/process_pending/start/stop/dropped`, `DropCounter.add/total`, `ModelArtifact.resample_interval_ms`, `make_train_fn(..., resample_interval_ms)`가 모든 Task에서 동일.
- **검증:** 이 계획의 코드와 테스트는 저장소 밖 복사본(scratchpad)에 Task 1~7을 모두 적용해 실행했고, 기존 131개 + 신규를 합친 전체 테스트 183개가 통과했다.
