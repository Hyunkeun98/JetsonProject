# jetson-studio 6단계 흐름 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** jetson-studio를 "연결 → 살펴보기 → 전체 수집 → 태그 추천 → 학습 → 감시" 6단계로 바꿔, 저장해 둔 데이터 없이 브로커에서 모은 원본으로 태그를 추천받고 학습한다.

**Architecture:** 브로커 payload를 원본 그대로 `workspace/collections/<id>/current/capture.jsonl`에 모으고(`Collector`), 그 폴더에 기존 `analyze_tags`와 `replay.load_records` 계열을 그대로 적용한다. 사이클 수는 태그 이름을 가정하지 않고 0/1 신호·카운터·주기 추정(`CycleCounter`)으로 센다. 학습은 파이프라인 없이 수집본에서 하고, 파이프라인은 6단계 감시에서만 만든다.

**Tech Stack:** Python 3.8, 표준 라이브러리 HTTP 서버, paho-mqtt<2, pytest, uv. torch는 쓰지 않는다(exe 제외 유지).

**Spec:** `docs/superpowers/specs/2026-10-15-studio-six-step-flow-design.md` (앞선 설계 `2026-10-14-studio-ui-design.md`의 보안·패키징 규칙은 그대로)

## Global Constraints

- Python 3.8 호환: 모듈 첫 줄 근처에 `from __future__ import annotations`, 런타임에 `list[...]`, `X | None`을 쓰지 않는다.
- studio 패키지는 torch를 import하지 않는다(`tests/test_torch_free.py`가 지킨다). 학습은 `replay.train_from_files`가 아니라 `trainer.train_from_collection`을 쓴다(그쪽은 torch를 부른다).
- 파일 쓰기는 `workspace/` 아래로만. 장비 이름은 `^[A-Za-z0-9_-]{1,40}$`.
- 보안 규칙 유지: 루프백 바인딩, 비루프백은 `--token` 없으면 POST 403, `X-Studio-Token`, Host/Origin 검사, 본문 64KB, 음수 Content-Length 400. 모든 동적 문자열은 `textContent`로만 넣는다(`innerHTML` 금지).
- 사이클 권장 수 `CYCLE_TARGET = 10`. 부족하면 **경고만** 하고 학습은 막지 않는다. `min_samples` 입력칸은 없앤다.
- 수집 상한 기본 `1 << 30`(1GB). 기본 포트 8800.
- 시험·수집 데이터는 저장소에 올리지 않는다(`workspace/`는 `.gitignore`). 커밋 메시지 끝에 `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- 파이썬 실행은 `uv run`. 작업 디렉터리는 `C:\WORK\10. Jetson\jetson_app`.

## Review Focus

스펙이 암시하지만 개별 테스트가 직접 다루지 않는 입력·조건 중 사용자가 겪기 쉬운 것. 각 줄의 테스트는 해당 Task에 들어 있다.

1. 수집 중 앱이 꺼지거나 연결이 끊긴 수집본 → "중단된 수집본"으로 표시하고 그때까지의 데이터는 쓸 수 있다 (Task 3 `test_a_capture_that_was_never_finished_is_reported_as_interrupted`).
2. payload 안에 줄바꿈이 있어도 한 줄에 하나씩 저장된다 (Task 3).
3. 브로커에 못 붙었을 때 기존 수집본이 그대로 남는다 (Task 3).
4. 사이클 신호로 오인하기 쉬운 것: 매 샘플 바뀌는 타이머/정수 카운터, 한 번만 바뀐 0/1 신호 (Task 1).
5. 라인이 멈춘 상태만 모은 데이터 → 추천이 비고, 학습은 "값이 변하지 않아" 사유로 실패하고 수집본은 남는다 (Task 5).
6. 새로 수집을 시작하면 이전 학습 결과는 더 이상 감시에 쓰이지 않는다 (Task 5).
7. 수집 크기 상한에 닿으면 자동 종료되고 화면에 이유가 나온다 (Task 3, Task 7).

---

## File Structure

| 파일 | 역할 |
|---|---|
| `src/jetson_app/studio/payloads.py` (새) | DX1 payload에서 레코드·숫자 값 꺼내기 |
| `src/jetson_app/studio/cycles.py` (새) | `CycleCounter`: 0/1 신호·카운터·주기 추정으로 사이클 수 |
| `src/jetson_app/studio/raw_source.py` (새) | 브로커 구독 어댑터 `RawSource` |
| `src/jetson_app/studio/probe.py` (새) | 2단계 시험 세션 `ProbeSession` |
| `src/jetson_app/studio/collector.py` (새) | 3단계 `Collector`: 원본 저장, 상한, archive, 폴더 불러오기 |
| `src/jetson_app/studio/trainer.py` (새) | `train_from_collection`: 수집본으로 deviation 학습 |
| `src/jetson_app/studio/wizard.py` (교체) | 연결 정보 저장, 태그 분석, 선택 태그로 YAML 생성 |
| `src/jetson_app/studio/controller.py` (교체) | 6단계 상태기계 |
| `src/jetson_app/studio/server.py` (수정) | 새 라우트 |
| `src/jetson_app/studio/cli.py` (수정) | 종료 시 `shutdown()` |
| `src/jetson_app/web/studio.html` (교체) | 6단계 화면 |
| `tests/studio_fakes.py` (새) | 가짜 구독/IO, payload 만들기 |
| `tests/test_studio_*.py` | 새·재작성 |

실행 방법: 이 계획서의 `<!-- file: 경로 -->` 블록은 통째로 그 파일의 내용이다. 스크래치 폴더의 `extract_plan.py`로 꺼낸다:
`uv run python <scratchpad>/extract_plan.py docs/superpowers/plans/2026-10-15-studio-six-step-flow.md <경로>...`

---

### Task 1: 사이클 세기 (`payloads.py`, `cycles.py`)

**Files:**
- Create: `src/jetson_app/studio/payloads.py`, `src/jetson_app/studio/cycles.py`, `tests/studio_fakes.py`
- Test: `tests/test_studio_cycles.py`

**Interfaces:**
- Produces: `iter_records(payload: bytes) -> Iterator[dict]`, `numeric_value(value) -> int|float|None`.
- Produces: `CycleCounter.feed_payload(payload: bytes) -> None`, `.report(preferred_tag: str|None = None) -> CycleReport`, `.duration_seconds: float`.
- Produces: `CycleReport(method, cycles, exact, tag, period_seconds, duration_seconds, candidates, message)` with `.to_dict()`; `method` ∈ `edge|counter|period|none`.
- Produces: `cycle_warning(report) -> str|None`, `scan_directory(data_dir) -> CycleCounter`, `CYCLE_TARGET = 10`.
- Produces (test helper): `studio_fakes.ts(i)`, `noise(i)`, `payload(i, **values) -> bytes`, `FakeSource`, `SourceFactory`, `FakeIO`.

- [ ] **Step 1: 시험 도구와 실패하는 테스트를 만든다**

<!-- file: tests/studio_fakes.py -->
```python
"""studio 시험용 가짜 부품과 DX1 payload 만들기."""
import json
from datetime import datetime, timedelta, timezone


def ts(i):
    moment = datetime(2026, 8, 4, tzinfo=timezone.utc) + timedelta(milliseconds=i * 100)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000+0000"


def noise(i):
    return 0.001 * (((i * 7919) % 13) - 6) / 6.0


def payload(i, **values):
    """100ms 간격 i번째 DX1 payload(레코드 하나)."""
    return json.dumps({"records": [dict(timestamp=ts(i), **values)]}).encode("utf-8")


class FakeSource:
    """RawSource와 같은 모양의 가짜 구독. emit()으로 payload를 흘려 넣는다."""

    def __init__(self, fail=False):
        self.fail = fail
        self.connected = False
        self.calls = []
        self._on_payload = None

    def connect(self, host, port, topics, on_payload):
        if self.fail:
            raise ConnectionRefusedError("refused")
        self.connected = True
        self.calls.append((host, port, tuple(topics)))
        self._on_payload = on_payload

    def disconnect(self):
        self.connected = False

    def is_connected(self):
        return self.connected

    def emit(self, payload_bytes, topic="dx1/data"):
        self._on_payload(topic, payload_bytes)


class SourceFactory:
    def __init__(self, fail=False):
        self.fail = fail
        self.created = []

    def __call__(self):
        source = FakeSource(self.fail)
        self.created.append(source)
        return source

    @property
    def last(self):
        return self.created[-1]


class FakeIO:
    """감시 파이프라인의 MQTT 연결을 흉내 낸다."""

    def __init__(self, fail=False):
        self.fail = fail
        self.up = False
        self.calls = []

    def connect(self, pipeline, host, port):
        if self.fail:
            raise ConnectionRefusedError("refused")
        self.up = True
        self.calls.append((host, port))

    def disconnect(self, pipeline):
        self.up = False

    def is_connected(self, pipeline):
        return self.up
```

<!-- file: tests/test_studio_cycles.py -->
```python
import json
import random

from jetson_app.studio.cycles import CYCLE_TARGET, CycleCounter, cycle_warning, scan_directory
from studio_fakes import payload


def _feed(counter, steps, **series):
    for i in range(steps):
        counter.feed_payload(payload(i, **{name: fn(i) for name, fn in series.items()}))
    return counter


def _ramp(i):
    k = i % 300                       # 30초 주기: 10초 올라가고 10초 내려가고 10초 쉰다
    return float(k if k < 100 else 200 - k if k < 200 else 0)


def test_rising_edges_of_a_binary_signal_are_counted_and_the_smallest_is_the_default():
    counter = _feed(
        CycleCounter(), 1200,
        U0=lambda i: 1 if i % 100 >= 90 else 0,
        slow=lambda i: 1 if i % 200 >= 190 else 0,
    )

    report = counter.report()

    assert report.method == "edge" and report.exact is True
    assert report.tag == "slow" and report.cycles == 6
    assert {(c.tag, c.cycles) for c in report.candidates} == {("U0", 12), ("slow", 6)}
    assert counter.report("U0").cycles == 12 and counter.report("U0").tag == "U0"
    json.dumps(report.to_dict())                          # 그대로 JSON 응답으로 보낼 수 있어야 한다


def test_a_boolean_signal_counts_like_0_and_1():
    counter = _feed(CycleCounter(), 500, run=lambda i: (i % 100) >= 50)

    assert counter.report().cycles == 5


def test_an_increasing_integer_counter_counts_its_increments():
    counter = _feed(CycleCounter(), 1200, parts=lambda i: i // 100)

    report = counter.report()

    assert report.method == "counter" and report.cycles == 11 and report.exact is True


def test_a_timer_that_changes_every_sample_is_not_a_cycle_signal():
    counter = _feed(CycleCounter(), 600, timer=lambda i: i)

    assert counter.report().method == "none"


def test_a_single_edge_is_not_a_cycle():
    counter = _feed(CycleCounter(), 300, started=lambda i: 1 if i > 100 else 0)

    assert counter.report().method == "none"


def test_period_is_estimated_from_repeating_motion_when_there_is_no_cycle_signal():
    rng = random.Random(7)
    counter = _feed(CycleCounter(), 3000, pos=_ramp, vel=lambda i: rng.uniform(-1, 1))

    report = counter.report()

    assert report.method == "period" and report.exact is False and report.tag is None
    assert 28 <= report.period_seconds <= 32
    assert 9 <= report.cycles <= 11


def test_noise_without_any_repetition_gives_no_cycle_count():
    rng = random.Random(3)
    counter = _feed(CycleCounter(), 3000, vel=lambda i: rng.uniform(-1, 1))

    report = counter.report()

    assert report.method == "none" and report.cycles is None
    assert report.duration_seconds > 290 and "수집 시간" in report.message


def test_a_recording_shorter_than_three_periods_is_not_estimated():
    counter = _feed(CycleCounter(), 400, pos=_ramp)          # 40초: 주기 30초의 3배가 안 된다

    assert counter.report().method == "none"


def test_warning_only_below_ten_cycles_and_it_says_when_the_number_is_an_estimate():
    enough = _feed(CycleCounter(), 1200, U0=lambda i: 1 if i % 100 >= 90 else 0).report()
    few = _feed(CycleCounter(), 700, U0=lambda i: 1 if i % 100 >= 90 else 0).report()
    unknown = _feed(CycleCounter(), 100, vel=lambda i: i % 2).report()
    estimated = _feed(CycleCounter(), 1800, pos=_ramp).report()

    assert CYCLE_TARGET == 10
    assert cycle_warning(enough) is None
    assert "6" in cycle_warning(few) and "추정" not in cycle_warning(few)
    assert "알 수 없습니다" in cycle_warning(unknown)
    assert estimated.method == "period" and "추정" in cycle_warning(estimated)


def test_scan_directory_reads_data_files_only(tmp_path):
    lines = [payload(i, U0=1 if i % 100 >= 90 else 0) for i in range(1200)]
    (tmp_path / "a.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    (tmp_path / "capture.meta").write_text('{"x": 1}', encoding="utf-8")

    assert scan_directory(tmp_path).report().cycles == 12
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_studio_cycles.py -q`
Expected: FAIL (`ModuleNotFoundError: jetson_app.studio.cycles`)

- [ ] **Step 3: 구현한다**

<!-- file: src/jetson_app/studio/payloads.py -->
```python
"""DX1 payload({"records": [...]})에서 숫자 값을 꺼내는 공용 도구."""
from __future__ import annotations

import json
import math
from typing import Iterator


def iter_records(payload: bytes) -> Iterator[dict]:
    try:
        data = json.loads(payload)
    except ValueError:
        return
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return
    for record in records:
        if isinstance(record, dict):
            yield record


def numeric_value(value):
    """숫자(불리언은 0/1)이면 그 값, 아니면 None. NaN과 무한대는 버린다."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None
```

<!-- file: src/jetson_app/studio/cycles.py -->
```python
"""수집한 원본 데이터에서 사이클 수를 센다. 태그 이름을 가정하지 않는다.

방식 A(0/1 신호의 상승 에지)와 B(정수 카운터의 증가 횟수)는 정확하다. 둘 다 없으면 D(연속 신호의
움직임이 반복되는 주기를 자기상관으로 추정)를 쓰고, 그것도 안 되면 사이클 수를 모른다고 답한다.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from operator import mul
from pathlib import Path

from ..timeparse import parse_dx1_timestamp
from .payloads import iter_records, numeric_value

CYCLE_TARGET = 10
_BUCKET_NS = 500_000_000
_BUCKET_SECONDS = 0.5
_MAX_DISTINCT = 12          # 이보다 많은 값을 가지면 연속 신호로 본다
_MAX_BUCKETS = 2400         # 주기 추정에는 최근 20분만 쓴다
_MIN_LAG = 6                # 3초보다 짧은 주기는 찾지 않는다
_MIN_PERIODS = 3            # 수집 시간이 주기의 3배는 되어야 한다
_MIN_PEAK = 0.3
_MIN_PROMINENCE = 0.1
_AGREEMENT = 0.2            # 태그들이 구한 주기가 서로 20% 안에서 맞아야 한다
_MAX_PERIOD_TAGS = 8


@dataclass(frozen=True)
class CycleCandidate:
    tag: str
    method: str     # "edge" | "counter"
    cycles: int


@dataclass(frozen=True)
class CycleReport:
    method: str                 # "edge" | "counter" | "period" | "none"
    cycles: object              # float, 세지 못하면 None
    exact: bool
    tag: object                 # edge/counter일 때 기준 태그
    period_seconds: object
    duration_seconds: float
    candidates: tuple
    message: str

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "cycles": self.cycles,
            "exact": self.exact,
            "tag": self.tag,
            "period_seconds": self.period_seconds,
            "duration_seconds": self.duration_seconds,
            "candidates": [{"tag": c.tag, "method": c.method, "cycles": c.cycles} for c in self.candidates],
            "message": self.message,
            "warning": cycle_warning(self),
        }


def cycle_warning(report: CycleReport):
    """권장 사이클 수보다 적을 때의 경고. 학습을 막지는 않는다."""
    if report.cycles is None:
        return "사이클 수를 알 수 없습니다. 정상 사이클이 10회 이상 들어갔는지 직접 확인하세요."
    if report.cycles < CYCLE_TARGET:
        suffix = "" if report.exact else " (추정값이라 정확하지 않을 수 있습니다)"
        return (
            f"수집된 사이클이 약 {report.cycles:.0f}개로 권장 {CYCLE_TARGET}개보다 적습니다. "
            f"학습 데이터가 적어 알람 기준이 불안정할 수 있습니다.{suffix}"
        )
    return None


class _Tag:
    __slots__ = ("n", "last", "distinct", "ups", "downs", "all_int", "lo", "hi")

    def __init__(self) -> None:
        self.n = 0
        self.last = None
        self.distinct = set()
        self.ups = 0
        self.downs = 0
        self.all_int = True
        self.lo = None
        self.hi = None

    def add(self, value) -> None:
        self.n += 1
        if len(self.distinct) <= _MAX_DISTINCT:
            self.distinct.add(value)
        if self.last is not None:
            if value > self.last:
                self.ups += 1
            elif value < self.last:
                self.downs += 1
        if self.all_int and not float(value).is_integer():
            self.all_int = False
        self.lo = value if self.lo is None or value < self.lo else self.lo
        self.hi = value if self.hi is None or value > self.hi else self.hi
        self.last = value

    def is_continuous(self) -> bool:
        return len(self.distinct) > _MAX_DISTINCT


class CycleCounter:
    def __init__(self) -> None:
        self._tags = {}
        self._activity = {}        # 태그 -> {0.5초 칸: 그 칸에서 값이 움직인 총량}
        self._t_min = None
        self._t_max = None

    @property
    def duration_seconds(self) -> float:
        if self._t_min is None:
            return 0.0
        return (self._t_max - self._t_min) / 1e9

    def feed_payload(self, payload: bytes) -> None:
        for record in iter_records(payload):
            ts = parse_dx1_timestamp(record.get("timestamp"))
            if ts is None:
                continue
            self._t_min = ts if self._t_min is None or ts < self._t_min else self._t_min
            self._t_max = ts if self._t_max is None or ts > self._t_max else self._t_max
            bucket = ts // _BUCKET_NS
            for key, raw in record.items():
                if key == "timestamp":
                    continue
                value = numeric_value(raw)
                if value is None:
                    continue
                tag = self._tags.get(key)
                if tag is None:
                    tag = self._tags[key] = _Tag()
                if tag.last is not None and value != tag.last:
                    series = self._activity.setdefault(key, {})
                    series[bucket] = series.get(bucket, 0.0) + abs(value - tag.last)
                tag.add(value)

    def candidates(self) -> list:
        found = []
        for name, tag in self._tags.items():
            if len(tag.distinct) == 2:
                if tag.ups >= 2:
                    found.append(CycleCandidate(name, "edge", tag.ups))
            elif (
                len(tag.distinct) >= 3
                and tag.downs == 0
                and tag.all_int
                and tag.ups >= 2
                and tag.ups * 2 <= tag.n          # 거의 매 샘플 바뀌는 정수(타이머)는 제외
            ):
                found.append(CycleCandidate(name, "counter", tag.ups))
        found.sort(key=lambda c: (c.cycles, 0 if c.method == "edge" else 1, c.tag))
        return found

    def report(self, preferred_tag=None) -> CycleReport:
        duration = self.duration_seconds
        found = self.candidates()
        if found:
            chosen = next((c for c in found if c.tag == preferred_tag), found[0])
            return CycleReport(
                chosen.method, float(chosen.cycles), True, chosen.tag, None, duration, tuple(found),
                f"'{chosen.tag}' 신호 기준 {chosen.cycles}사이클",
            )
        period = self._period_seconds()
        if period is not None:
            cycles = duration / period
            return CycleReport(
                "period", cycles, False, None, period, duration, (),
                f"주기 약 {period:.0f}초로 추정한 약 {cycles:.0f}사이클 (정확하지 않을 수 있음)",
            )
        return CycleReport(
            "none", None, False, None, None, duration, (),
            "사이클 수를 구하지 못했습니다(사이클 신호나 반복되는 움직임이 없습니다). "
            f"수집 시간은 {duration / 60:.1f}분입니다.",
        )

    def _period_seconds(self):
        """연속 신호의 움직임이 되풀이되는 주기(초). 믿을 만한 주기가 없으면 None."""
        if self._t_min is None:
            return None
        last = self._t_max // _BUCKET_NS
        start = max(self._t_min // _BUCKET_NS, last - _MAX_BUCKETS + 1)
        series_by_tag = []
        for name, tag in self._tags.items():
            activity = self._activity.get(name)
            if not activity or not tag.is_continuous() or tag.hi == tag.lo:
                continue
            x = [activity.get(b, 0.0) for b in range(start, last + 1)]
            mean = sum(x) / len(x)
            if mean <= 0.0:
                continue
            burstiness = statistics.pstdev(x) / mean      # 쉬었다 움직이길 되풀이하면 크고, 늘 떨리는 잡음이면 작다
            series_by_tag.append((burstiness, [v - mean for v in x]))
        series_by_tag.sort(key=lambda item: item[0], reverse=True)
        periods = []
        for _, x in series_by_tag[:_MAX_PERIOD_TAGS]:
            lag = _first_peak(x)
            if lag is not None:
                periods.append(lag * _BUCKET_SECONDS)
        if not periods:
            return None
        middle = statistics.median(periods)
        agreeing = [p for p in periods if abs(p - middle) <= _AGREEMENT * middle]
        if len(agreeing) * 2 < len(periods):
            return None
        period = statistics.median(agreeing)
        return period if self.duration_seconds >= _MIN_PERIODS * period else None


def _first_peak(x):
    """평균을 뺀 열 x의 자기상관에서 처음 나오는 뚜렷한 봉우리의 지연(칸 수). 없으면 None."""
    max_lag = len(x) // _MIN_PERIODS
    if max_lag <= _MIN_LAG + 1:
        return None
    denominator = sum(map(mul, x, x))
    if denominator <= 0.0:
        return None
    r = [1.0] + [sum(map(mul, x, x[k:])) / denominator for k in range(1, max_lag + 2)]
    floor = min(r[1:_MIN_LAG])
    for k in range(_MIN_LAG, max_lag + 1):
        floor = min(floor, r[k - 1])
        if r[k] >= _MIN_PEAK and r[k] - floor >= _MIN_PROMINENCE and r[k] > r[k - 1] and r[k] >= r[k + 1]:
            return k
    return None


def scan_directory(data_dir) -> CycleCounter:
    """폴더의 *.json/*.jsonl(한 줄 = payload 하나)을 읽어 CycleCounter를 채운다."""
    counter = CycleCounter()
    for path in sorted(Path(data_dir).iterdir()):
        if path.suffix not in (".json", ".jsonl"):
            continue
        with path.open("rb") as f:
            for line in f:
                if line.strip():
                    counter.feed_payload(line)
    return counter
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_studio_cycles.py -q`
Expected: 10 passed. 실패하면 임계값이 아니라 먼저 실패 원인(데이터 모양, 봉우리 위치)을 확인한다.

- [ ] **Step 5: 커밋**

```bash
git add src/jetson_app/studio/payloads.py src/jetson_app/studio/cycles.py tests/studio_fakes.py tests/test_studio_cycles.py
git commit -m "feat: studio cycle counting (edges, counters, period estimate)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 브로커 구독 어댑터와 시험 세션 (`raw_source.py`, `probe.py`)

**Files:**
- Create: `src/jetson_app/studio/raw_source.py`, `src/jetson_app/studio/probe.py`
- Test: `tests/test_studio_probe.py`

**Interfaces:**
- Consumes: `payloads.iter_records`, `payloads.numeric_value`, `studio_fakes.FakeSource`.
- Produces: `RawSource().connect(host, port, topics, on_payload)`, `.disconnect()`, `.is_connected() -> bool`. `on_payload(topic: str, payload: bytes)`. 연결 실패는 `OSError`.
- Produces: `ProbeSession(source, host, port, topics, clock=time.monotonic, auto_stop_seconds=300.0)` with `.start()` (OSError 전달), `.stop()`, `.snapshot() -> dict` (키: `active, expired, connected, messages, rate, last_rx_age, last_message, topics, tags, tag_count`).

- [ ] **Step 1: 실패하는 테스트**

<!-- file: tests/test_studio_probe.py -->
```python
from types import SimpleNamespace

import pytest

from jetson_app.studio.probe import ProbeSession
from jetson_app.studio.raw_source import RawSource
from studio_fakes import FakeSource, payload


class FakeClient:
    def __init__(self):
        self.subscribed = []
        self.events = []
        self.on_connect = self.on_disconnect = self.on_message = None

    def connect(self, host, port, keepalive=60):
        self.events.append(("connect", host, port))

    def loop_start(self):
        self.events.append("loop_start")

    def loop_stop(self):
        self.events.append("loop_stop")

    def disconnect(self):
        self.events.append("disconnect")

    def subscribe(self, topic, qos=0):
        self.subscribed.append((topic, qos))


def test_raw_source_subscribes_on_connect_and_forwards_payloads(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr("paho.mqtt.client.Client", lambda: client)
    received = []
    source = RawSource()

    source.connect("broker", 1883, ["dx1/AxisData", "dx1/ProcStart"], lambda topic, data: received.append((topic, data)))
    assert source.is_connected() is False                        # 접속 응답(on_connect)을 받기 전

    client.on_connect(client, None, {}, 0)
    client.on_message(client, None, SimpleNamespace(topic="dx1/AxisData", payload=b"{}"))

    assert source.is_connected() is True
    assert client.subscribed == [("dx1/AxisData", 1), ("dx1/ProcStart", 1)]
    assert received == [("dx1/AxisData", b"{}")]
    client.on_disconnect(client, None, 1)
    assert source.is_connected() is False
    source.disconnect()
    assert client.events[-2:] == ["loop_stop", "disconnect"]


def test_raw_source_does_not_subscribe_when_the_broker_refuses(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr("paho.mqtt.client.Client", lambda: client)
    source = RawSource()
    source.connect("broker", 1883, ["t"], lambda topic, data: None)

    client.on_connect(client, None, {}, 5)

    assert client.subscribed == [] and source.is_connected() is False


def _probe(clock=None, **kwargs):
    source = FakeSource()
    now = clock if clock is not None else [0.0]
    session = ProbeSession(source, "broker", 1883, ["dx1/AxisData"], clock=lambda: now[0], **kwargs)
    return session, source, now


def test_probe_reports_counts_rate_last_message_and_numeric_tags():
    session, source, now = _probe()
    session.start()

    for i in range(10):
        now[0] = i * 0.5
        source.emit(payload(i, pos=1.5, state="RUN", on=True), topic="dx1/AxisData")
    now[0] = 5.0
    info = session.snapshot()

    assert info["active"] is True and info["connected"] is True
    assert info["messages"] == 10 and info["rate"] == pytest.approx(2.0)
    assert info["tags"] == ["on", "pos"] and info["tag_count"] == 2          # 문자열 값은 숫자 태그가 아니다
    assert '"pos":1.5' in info["last_message"].replace(" ", "")
    assert info["topics"] == ["dx1/AxisData"] and info["last_rx_age"] == pytest.approx(0.5)


def test_probe_stop_disconnects_and_ignores_later_payloads():
    session, source, _ = _probe()
    session.start()
    source.emit(payload(0, pos=1.0))

    session.stop()
    source.emit(payload(1, pos=2.0))

    assert source.connected is False
    assert session.snapshot()["messages"] == 1 and session.snapshot()["active"] is False


def test_probe_stops_itself_after_five_minutes():
    session, source, now = _probe()
    session.start()

    now[0] = 301.0
    info = session.snapshot()

    assert info["active"] is False and info["expired"] is True and source.connected is False


def test_probe_start_passes_a_refused_connection_on():
    session = ProbeSession(FakeSource(fail=True), "broker", 1883, ["t"])

    with pytest.raises(OSError):
        session.start()

    assert session.snapshot()["active"] is False


def test_probe_keeps_only_the_last_message_text_bounded():
    session, source, _ = _probe()
    session.start()
    source.emit(b'{"records":[{"timestamp":"x","pos":1}]}' + b" " * 5000)

    assert len(session.snapshot()["last_message"]) <= 2000
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_studio_probe.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: 구현**

<!-- file: src/jetson_app/studio/raw_source.py -->
```python
"""브로커 구독 어댑터: 받은 payload를 가공하지 않고 콜백으로 넘긴다(시험·수집이 같이 쓴다)."""
from __future__ import annotations


class RawSource:
    def __init__(self) -> None:
        self._client = None
        self._connected = False

    def connect(self, host: str, port: int, topics, on_payload) -> None:
        """접속을 시도한다. 브로커에 닿지 못하면 OSError. 접속 뒤에는 백그라운드 스레드가 받는다."""
        import paho.mqtt.client as mqtt

        client = mqtt.Client()

        def on_connect(c, userdata, flags, rc):
            self._connected = rc == 0
            if rc != 0:
                print(f"[studio] MQTT 연결 실패 (rc={rc}) — 브로커 인증/ACL 설정을 확인하세요")
                return
            for topic in topics:
                c.subscribe(topic, qos=1)

        def on_disconnect(c, userdata, rc):
            self._connected = False

        def on_message(c, userdata, msg):
            on_payload(msg.topic, msg.payload)

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message
        client.connect(host, port, keepalive=60)
        client.loop_start()
        self._client = client

    def disconnect(self) -> None:
        client, self._client = self._client, None
        self._connected = False
        if client is not None:
            client.loop_stop()
            client.disconnect()

    def is_connected(self) -> bool:
        return self._connected
```

<!-- file: src/jetson_app/studio/probe.py -->
```python
"""2단계 살펴보기: 토픽을 임시로 구독해 데이터가 들어오는지만 본다. 아무것도 저장하지 않는다."""
from __future__ import annotations

import threading
import time
from collections import deque

from .payloads import iter_records, numeric_value

AUTO_STOP_SECONDS = 300.0
RATE_WINDOW_SECONDS = 5.0
LAST_MESSAGE_CHARS = 2000
MAX_TAGS = 500


class ProbeSession:
    def __init__(self, source, host, port, topics, clock=time.monotonic, auto_stop_seconds=AUTO_STOP_SECONDS):
        self._source = source
        self._host = host
        self._port = port
        self._topics = list(topics)
        self._clock = clock
        self._auto_stop = auto_stop_seconds
        self._lock = threading.Lock()
        self._active = False
        self._expired = False
        self._started = 0.0
        self._messages = 0
        self._arrivals = deque()
        self._last_rx = None
        self._last_message = ""
        self._tags = set()
        self._seen_topics = set()

    def start(self) -> None:
        with self._lock:
            if self._active:
                return
            self._active = True
            self._expired = False
            self._started = self._clock()
        try:
            self._source.connect(self._host, self._port, self._topics, self._on_payload)
        except OSError:
            with self._lock:
                self._active = False
            raise

    def stop(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
        self._source.disconnect()

    def _on_payload(self, topic: str, payload: bytes) -> None:
        with self._lock:
            if not self._active:
                return
            now = self._clock()
            self._messages += 1
            self._arrivals.append(now)
            self._last_rx = now
            self._last_message = payload[:LAST_MESSAGE_CHARS].decode("utf-8", errors="replace")
            self._seen_topics.add(topic)
            for record in iter_records(payload):
                for key, raw in record.items():
                    if key != "timestamp" and numeric_value(raw) is not None and len(self._tags) < MAX_TAGS:
                        self._tags.add(key)

    def snapshot(self) -> dict:
        with self._lock:
            expired_now = self._active and self._clock() - self._started > self._auto_stop
        if expired_now:
            self.stop()
            with self._lock:
                self._expired = True
        with self._lock:
            now = self._clock()
            while self._arrivals and now - self._arrivals[0] > RATE_WINDOW_SECONDS:
                self._arrivals.popleft()
            window = min(RATE_WINDOW_SECONDS, max(now - self._started, 1e-3)) if self._active else 0.0
            return {
                "active": self._active,
                "expired": self._expired,
                "connected": bool(self._active and self._source.is_connected()),
                "messages": self._messages,
                "rate": len(self._arrivals) / window if window else 0.0,
                "last_rx_age": None if self._last_rx is None else now - self._last_rx,
                "last_message": self._last_message,
                "topics": sorted(self._seen_topics),
                "tags": sorted(self._tags),
                "tag_count": len(self._tags),
            }
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_studio_probe.py -q`
Expected: 7 passed

- [ ] **Step 5: 커밋**

```bash
git add src/jetson_app/studio/raw_source.py src/jetson_app/studio/probe.py tests/test_studio_probe.py
git commit -m "feat: studio broker source adapter and probe session" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: 전체 수집 (`collector.py`)

**Files:**
- Create: `src/jetson_app/studio/collector.py`
- Test: `tests/test_studio_collector.py`

**Interfaces:**
- Consumes: `cycles.CycleCounter`, `cycles.scan_directory`, `errors.StudioError`, `studio_fakes.SourceFactory`.
- Produces: `Collector(workspace, source_factory, clock=time.monotonic, max_bytes=DEFAULT_MAX_BYTES, report_interval=5.0)`:
  `set_equipment(equipment_id|None)`, `start(host, port, topics)`, `stop()`, `discard()`, `import_folder(path)`, `state() -> dict`, `cycle_report(preferred_tag=None) -> CycleReport`, `current_dir -> Path`.
- `state()` 키: `active, has_data, connected, messages, bytes, limit_bytes, seconds, no_data, interrupted, ended_reason, origin, cycles(dict|None)`.
- 저장: `workspace/collections/<id>/current/{capture.jsonl, capture.meta}`, 이전 수집본은 `archive/<시각>/`.

- [ ] **Step 1: 실패하는 테스트**

<!-- file: tests/test_studio_collector.py -->
```python
import json

import pytest

from jetson_app.studio.collector import Collector
from jetson_app.studio.errors import StudioError
from studio_fakes import SourceFactory, payload


def _collector(tmp_path, factory=None, clock=None, **kwargs):
    factory = factory or SourceFactory()
    now = clock if clock is not None else [0.0]
    collector = Collector(tmp_path, factory, clock=lambda: now[0], **kwargs)
    collector.set_equipment("nx5")
    return collector, factory, now


def _cycle_payload(i):
    return payload(i, pos=float(i), U0=1 if i % 100 >= 90 else 0)


def test_payloads_are_stored_as_they_arrive_and_cycles_are_counted(tmp_path):
    collector, factory, now = _collector(tmp_path)
    collector.start("broker", 1883, ["dx1/AxisData"])
    for i in range(1200):
        now[0] = i * 0.1
        factory.last.emit(_cycle_payload(i))
    collector.stop()

    lines = (tmp_path / "collections" / "nx5" / "current" / "capture.jsonl").read_bytes().splitlines()
    state = collector.state()

    assert len(lines) == 1200 and json.loads(lines[5])["records"][0]["pos"] == 5.0
    assert state["active"] is False and state["has_data"] is True and state["ended_reason"] == "user"
    assert state["messages"] == 1200 and state["bytes"] > 0 and state["seconds"] == pytest.approx(119.9, abs=0.2)
    assert state["cycles"]["method"] == "edge" and state["cycles"]["cycles"] == 12
    assert factory.last.connected is False and factory.last.calls == [("broker", 1883, ("dx1/AxisData",))]


def test_a_payload_with_line_breaks_is_stored_on_one_line(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])

    factory.last.emit(b'{\n  "records": [\n    {"timestamp": "2026-08-04T00:00:00.000000000+0000", "pos": 1}\n  ]\n}\n')
    collector.stop()

    lines = (tmp_path / "collections" / "nx5" / "current" / "capture.jsonl").read_bytes().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["records"][0]["pos"] == 1


def test_collection_ends_by_itself_at_the_size_limit(tmp_path):
    collector, factory, _ = _collector(tmp_path, max_bytes=3000)
    collector.start("broker", 1883, ["t"])

    for i in range(200):
        factory.last.emit(payload(i, pos=1.0))

    state = collector.state()
    assert state["active"] is False and state["ended_reason"] == "size"
    assert factory.last.connected is False
    assert state["bytes"] < 3000 + 200               # 상한을 크게 넘겨 쓰지 않는다


def test_starting_again_moves_the_previous_collection_to_archive(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=111.0))
    collector.stop()

    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=222.0))
    collector.stop()

    root = tmp_path / "collections" / "nx5"
    archived = list((root / "archive").iterdir())
    assert len(archived) == 1 and b"111.0" in (archived[0] / "capture.jsonl").read_bytes()
    assert b"222.0" in (root / "current" / "capture.jsonl").read_bytes()


def test_a_refused_connection_leaves_the_existing_collection_untouched(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=111.0))
    collector.stop()
    factory.fail = True

    with pytest.raises(StudioError, match="broker:1883"):
        collector.start("broker", 1883, ["t"])

    state = collector.state()
    assert state["active"] is False and state["has_data"] is True and state["messages"] == 1
    assert not (tmp_path / "collections" / "nx5" / "archive").exists()


def test_start_twice_is_refused(tmp_path):
    collector, _, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])

    with pytest.raises(StudioError, match="이미 수집"):
        collector.start("broker", 1883, ["t"])


def test_discard_removes_the_data_but_not_while_collecting(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=1.0))
    with pytest.raises(StudioError, match="수집 중"):
        collector.discard()
    collector.stop()

    collector.discard()

    assert collector.state()["has_data"] is False and not (tmp_path / "collections" / "nx5" / "current").exists()


def test_import_folder_copies_the_data_and_archives_the_previous_collection(tmp_path):
    source_dir = tmp_path / "saved data"
    source_dir.mkdir()
    lines = [_cycle_payload(i) for i in range(1200)]
    (source_dir / "DX_AxisData.json").write_bytes(b"\n".join(lines) + b"\n")
    (source_dir / "notes.txt").write_text("not data", encoding="utf-8")
    collector, factory, _ = _collector(tmp_path / "ws")
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=1.0))
    collector.stop()

    collector.import_folder(f'  "{source_dir}"  ')

    current = tmp_path / "ws" / "collections" / "nx5" / "current"
    state = collector.state()
    assert (current / "DX_AxisData.json").exists() and not (current / "notes.txt").exists()
    assert state["has_data"] is True and state["ended_reason"] == "import" and state["cycles"]["cycles"] == 12
    assert len(list((current.parent / "archive").iterdir())) == 1


def test_import_folder_rejects_a_missing_or_empty_folder(tmp_path):
    collector, _, _ = _collector(tmp_path / "ws")
    (tmp_path / "empty").mkdir()

    with pytest.raises(StudioError, match="찾을 수 없습니다"):
        collector.import_folder(str(tmp_path / "nope"))
    with pytest.raises(StudioError, match=r"\.json"):
        collector.import_folder(str(tmp_path / "empty"))


def test_a_capture_that_was_never_finished_is_reported_as_interrupted(tmp_path):
    collector, factory, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    factory.last.emit(payload(0, pos=1.0))
    # 앱이 꺼진 것과 같다: stop()을 부르지 않고 새 Collector가 같은 작업 폴더를 연다
    reopened, _, _ = _collector(tmp_path)

    state = reopened.state()

    assert state["active"] is False and state["has_data"] is True and state["interrupted"] is True


def test_no_data_flag_after_thirty_silent_seconds_while_collecting(tmp_path):
    collector, factory, now = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])
    assert collector.state()["no_data"] is False

    now[0] = 31.0

    assert collector.state()["no_data"] is True
    factory.last.emit(payload(0, pos=1.0))
    assert collector.state()["no_data"] is False


def test_the_equipment_cannot_change_while_collecting(tmp_path):
    collector, _, _ = _collector(tmp_path)
    collector.start("broker", 1883, ["t"])

    with pytest.raises(StudioError, match="수집 중"):
        collector.set_equipment("other")


def test_state_without_an_equipment_is_empty(tmp_path):
    collector = Collector(tmp_path, SourceFactory())

    state = collector.state()

    assert state["active"] is False and state["has_data"] is False and state["cycles"] is None
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_studio_collector.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: 구현**

<!-- file: src/jetson_app/studio/collector.py -->
```python
"""3단계 전체 수집: 브로커에서 받은 원본 payload를 한 줄씩 파일에 그대로 모은다."""
from __future__ import annotations

import json
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from .cycles import CycleCounter, scan_directory
from .errors import StudioError

DEFAULT_MAX_BYTES = 1 << 30
NO_DATA_SECONDS = 30.0
REPORT_INTERVAL_SECONDS = 5.0
CAPTURE_FILE = "capture.jsonl"
META_FILE = "capture.meta"          # .json이 아니라서 데이터 폴더를 읽는 도구가 건너뛴다
DATA_SUFFIXES = (".json", ".jsonl")
NO_EQUIPMENT = "먼저 1단계에서 연결 정보를 저장하세요."


def _empty_state(limit_bytes: int) -> dict:
    return {
        "active": False, "has_data": False, "connected": False, "messages": 0, "bytes": 0,
        "limit_bytes": limit_bytes, "seconds": 0.0, "no_data": False, "interrupted": False,
        "ended_reason": None, "origin": None, "cycles": None,
    }


class Collector:
    def __init__(
        self, workspace, source_factory, clock=time.monotonic, max_bytes=DEFAULT_MAX_BYTES,
        report_interval=REPORT_INTERVAL_SECONDS,
    ) -> None:
        self._root = Path(workspace) / "collections"
        self._source_factory = source_factory
        self._clock = clock
        self._max_bytes = max_bytes
        self._report_interval = report_interval
        self._lock = threading.Lock()
        self._equipment_id = None
        self._active = False
        self._source = None
        self._file = None
        self._messages = 0
        self._bytes = 0
        self._started = 0.0
        self._last_rx = None
        self._ended_reason = None
        self._counter = None
        self._report = None
        self._report_at = 0.0

    # --- 준비 ---------------------------------------------------------------------------

    def set_equipment(self, equipment_id) -> None:
        with self._lock:
            if self._active:
                raise StudioError("수집 중에는 장비를 바꿀 수 없습니다. 먼저 수집을 종료하세요.")
            self._equipment_id = equipment_id
            self._forget_locked()

    @property
    def current_dir(self) -> Path:
        if self._equipment_id is None:
            raise StudioError(NO_EQUIPMENT)
        return self._root / self._equipment_id / "current"

    def _forget_locked(self) -> None:
        self._counter = None
        self._report = None
        self._ended_reason = None

    # --- 수집 ---------------------------------------------------------------------------

    def start(self, host: str, port: int, topics) -> None:
        with self._lock:
            if self._equipment_id is None:
                raise StudioError(NO_EQUIPMENT)
            if self._active:
                raise StudioError("이미 수집 중입니다.")
        source = self._source_factory()
        try:
            source.connect(host, port, list(topics), self._on_payload)
        except OSError as e:
            raise StudioError(
                f"브로커({host}:{port})에 연결할 수 없습니다. 브로커가 실행 중인지, 주소와 포트가 맞는지 확인하세요. ({e})"
            ) from e
        with self._lock:
            self._archive_current_locked()
            current = self.current_dir
            current.mkdir(parents=True, exist_ok=True)
            self._file = open(current / CAPTURE_FILE, "ab")
            self._messages = 0
            self._bytes = 0
            self._started = self._clock()
            self._last_rx = None
            self._ended_reason = None
            self._counter = CycleCounter()
            self._report = None
            self._source = source
            self._active = True
            self._write_meta_locked(
                host=host, port=port, topics=list(topics), started_at=_now_text(), stopped_at=None,
                reason=None, messages=0, bytes=0, seconds=0.0,
            )

    def _on_payload(self, topic: str, payload: bytes) -> None:
        reason = None
        with self._lock:
            if not self._active:
                return
            line = payload.replace(b"\r", b" ").replace(b"\n", b" ").strip()
            if not line:
                return
            self._file.write(line + b"\n")
            self._file.flush()
            self._bytes += len(line) + 1
            self._messages += 1
            self._last_rx = self._clock()
            self._counter.feed_payload(line)
            if self._bytes >= self._max_bytes:
                reason = "size"
        if reason is not None:
            self._finish(reason)

    def stop(self) -> None:
        self._finish("user")

    def _finish(self, reason: str) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._ended_reason = reason
            source, self._source = self._source, None
            try:
                self._file.close()
            finally:
                self._file = None
            self._write_meta_locked(
                stopped_at=_now_text(), reason=reason, messages=self._messages, bytes=self._bytes,
                seconds=round(self._clock() - self._started, 1),
            )
            self._report = None
        if source is not None:
            try:
                source.disconnect()
            except Exception as e:
                print(f"[studio] 수집 연결을 닫는 중 오류(무시): {e!r}")

    def discard(self) -> None:
        with self._lock:
            if self._active:
                raise StudioError("수집 중에는 지울 수 없습니다. 먼저 수집을 종료하세요.")
            shutil.rmtree(self.current_dir, ignore_errors=True)
            self._forget_locked()

    def import_folder(self, folder) -> None:
        """저장해 둔 데이터 폴더의 *.json/*.jsonl을 수집본으로 복사해 온다(원본 폴더는 읽기만 한다)."""
        path = Path(str(folder).strip().strip('"').strip("'").strip())
        if not path.is_dir():
            raise StudioError(f"데이터 폴더를 찾을 수 없습니다: {path}")
        files = sorted(p for p in path.iterdir() if p.suffix in DATA_SUFFIXES)
        if not files:
            raise StudioError(f"데이터 폴더에 .json/.jsonl 파일이 없습니다: {path}")
        with self._lock:
            if self._equipment_id is None:
                raise StudioError(NO_EQUIPMENT)
            if self._active:
                raise StudioError("수집 중에는 불러올 수 없습니다. 먼저 수집을 종료하세요.")
            self._archive_current_locked()
            current = self.current_dir
            current.mkdir(parents=True, exist_ok=True)
            total = 0
            for source in files:
                shutil.copy2(source, current / source.name)
                total += source.stat().st_size
            self._counter = scan_directory(current)
            self._report = None
            self._ended_reason = "import"
            self._write_meta_locked(
                origin=str(path), started_at=_now_text(), stopped_at=_now_text(), reason="import",
                messages=None, bytes=total, seconds=round(self._counter.duration_seconds, 1),
            )

    # --- 조회 ---------------------------------------------------------------------------

    def state(self) -> dict:
        with self._lock:
            if self._equipment_id is None:
                return _empty_state(self._max_bytes)
            now = self._clock()
            files = self._data_files_locked()
            meta = self._read_meta_locked() or {}
            active = self._active
            info = _empty_state(self._max_bytes)
            info["active"] = active
            info["has_data"] = bool(files)
            info["bytes"] = self._bytes if active else sum(_size(p) for p in files)
            info["messages"] = self._messages if active else (meta.get("messages") or 0)
            info["seconds"] = (now - self._started) if active else float(meta.get("seconds") or 0.0)
            info["connected"] = bool(active and self._source is not None and self._source.is_connected())
            if active:
                since = self._last_rx if self._last_rx is not None else self._started
                info["no_data"] = (now - since) > NO_DATA_SECONDS
            info["interrupted"] = bool(files) and not active and "stopped_at" in meta and meta["stopped_at"] is None
            info["ended_reason"] = self._ended_reason if self._ended_reason else meta.get("reason")
            info["origin"] = meta.get("origin")
            if files:
                info["cycles"] = self._report_locked(now).to_dict()
            return info

    def cycle_report(self, preferred_tag=None):
        with self._lock:
            self._ensure_counter_locked()
            return self._counter.report(preferred_tag)

    # --- 내부 ---------------------------------------------------------------------------

    def _ensure_counter_locked(self) -> None:
        if self._counter is None:
            self._counter = scan_directory(self.current_dir) if self.current_dir.exists() else CycleCounter()
            self._report = None

    def _report_locked(self, now: float):
        self._ensure_counter_locked()
        stale = self._report is None or (self._active and now - self._report_at >= self._report_interval)
        if stale:
            self._report = self._counter.report()
            self._report_at = now
        return self._report

    def _data_files_locked(self) -> list:
        current = self.current_dir
        if not current.exists():
            return []
        return sorted(p for p in current.iterdir() if p.suffix in DATA_SUFFIXES)

    def _read_meta_locked(self):
        try:
            data = json.loads((self.current_dir / META_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _write_meta_locked(self, **fields) -> None:
        meta = dict(self._read_meta_locked() or {})
        meta.update(fields)
        (self.current_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def _archive_current_locked(self) -> None:
        current = self.current_dir
        if not current.exists() or not any(current.iterdir()):
            return
        archive = current.parent / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = archive / stamp
        n = 1
        while destination.exists():
            destination = archive / f"{stamp}-{n}"
            n += 1
        shutil.move(str(current), str(destination))


def _now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_studio_collector.py -q`
Expected: 12 passed. 크기 상한 테스트가 실패하면 `_on_payload`가 쓴 바이트를 센 뒤 비교하는지 확인한다.

- [ ] **Step 5: 커밋**

```bash
git add src/jetson_app/studio/collector.py tests/test_studio_collector.py
git commit -m "feat: studio full-data collector (raw capture, size cap, archive, folder import)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 마법사와 수집본 학습 (`wizard.py`, `trainer.py`)

**Files:**
- Replace: `src/jetson_app/studio/wizard.py`, `tests/test_studio_wizard.py`
- Create: `src/jetson_app/studio/trainer.py`
- Test: `tests/test_studio_wizard.py`, `tests/test_studio_trainer.py`

**Interfaces:**
- Consumes: `deviation_suggest.analyze_tags`, `deviation_suggest._signal_name`, `config.load_equipment_config`, `replay.load_records`, `replay.steps_for_segment`, `deviation.make_deviation_train_fn`, `deviation.deviation_artifact_path`, `artifact_paths.state_marker_path`, `calibration.StateStore/CalibrationState/CalibrationSample`, `timeparse.format_epoch_ns`.
- Produces (`wizard`): `analyze(data_dir) -> {"tags": [...]}`, `save_connection(workspace, form) -> dict`, `load_settings(workspace) -> {equipment_id, host, port, topics}|None`, `saved_connection(workspace) -> {equipment_id, host, port, subscribe_topics}|{}`, `save_config(workspace, tags) -> EquipmentConfig`, `load_config(workspace) -> EquipmentConfig|None`, `saved_tags(workspace) -> list[str]`, `NO_CONNECTION`.
- Produces (`trainer`): `train_from_collection(config, collection_dir, model_dir) -> int` (학습에 쓴 스텝 수; 실패는 `StudioError` 또는 `ValueError`).

- [ ] **Step 1: 실패하는 테스트**

<!-- file: tests/test_studio_wizard.py -->
```python
import json
import math

import pytest

from jetson_app.studio import wizard
from jetson_app.studio.errors import StudioError
from jetson_app.timeparse import format_epoch_ns, parse_dx1_timestamp

BASE_NS = parse_dx1_timestamp("2026-10-02T00:00:00Z")


def _timestamp(second):
    return format_epoch_ns(BASE_NS + int(second * 1_000_000_000)).replace("+00:00", "+0000")


def _noise(n, scale):
    return scale * (((n * 7919) % 13) - 6) / 6.0


def _write_data(directory, seconds=300):
    """pos는 가끔 크게 움직이고 평소엔 거의 일정(추천 대상), vel은 늘 흔들림(비추천)."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "DX_0.json").open("w", encoding="utf-8") as f:
        for n in range(seconds * 10):
            moving = (n // 600) % 2 == 1 and (n % 600) < 80
            pos = 100.0 * ((n % 600) / 80.0 if moving else (1.0 if (n // 600) % 2 == 1 else 0.0))
            record = {
                "timestamp": _timestamp(n / 10),
                "m:pos": pos + _noise(n, 1e-5),
                "m:vel": 5.0 * math.sin(n / 3.0) + _noise(n, 0.5),
            }
            f.write(json.dumps({"records": [record]}) + "\n")


def _connection(**overrides):
    form = {
        "equipment_id": "nx5",
        "host": "localhost",
        "port": 1883,
        "subscribe_topics": ["dx1/AxisData"],
    }
    form.update(overrides)
    return form


def test_analyze_recommends_the_position_tag(tmp_path):
    _write_data(tmp_path / "d")

    result = wizard.analyze(str(tmp_path / "d"))

    by_tag = {t["tag"]: t for t in result["tags"]}
    assert by_tag["m:pos"]["recommended"] is True
    assert by_tag["m:vel"]["recommended"] is False and by_tag["m:vel"]["reason"]
    json.dumps(result)                                   # 그대로 JSON 응답으로 보낼 수 있어야 한다


def test_analyze_accepts_a_path_pasted_with_quotes_and_spaces(tmp_path):
    _write_data(tmp_path / "my data")

    result = wizard.analyze(f'  "{tmp_path / "my data"}"  ')

    assert any(t["recommended"] for t in result["tags"])


def test_analyze_reports_a_missing_folder_as_studio_error(tmp_path):
    with pytest.raises(StudioError, match="찾을 수 없습니다"):
        wizard.analyze(str(tmp_path / "nope"))


def test_save_connection_persists_and_is_restored(tmp_path):
    saved = wizard.save_connection(tmp_path, _connection(subscribe_topics=["dx1/AxisData", " dx1/ProcStart "]))

    assert saved["topics"] == ["dx1/AxisData", "dx1/ProcStart"]
    assert wizard.load_settings(tmp_path) == {
        "equipment_id": "nx5", "host": "localhost", "port": 1883, "topics": ["dx1/AxisData", "dx1/ProcStart"],
    }
    assert wizard.saved_connection(tmp_path) == {
        "equipment_id": "nx5", "host": "localhost", "port": 1883,
        "subscribe_topics": ["dx1/AxisData", "dx1/ProcStart"],
    }


def test_nothing_is_saved_before_the_connection_is_entered(tmp_path):
    assert wizard.load_settings(tmp_path) is None
    assert wizard.saved_connection(tmp_path) == {}
    assert wizard.load_config(tmp_path) is None
    assert wizard.saved_tags(tmp_path) == []


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"equipment_id": "../evil"}, "장비 이름"),
        ({"equipment_id": ""}, "장비 이름"),
        ({"host": "localhost:1883"}, "포트"),
        ({"host": ""}, "브로커 주소"),
        ({"port": 0}, "포트"),
        ({"port": "abc"}, "포트"),
        ({"subscribe_topics": []}, "토픽"),
    ],
)
def test_save_connection_rejects_bad_input_without_writing(tmp_path, overrides, message):
    with pytest.raises(StudioError, match=message):
        wizard.save_connection(tmp_path, _connection(**overrides))

    assert wizard.load_settings(tmp_path) is None


def test_a_hand_broken_settings_file_is_ignored(tmp_path):
    (tmp_path / "studio.json").write_text('{"equipment_id": "../x", "host": "h", "port": 1, "topics": ["t"]}')

    assert wizard.load_settings(tmp_path) is None


def test_save_config_needs_the_connection_first(tmp_path):
    with pytest.raises(StudioError, match="연결 정보"):
        wizard.save_config(tmp_path, ["m:pos"])


def test_save_config_writes_a_valid_deviation_config(tmp_path):
    wizard.save_connection(tmp_path, _connection(subscribe_topics=["dx1/AxisData", "dx1/ProcStart"]))

    config = wizard.save_config(tmp_path, ["m:pos", "m:pos2", "m:pos"])

    assert config.equipment_id == "nx5" and config.tags == ()
    assert [s.tag for s in config.deviation.signals] == ["m:pos", "m:pos2"]
    assert list(config.subscribe_topics) == ["dx1/AxisData", "dx1/ProcStart"]
    assert (tmp_path / "configs" / "nx5.yaml").exists()
    assert wizard.saved_tags(tmp_path) == ["m:pos", "m:pos2"]
    assert wizard.load_config(tmp_path).equipment_id == "nx5"


def test_save_config_rejects_no_tags_without_writing(tmp_path):
    wizard.save_connection(tmp_path, _connection())

    with pytest.raises(StudioError, match="태그"):
        wizard.save_config(tmp_path, [])

    assert not (tmp_path / "configs").exists() or not list((tmp_path / "configs").iterdir())
```

<!-- file: tests/test_studio_trainer.py -->
```python
import pytest

from jetson_app.calibration import CalibrationState, StateStore
from jetson_app.deviation import deviation_artifact_path
from jetson_app.artifact_paths import state_marker_path
from jetson_app.studio import wizard
from jetson_app.studio.errors import StudioError
from jetson_app.studio.trainer import train_from_collection
from studio_fakes import noise, payload


def _config(tmp_path, tags=("pos",)):
    wizard.save_connection(
        tmp_path / "ws",
        {"equipment_id": "nx5", "host": "localhost", "port": 1883, "subscribe_topics": ["dx1/data"]},
    )
    return wizard.save_config(tmp_path / "ws", list(tags))


def _write(directory, build):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "capture.jsonl").write_bytes(b"\n".join(build(i) for i in range(400)) + b"\n")


def test_training_from_a_collection_writes_the_model_and_marks_monitoring(tmp_path):
    config = _config(tmp_path)
    collection = tmp_path / "collection"
    _write(collection, lambda i: payload(i, pos=noise(i), other=5.0))

    steps = train_from_collection(config, collection, tmp_path / "models")

    assert steps > 300
    assert deviation_artifact_path(tmp_path / "models", "nx5").exists()
    assert StateStore(state_marker_path(tmp_path / "models", "nx5")).read() == CalibrationState.MONITORING


def test_training_on_data_that_never_changes_explains_why(tmp_path):
    config = _config(tmp_path)
    collection = tmp_path / "collection"
    _write(collection, lambda i: payload(i, pos=1.0))

    with pytest.raises(ValueError, match="변하지"):
        train_from_collection(config, collection, tmp_path / "models")

    assert not deviation_artifact_path(tmp_path / "models", "nx5").exists()


def test_training_when_the_chosen_tag_is_not_in_the_collection(tmp_path):
    config = _config(tmp_path, tags=("missing",))
    collection = tmp_path / "collection"
    _write(collection, lambda i: payload(i, pos=noise(i)))

    with pytest.raises(StudioError, match="missing"):
        train_from_collection(config, collection, tmp_path / "models")
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_studio_wizard.py tests/test_studio_trainer.py -q`
Expected: FAIL (wizard 함수 없음, trainer 모듈 없음)

- [ ] **Step 3: 구현**

<!-- file: src/jetson_app/studio/wizard.py -->
```python
"""설정 마법사: 연결 정보를 저장하고, 수집한 데이터의 태그를 추천하고, 고른 태그로 설정 YAML을 만들어 검증한다."""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from ..config import ConfigError, EquipmentConfig, load_equipment_config
from ..deviation_suggest import _signal_name, analyze_tags  # 신호 이름 규칙을 CLI 추천과 똑같이 맞춘다
from ..replay import ReplayError
from .errors import StudioError

SETTINGS_FILE = "studio.json"
DEFAULT_MIN_SAMPLES = 3000      # 설정 파일 형식상 필요한 값. 이 화면은 수집본으로 학습하므로 쓰이지 않는다
NO_CONNECTION = "먼저 1단계에서 연결 정보를 저장하세요."
_EQUIPMENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_HOST = re.compile(r"^[A-Za-z0-9._-]+$")


def analyze(data_dir: str) -> dict:
    """데이터 폴더의 숫자 태그를 분석해 화면에 보여 줄 표를 만든다. 폴더는 읽기만 한다."""
    path = Path(str(data_dir).strip().strip('"').strip("'").strip())
    try:
        reports = analyze_tags(path)
    except (ReplayError, OSError, ValueError) as e:
        raise StudioError(str(e)) from e
    return {
        "tags": [
            {
                "tag": r.tag,
                "kind": r.kind,
                "value_range": r.value_range,
                "sigma": r.sigma,
                "limit": r.limit,
                "dynamic_range": r.dynamic_range,
                "moving_fraction": r.moving_fraction,
                "recommended": r.recommended,
                "reason": r.reason,
            }
            for r in reports
        ]
    }


# --- 1단계: 연결 정보 -------------------------------------------------------------------------


def _clean_connection(form: dict) -> dict:
    equipment_id = str(form.get("equipment_id", "")).strip()
    if not _EQUIPMENT_ID.match(equipment_id):
        raise StudioError("장비 이름은 영문, 숫자, _, - 만 쓸 수 있습니다(1~40자).")
    host = str(form.get("host", "")).strip()
    if not host:
        raise StudioError("브로커 주소를 입력하세요(예: localhost).")
    if not _HOST.match(host):
        raise StudioError("브로커 주소에는 포트나 공백을 넣을 수 없습니다. 포트는 아래 칸에 따로 적으세요.")
    try:
        port = int(form.get("port"))
    except (TypeError, ValueError):
        raise StudioError("포트는 1~65535 사이의 숫자여야 합니다.") from None
    if not 1 <= port <= 65535:
        raise StudioError("포트는 1~65535 사이의 숫자여야 합니다.")
    topics = [str(t).strip() for t in form.get("subscribe_topics") or [] if str(t).strip()]
    if not topics:
        raise StudioError("구독할 토픽을 하나 이상 입력하세요(예: dx1/AxisData).")
    return {"equipment_id": equipment_id, "host": host, "port": port, "topics": topics}


def save_connection(workspace, form: dict) -> dict:
    clean = _clean_connection(form)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / SETTINGS_FILE).write_text(json.dumps(clean, ensure_ascii=False), encoding="utf-8")
    return clean


def load_settings(workspace):
    path = Path(workspace) / SETTINGS_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return _clean_connection(
            {
                "equipment_id": data.get("equipment_id"),
                "host": data.get("host"),
                "port": data.get("port"),
                "subscribe_topics": data.get("topics"),
            }
        )
    except StudioError:
        return None


def saved_connection(workspace) -> dict:
    """저장된 연결 정보를 화면 입력칸 값으로 돌려준다. 아직 없으면 빈 딕셔너리."""
    settings = load_settings(workspace)
    if settings is None:
        return {}
    return {
        "equipment_id": settings["equipment_id"],
        "host": settings["host"],
        "port": settings["port"],
        "subscribe_topics": list(settings["topics"]),
    }


# --- 5단계: 고른 태그로 설정 만들기 -----------------------------------------------------------------


def _config_path(workspace, equipment_id: str) -> Path:
    return Path(workspace) / "configs" / f"{equipment_id}.yaml"


def _config_dict(settings: dict, tags: list) -> dict:
    used = set()
    signals = {_signal_name(tag, used): tag for tag in tags}
    eid = settings["equipment_id"]
    return {
        "equipment_id": eid,
        "mqtt": {
            "subscribe_topics": settings["topics"],
            "publish_topic": f"jetson/{eid}/anomaly",
            "command_topic": f"jetson/{eid}/cmd",
        },
        "deviation": {"signals": signals},
        "resample_interval_ms": 100,
        "max_lateness_ms": 500,
        "window_size": 30,
        "calibration": {"max_duration": "7d", "min_samples": DEFAULT_MIN_SAMPLES},
    }


def save_config(workspace, tags) -> EquipmentConfig:
    """저장된 연결 정보와 고른 태그로 설정 YAML을 만들어 기존 로더로 검증한 뒤에만 저장한다."""
    settings = load_settings(workspace)
    if settings is None:
        raise StudioError(NO_CONNECTION)
    chosen = []
    for tag in tags or []:
        tag = str(tag).strip()
        if tag and tag not in chosen:
            chosen.append(tag)
    if not chosen:
        raise StudioError("점수를 낼 태그를 하나 이상 선택하세요.")
    final = _config_path(workspace, settings["equipment_id"])
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.tmp")
    tmp.write_text(
        yaml.safe_dump(_config_dict(settings, chosen), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    try:
        config = load_equipment_config(tmp)
    except ConfigError as e:
        tmp.unlink()
        raise StudioError(f"설정이 올바르지 않습니다: {e}") from e
    tmp.replace(final)
    return config


def load_config(workspace):
    """연결 정보에 맞는 설정 YAML이 있으면 읽고, 없으면 None."""
    settings = load_settings(workspace)
    if settings is None:
        return None
    path = _config_path(workspace, settings["equipment_id"])
    if not path.exists():
        return None
    try:
        return load_equipment_config(path)
    except ConfigError as e:
        raise StudioError(f"저장된 설정을 읽지 못했습니다: {e}") from e


def saved_tags(workspace) -> list:
    config = load_config(workspace)
    if config is None or config.deviation is None:
        return []
    return [s.tag for s in config.deviation.signals]
```

<!-- file: src/jetson_app/studio/trainer.py -->
```python
"""5단계 학습: 수집본(원본)에서 고른 태그의 값을 읽어 기준값 이탈 기준을 만든다. 파이프라인도 torch도 쓰지 않는다."""
from __future__ import annotations

from pathlib import Path

from ..artifact_paths import state_marker_path
from ..calibration import CalibrationSample, CalibrationState, StateStore
from ..deviation import deviation_artifact_path, make_deviation_train_fn
from ..replay import load_records, steps_for_segment
from ..segments import Segment
from ..timeparse import format_epoch_ns
from .errors import StudioError

_EVERYTHING = Segment(kind="train", name="all", start_ns=0, end_ns=2**62)


def train_from_collection(config, collection_dir, model_dir) -> int:
    """수집본으로 학습해 모델과 상태 마커(MONITORING)를 쓰고, 학습에 쓴 스텝 수를 돌려준다.
    학습이 실패하면 예외를 그대로 올리고 아무것도 쓰지 않는다."""
    records = load_records(Path(collection_dir), config.all_tags, (_EVERYTHING,))
    samples = []
    for step in steps_for_segment(config, records, _EVERYTHING):
        if step.reset_window and samples:
            samples.append(CalibrationSample(timestamp="", values={}))   # 공백 표시: 끊긴 구간을 이어 붙이지 않는다
        samples.append(CalibrationSample(timestamp=format_epoch_ns(step.epoch_ns), values=step.snapshot.values))
    if not samples:
        raise StudioError(
            "선택한 태그의 값이 수집 데이터에 없습니다: " + ", ".join(config.all_tags) + ". 태그 이름을 확인하세요."
        )
    artifact_path = deviation_artifact_path(model_dir, config.equipment_id)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    make_deviation_train_fn(config, artifact_path)(samples)
    StateStore(state_marker_path(model_dir, config.equipment_id)).write(CalibrationState.MONITORING)
    return len([s for s in samples if s.values])
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_studio_wizard.py tests/test_studio_trainer.py -q`
Expected: 모두 통과. `missing` 태그 테스트가 `fit_deviation`의 메시지로 먼저 실패하면 `samples`가 비는 경로(빈 values)를 확인한다: 값이 없는 스텝은 `step.snapshot.values`가 비어 있어도 샘플이 만들어질 수 있으니, 그 경우 `"선택한 태그"` 안내가 나오도록 `samples`에서 `values`가 있는 것이 하나도 없을 때도 같은 오류를 내게 고친다.

- [ ] **Step 5: 커밋**

```bash
git add src/jetson_app/studio/wizard.py src/jetson_app/studio/trainer.py tests/test_studio_wizard.py tests/test_studio_trainer.py
git commit -m "feat: studio wizard for connection+tags and training from a collection" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

(이 시점에는 `controller.py`가 옛 `wizard` 함수를 부르므로 전체 테스트는 Task 5가 끝나야 다시 통과한다. Task 4는 위 두 파일의 테스트만 확인한다.)

---

### Task 5: 컨트롤러 (6단계 상태기계)

**Files:**
- Replace: `src/jetson_app/studio/controller.py`, `tests/test_studio_controller.py`
- Modify: `src/jetson_app/studio/cli.py` (종료 시 `shutdown()`)
- Test: `tests/test_studio_controller.py`

**Interfaces:**
- Consumes: Task 1~4의 모든 부품, `pipeline.build_pipeline`, `score_history.DEFAULT_HISTORY_SECONDS`.
- Produces: `Stage` (`CONNECT, PROBE, COLLECT, TAGS, TRAIN, MONITOR`), `STEP` (단계→1..6), `StudioController(workspace, io=None, source_factory=RawSource, pipeline_factory=build_pipeline, clock=time.monotonic, run_worker=True, max_collect_bytes=DEFAULT_MAX_BYTES)`.
- 메서드: `connection() -> dict`, `save_connection(form)`, `probe_start()`, `probe_stop()`, `collect_start()`, `collect_stop()`, `discard()`, `import_folder(data_dir)`, `analyze(cycle_tag=None, with_tags=True) -> {"tags"?, "cycles"}`, `train(tags)`, `join_training(timeout=None)`, `start()`, `stop()`, `retrain()`, `shutdown()`, `state() -> dict`, `meta()`, `scores(seconds, max_points)`, `feed_record(record)`, `pump()`, `pipeline`.
- `state()` 키: `stage, step, equipment_id, error, probe, collect, training, model_ready, running, connected, signals, threshold, can`. `can` 키: `save_connection, probe_start, probe_stop, collect_start, collect_stop, discard, import, analyze, train, start, stop, retrain`.

- [ ] **Step 1: 실패하는 테스트**

<!-- file: tests/test_studio_controller.py -->
```python
import json
from types import SimpleNamespace

import pytest

from jetson_app.studio.controller import Stage, StudioController
from jetson_app.studio.errors import StudioError
from studio_fakes import FakeIO, SourceFactory, noise, payload

CONNECTION = {"equipment_id": "nx5", "host": "localhost", "port": 1883, "subscribe_topics": ["dx1/data"]}


def _controller(tmp_path, fail_source=False, io=None, **kwargs):
    factory = SourceFactory(fail=fail_source)
    io = io or FakeIO()
    controller = StudioController(tmp_path, io=io, source_factory=factory, run_worker=False, **kwargs)
    return controller, factory, io


def _collect(controller, factory, steps=400, value=None, stop=True):
    """수집을 시작하고 pos(와 사이클 신호 cyc)를 흘려 넣는다."""
    controller.collect_start()
    for i in range(steps):
        pos = value if value is not None else noise(i)
        factory.last.emit(payload(i, pos=pos, cyc=1 if i % 100 >= 90 else 0))
    if stop:
        controller.collect_stop()


def _train(controller, tags=("pos",)):
    controller.train(list(tags))
    controller.join_training(timeout=30)


def _feed_pipeline(controller, start, stop, spike_at=None):
    for i in range(start, stop):
        value = noise(i) + (0.01 if i == spike_at else 0.0)
        message = SimpleNamespace(payload=payload(i, pos=value))
        controller.pipeline.mqtt_subscriber._handle_message(None, None, message)
    controller.pump()


def test_a_fresh_workspace_starts_at_the_connection_step(tmp_path):
    controller, _, _ = _controller(tmp_path)

    state = controller.state()

    assert state["stage"] == Stage.CONNECT.value and state["step"] == 1
    assert controller.connection() == {}
    assert state["can"]["save_connection"] is True and state["can"]["probe_start"] is False


def test_saving_the_connection_moves_to_probe_and_survives_a_restart(tmp_path):
    controller, _, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)

    assert controller.state()["stage"] == "PROBE" and controller.state()["step"] == 2
    again, _, _ = _controller(tmp_path)
    assert again.state()["stage"] == "PROBE" and again.connection()["host"] == "localhost"


def test_bad_connection_input_is_a_studio_error(tmp_path):
    controller, _, _ = _controller(tmp_path)

    with pytest.raises(StudioError, match="포트"):
        controller.save_connection(dict(CONNECTION, host="localhost:1883"))


def test_probe_shows_incoming_data_and_collecting_stops_it(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)

    controller.probe_start()
    factory.last.emit(payload(0, pos=1.0, cyc=0))
    probe = controller.state()["probe"]
    assert probe["active"] is True and probe["messages"] == 1 and probe["tags"] == ["cyc", "pos"]

    controller.collect_start()                              # 시험은 저절로 끝난다
    assert controller.state()["probe"]["active"] is False and factory.created[0].connected is False
    assert controller.state()["stage"] == "COLLECT"


def test_probe_without_a_saved_connection_is_refused(tmp_path):
    controller, _, _ = _controller(tmp_path)

    with pytest.raises(StudioError, match="연결 정보"):
        controller.probe_start()


def test_a_broker_that_refuses_the_connection_keeps_the_stage_and_the_old_data(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory, steps=50)
    factory.fail = True

    with pytest.raises(StudioError, match="localhost:1883"):
        controller.collect_start()
    with pytest.raises(StudioError, match="localhost:1883"):
        controller.probe_start()

    state = controller.state()
    assert state["stage"] == "TAGS" and state["collect"]["has_data"] is True


def test_collect_then_analyze_gives_tags_and_the_cycle_count(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory)

    state = controller.state()
    assert state["stage"] == "TAGS" and state["step"] == 4 and state["collect"]["messages"] == 400

    result = controller.analyze()
    assert {t["tag"] for t in result["tags"]} == {"pos", "cyc"}
    assert result["cycles"]["method"] == "edge" and result["cycles"]["cycles"] == 3
    assert "적습니다" in result["cycles"]["warning"]              # 10사이클 미만은 경고만 한다
    assert "tags" not in controller.analyze(with_tags=False)


def test_analyze_without_data_is_refused(tmp_path):
    controller, _, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)

    with pytest.raises(StudioError, match="수집"):
        controller.analyze()


def test_train_needs_tags_and_data(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    with pytest.raises(StudioError, match="수집"):
        controller.train(["pos"])
    _collect(controller, factory)

    with pytest.raises(StudioError, match="태그"):
        controller.train([])


def test_collect_train_monitor_and_alarm_on_a_shock(tmp_path):
    controller, factory, io = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory)

    _train(controller)

    state = controller.state()
    assert state["stage"] == "MONITOR" and state["step"] == 6 and state["error"] is None
    assert state["model_ready"] is True and state["running"] is False
    assert [s["name"] for s in state["signals"]] == ["pos"] and state["signals"][0]["sigma"] > 0

    controller.start()
    assert io.calls == [("localhost", 1883)] and controller.state()["running"] is True
    _feed_pipeline(controller, 400, 700, spike_at=500)

    points = controller.scores(seconds=3600, max_points=5000)["points"]
    alarms = [p for p in points if p["g"]["pos"][1]]
    assert len(alarms) == 1 and alarms[0]["g"]["pos"][2] == "pos"
    assert controller.meta()["groups"] == ["pos"] and controller.meta()["state"] == "MONITORING"


def test_stop_keeps_the_model_and_a_restart_can_monitor_again(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory)
    _train(controller)
    controller.start()

    controller.stop()
    assert controller.state()["stage"] == "MONITOR" and controller.state()["running"] is False

    again, _, _ = _controller(tmp_path)                       # 앱을 껐다 켠 것과 같다
    assert again.state()["stage"] == "MONITOR" and again.state()["model_ready"] is True
    again.start()
    assert again.state()["running"] is True


def test_training_on_data_that_never_changes_reports_why_and_keeps_the_data(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory, value=1.0)

    _train(controller)

    state = controller.state()
    assert state["stage"] == "TAGS" and "변하지" in state["error"]
    assert state["collect"]["has_data"] is True and state["model_ready"] is False
    assert state["can"]["train"] is True                        # 태그를 바꿔 다시 시도할 수 있다


def test_retrain_goes_back_to_the_tag_step_with_the_same_data(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory)
    _train(controller)
    controller.start()

    controller.retrain()

    state = controller.state()
    assert state["stage"] == "TAGS" and state["running"] is False and state["model_ready"] is False
    assert state["collect"]["has_data"] is True


def test_a_new_collection_makes_the_old_model_unusable(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory)
    _train(controller)
    assert controller.state()["stage"] == "MONITOR"

    _collect(controller, factory, steps=50)

    state = controller.state()
    assert state["stage"] == "TAGS" and state["model_ready"] is False
    with pytest.raises(StudioError, match="모델"):
        controller.start()


def test_busy_states_refuse_conflicting_requests(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory)
    _train(controller)
    controller.start()

    with pytest.raises(StudioError, match="이미"):
        controller.start()
    with pytest.raises(StudioError, match="중지"):
        controller.save_connection(CONNECTION)
    with pytest.raises(StudioError, match="중지"):
        controller.collect_start()
    controller._training = True                                # 학습 스레드가 도는 중인 상태를 흉내 낸다
    with pytest.raises(StudioError, match="이미 학습"):
        controller.train(["pos"])
    controller._training = False


def test_saving_the_connection_while_collecting_is_refused(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory, steps=10, stop=False)

    with pytest.raises(StudioError, match="중지|종료"):
        controller.save_connection(CONNECTION)
    assert controller.state()["can"]["collect_stop"] is True and controller.state()["can"]["collect_start"] is False


def test_import_folder_replaces_the_collection(tmp_path):
    saved = tmp_path / "saved"
    saved.mkdir()
    (saved / "DX_0.json").write_bytes(b"\n".join(payload(i, pos=noise(i)) for i in range(400)) + b"\n")
    controller, _, _ = _controller(tmp_path / "ws")
    controller.save_connection(CONNECTION)

    controller.import_folder(str(saved))

    assert controller.state()["stage"] == "TAGS" and controller.state()["collect"]["origin"] == str(saved)


def test_discard_returns_to_probe(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    _collect(controller, factory, steps=20)

    controller.discard()

    assert controller.state()["stage"] == "PROBE" and controller.state()["collect"]["has_data"] is False


def test_shutdown_closes_everything_and_tolerates_being_called_twice(tmp_path):
    controller, factory, _ = _controller(tmp_path)
    controller.save_connection(CONNECTION)
    controller.collect_start()
    factory.last.emit(payload(0, pos=1.0))

    controller.shutdown()
    controller.shutdown()

    assert factory.last.connected is False and controller.state()["collect"]["active"] is False
    json.dumps(controller.state())                              # 상태는 항상 JSON으로 보낼 수 있어야 한다
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_studio_controller.py -q`
Expected: FAIL (`StudioController`에 새 메서드 없음)

- [ ] **Step 3: 구현**

<!-- file: src/jetson_app/studio/controller.py -->
```python
"""화면이 쓰는 앱 제어 계층: 6단계(연결 → 살펴보기 → 전체 수집 → 태그 추천 → 학습 → 감시)를 관리한다.

새로 계산하는 로직은 없다. 사이클 세기·수집·시험은 각 모듈, 학습은 trainer, 감시 파이프라인은
build_pipeline, 태그 추천은 deviation_suggest를 그대로 부른다.
"""
from __future__ import annotations

import threading
import time
from enum import Enum
from pathlib import Path

from ..artifact_paths import state_marker_path
from ..calibration import CalibrationState, StateStore
from ..deviation import deviation_artifact_path, load_deviation_artifact, make_deviation_train_fn
from ..pipeline import build_pipeline
from ..score_history import DEFAULT_HISTORY_SECONDS
from . import wizard
from .collector import DEFAULT_MAX_BYTES, Collector
from .errors import StudioError
from .probe import ProbeSession
from .raw_source import RawSource
from .trainer import train_from_collection

_EMPTY_SCORES = {"now_ms": None, "bucket_ms": 0, "points": []}


class Stage(str, Enum):
    CONNECT = "CONNECT"    # 1: 연결 정보가 없다
    PROBE = "PROBE"        # 2: 연결 정보는 있고 수집한 데이터는 없다(살펴보기)
    COLLECT = "COLLECT"    # 3: 전체 수집 중
    TAGS = "TAGS"          # 4: 수집본이 있고 학습한 모델은 없다(태그 추천)
    TRAIN = "TRAIN"        # 5: 학습 중
    MONITOR = "MONITOR"    # 6: 학습한 모델이 있다(감시 중이거나 감시를 시작할 수 있다)


STEP = {Stage.CONNECT: 1, Stage.PROBE: 2, Stage.COLLECT: 3, Stage.TAGS: 4, Stage.TRAIN: 5, Stage.MONITOR: 6}


class MqttIO:
    """감시 파이프라인의 MQTT 클라이언트를 백그라운드 스레드로 돌리고 멈춘다(테스트에서는 가짜로 바꾼다)."""

    def connect(self, pipeline, host: str, port: int) -> None:
        pipeline.mqtt_subscriber.connect(host, port)
        pipeline.mqtt_subscriber.client.loop_start()

    def disconnect(self, pipeline) -> None:
        client = pipeline.mqtt_subscriber.client
        client.loop_stop()
        client.disconnect()

    def is_connected(self, pipeline) -> bool:
        return bool(pipeline.mqtt_subscriber.client.is_connected())


class StudioController:
    def __init__(
        self,
        workspace,
        io=None,
        source_factory=RawSource,
        pipeline_factory=build_pipeline,
        clock=time.monotonic,
        run_worker: bool = True,
        max_collect_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._workspace = Path(workspace)
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._calibration_dir = self._workspace / "calibration"
        self._model_dir = self._workspace / "models"
        self._io = io or MqttIO()
        self._source_factory = source_factory
        self._factory = pipeline_factory
        self._clock = clock
        self._run_worker = run_worker
        self._lock = threading.RLock()
        self._collector = Collector(self._workspace, source_factory, clock=clock, max_bytes=max_collect_bytes)
        self._probe = None
        self._pipeline = None
        self._settings = None
        self._config = None
        self._training = False
        self._train_thread = None
        self._error = None
        self._load_saved()

    # --- 저장된 상태 ----------------------------------------------------------------------

    def _load_saved(self) -> None:
        self._settings = wizard.load_settings(self._workspace)
        if self._settings is None:
            return
        self._collector.set_equipment(self._settings["equipment_id"])
        try:
            self._config = wizard.load_config(self._workspace)
        except StudioError as e:
            self._error = str(e)

    def _need_settings(self) -> dict:
        if self._settings is None:
            raise StudioError(wizard.NO_CONNECTION)
        return self._settings

    def _require_idle(self, action: str) -> None:
        """감시·학습·수집 중에는 새 작업을 시작할 수 없다."""
        if self._pipeline is not None:
            raise StudioError(f"감시 중에는 {action}할 수 없습니다. 먼저 감시를 중지하세요.")
        if self._training:
            raise StudioError(f"학습이 끝난 뒤에 {action}할 수 있습니다.")
        if self._collector.state()["active"]:
            raise StudioError(f"수집 중에는 {action}할 수 없습니다. 먼저 수집을 종료하세요.")

    def _model_ready_locked(self) -> bool:
        if self._settings is None:
            return False
        eid = self._settings["equipment_id"]
        if not deviation_artifact_path(self._model_dir, eid).exists():
            return False
        return StateStore(state_marker_path(self._model_dir, eid)).read() == CalibrationState.MONITORING

    def _reset_model_locked(self) -> None:
        """학습 결과를 감시에 쓰지 않도록 표시한다(파일은 지우지 않고 다음 학습이 덮어쓴다)."""
        if self._settings is not None:
            StateStore(state_marker_path(self._model_dir, self._settings["equipment_id"])).write(
                CalibrationState.CALIBRATING
            )

    # --- 1단계: 연결 ---------------------------------------------------------------------

    def connection(self) -> dict:
        return wizard.saved_connection(self._workspace)

    def save_connection(self, form: dict) -> None:
        with self._lock:
            if self._pipeline is not None or self._training or self._collector.state()["active"]:
                raise StudioError("진행 중인 작업이 있습니다. 먼저 감시를 중지하거나 수집을 종료하세요.")
            if self._probe is not None and self._probe.snapshot()["active"]:
                raise StudioError("살펴보기가 진행 중입니다. 먼저 시험을 종료하세요.")
            wizard.save_connection(self._workspace, form)
            self._settings = wizard.load_settings(self._workspace)
            self._collector.set_equipment(self._settings["equipment_id"])
            self._config = None
            self._error = None
            try:
                self._config = wizard.load_config(self._workspace)
            except StudioError as e:
                self._error = str(e)

    # --- 2단계: 살펴보기 -----------------------------------------------------------------

    def probe_start(self) -> None:
        with self._lock:
            settings = self._need_settings()
            self._require_idle("시험을 시작")
            if self._probe is not None and self._probe.snapshot()["active"]:
                return
            probe = ProbeSession(
                self._source_factory(), settings["host"], settings["port"], settings["topics"], clock=self._clock
            )
            try:
                probe.start()
            except OSError as e:
                raise StudioError(
                    f"브로커({settings['host']}:{settings['port']})에 연결할 수 없습니다. "
                    f"브로커가 실행 중인지 확인하세요. ({e})"
                ) from e
            self._probe = probe
            self._error = None

    def probe_stop(self) -> None:
        with self._lock:
            self._stop_probe_locked()

    def _stop_probe_locked(self) -> None:
        if self._probe is not None:
            self._probe.stop()

    # --- 3단계: 전체 수집 ----------------------------------------------------------------

    def collect_start(self) -> None:
        with self._lock:
            settings = self._need_settings()
            self._require_idle("수집을 시작")
            self._stop_probe_locked()
            self._collector.start(settings["host"], settings["port"], settings["topics"])
            self._reset_model_locked()                      # 새 데이터로 다시 학습해야 한다
            self._error = None

    def collect_stop(self) -> None:
        with self._lock:
            self._collector.stop()

    def discard(self) -> None:
        with self._lock:
            self._need_settings()
            self._require_idle("지우기")
            self._collector.discard()

    def import_folder(self, data_dir: str) -> None:
        with self._lock:
            self._need_settings()
            self._require_idle("불러오기")
            self._stop_probe_locked()
            self._collector.import_folder(data_dir)
            self._reset_model_locked()
            self._error = None

    # --- 4단계: 태그 추천 ----------------------------------------------------------------

    def analyze(self, cycle_tag=None, with_tags: bool = True) -> dict:
        """수집본으로 태그를 추천하고 사이클 수를 센다. 오래 걸릴 수 있어 컨트롤러 락을 잡지 않는다."""
        with self._lock:
            self._need_settings()
            if self._training or self._collector.state()["active"]:
                raise StudioError("수집이나 학습이 끝난 뒤에 분석할 수 있습니다.")
            if not self._collector.state()["has_data"]:
                raise StudioError("수집한 데이터가 없습니다. 먼저 3단계에서 수집하세요.")
            directory = self._collector.current_dir
        result = {}
        if with_tags:
            result.update(wizard.analyze(str(directory)))
        result["cycles"] = self._collector.cycle_report(cycle_tag).to_dict()
        return result

    # --- 5단계: 학습 ---------------------------------------------------------------------

    def train(self, tags) -> None:
        with self._lock:
            self._need_settings()
            if self._training:
                raise StudioError("이미 학습 중입니다.")
            self._require_idle("학습")
            if not self._collector.state()["has_data"]:
                raise StudioError("수집한 데이터가 없습니다. 먼저 3단계에서 수집하세요.")
            config = wizard.save_config(self._workspace, tags)
            self._config = config
            self._reset_model_locked()                       # 실패하면 4단계에 남도록 먼저 표시한다
            self._training = True
            self._error = None
            directory = self._collector.current_dir

        def _run() -> None:
            try:
                train_from_collection(config, directory, self._model_dir)
            except Exception as e:
                with self._lock:
                    self._error = f"학습에 실패했습니다: {e}"
            finally:
                with self._lock:
                    self._training = False

        self._train_thread = threading.Thread(target=_run, daemon=True)
        self._train_thread.start()

    def join_training(self, timeout=None) -> None:
        thread = self._train_thread
        if thread is not None:
            thread.join(timeout)

    # --- 6단계: 감시 ---------------------------------------------------------------------

    @property
    def pipeline(self):
        return self._pipeline

    def start(self) -> None:
        with self._lock:
            settings = self._need_settings()
            if self._pipeline is not None:
                raise StudioError("이미 감시 중입니다.")
            self._require_idle("감시를 시작")
            if not self._model_ready_locked():
                raise StudioError("학습이 끝난 모델이 없습니다. 먼저 4~5단계에서 학습하세요.")
            # 연결 정보가 바뀌었을 수 있으니 같은 태그로 설정 파일을 다시 만든다.
            config = wizard.save_config(self._workspace, wizard.saved_tags(self._workspace))
            self._config = config
            train_fn = make_deviation_train_fn(config, deviation_artifact_path(self._model_dir, config.equipment_id))
            pipeline = self._factory(
                config=config,
                calibration_dir=self._calibration_dir,
                model_dir=self._model_dir,
                train_fn=train_fn,
                history_seconds=DEFAULT_HISTORY_SECONDS,
            )
            if pipeline.calibration_manager.state != CalibrationState.MONITORING:
                raise StudioError("저장된 학습 결과를 불러오지 못했습니다. 4단계에서 다시 학습하세요(콘솔 로그에 이유가 있습니다).")
            try:
                self._io.connect(pipeline, settings["host"], settings["port"])
            except OSError as e:
                raise StudioError(
                    f"브로커({settings['host']}:{settings['port']})에 연결할 수 없습니다. "
                    f"브로커가 실행 중인지 확인하세요. ({e})"
                ) from e
            if self._run_worker:
                pipeline.snapshotter.start()
            self._pipeline = pipeline
            self._error = None

    def stop(self) -> None:
        with self._lock:
            if self._pipeline is None:
                return
            pipeline = self._pipeline
            self._pipeline = None
        if self._run_worker:
            pipeline.snapshotter.stop()
        self._io.disconnect(pipeline)

    def retrain(self) -> None:
        """감시를 멈추고 4단계로 돌아간다(같은 수집본으로 태그를 다시 골라 학습한다)."""
        self.stop()
        with self._lock:
            if self._training:
                raise StudioError("학습이 끝난 뒤에 다시 학습할 수 있습니다.")
            self._reset_model_locked()
            self._error = None

    def shutdown(self) -> None:
        """앱을 끝낼 때: 시험·수집·감시를 모두 닫는다. 이미 닫혀 있어도 오류를 내지 않는다."""
        for action in (self.probe_stop, self.collect_stop, self.stop):
            try:
                action()
            except Exception as e:
                print(f"[studio] 종료 정리 중 오류(무시): {e!r}")

    def feed_record(self, record) -> None:
        """MQTT 스레드가 하는 일(리샘플 후 처리 큐에 넣기)을 그대로 한다. 시험과 재생용."""
        pipeline = self._pipeline
        if pipeline is None:
            raise StudioError("감시 중이 아닙니다.")
        pipeline.snapshotter.submit(pipeline.resampler.add(record))

    def pump(self) -> None:
        """처리 큐를 동기로 비운다(run_worker=False일 때 시험이 쓴다)."""
        if self._pipeline is not None:
            self._pipeline.snapshotter.process_pending()

    # --- 화면용 조회 ----------------------------------------------------------------------

    def _stage_locked(self, collect: dict) -> Stage:
        if self._settings is None:
            return Stage.CONNECT
        if self._pipeline is not None:
            return Stage.MONITOR
        if self._training:
            return Stage.TRAIN
        if collect["active"]:
            return Stage.COLLECT
        if self._model_ready_locked():
            return Stage.MONITOR
        if collect["has_data"]:
            return Stage.TAGS
        return Stage.PROBE

    def _can_locked(self, collect: dict, probe, ready: bool) -> dict:
        settings = self._settings is not None
        running = self._pipeline is not None
        probing = bool(probe and probe["active"])
        collecting = collect["active"]
        idle = settings and not (running or self._training or collecting)
        return {
            "save_connection": not (running or self._training or collecting or probing),
            "probe_start": idle and not probing,
            "probe_stop": probing,
            "collect_start": idle,
            "collect_stop": collecting,
            "discard": idle and collect["has_data"],
            "import": idle,
            "analyze": idle and collect["has_data"],
            "train": idle and collect["has_data"],
            "start": idle and ready,
            "stop": running,
            "retrain": settings and not (self._training or collecting) and (running or ready),
        }

    def _signals(self, config) -> list:
        path = deviation_artifact_path(self._model_dir, config.equipment_id)
        if not path.exists():
            return []
        try:
            artifact = load_deviation_artifact(path)
        except Exception:
            return []
        return [
            {"name": n, "tag": s.tag, "sigma": s.sigma, "limit": s.limit, "move_range": s.move_range}
            for n, s in artifact.signals.items()
        ]

    def state(self) -> dict:
        with self._lock:
            collect = self._collector.state()
            probe = self._probe.snapshot() if self._probe is not None else None
            stage = self._stage_locked(collect)
            config = self._config
            ready = self._model_ready_locked()
            return {
                "stage": stage.value,
                "step": STEP[stage],
                "equipment_id": self._settings["equipment_id"] if self._settings else None,
                "error": self._error,
                "probe": probe,
                "collect": collect,
                "training": self._training,
                "model_ready": ready,
                "running": self._pipeline is not None,
                "connected": bool(self._pipeline is not None and self._io.is_connected(self._pipeline)),
                "signals": self._signals(config) if (ready and config is not None) else [],
                "threshold": config.alarm.threshold if config else None,
                "can": self._can_locked(collect, probe, ready),
            }

    def meta(self) -> dict:
        """기존 점수 그래프 화면(dashboard.html)이 읽는 /api/meta와 같은 모양."""
        with self._lock:
            config, pipeline = self._config, self._pipeline
            if config is None or pipeline is None or pipeline.history is None:
                return {
                    "equipment_id": config.equipment_id if config else "",
                    "groups": [],
                    "threshold": config.alarm.threshold if config else 0,
                    "confirm_steps": config.alarm.confirm_steps if config else 1,
                    "interval_ms": config.resample_interval_ms if config else 100,
                    "history_seconds": DEFAULT_HISTORY_SECONDS,
                    "state": "IDLE",
                }
            return {
                "equipment_id": config.equipment_id,
                "groups": list(pipeline.history.group_names),
                "threshold": config.alarm.threshold,
                "confirm_steps": config.alarm.confirm_steps,
                "interval_ms": config.resample_interval_ms,
                "history_seconds": DEFAULT_HISTORY_SECONDS,
                "state": pipeline.calibration_manager.state.value,
            }

    def scores(self, seconds: float, max_points: int) -> dict:
        pipeline = self._pipeline
        if pipeline is None or pipeline.history is None:
            return dict(_EMPTY_SCORES)
        return pipeline.history.snapshot(seconds, max_points)
```

- [ ] **Step 4: cli 종료 처리를 `shutdown()`으로 바꾼다**

`src/jetson_app/studio/cli.py`의 `finally` 블록에서 `controller.stop()`을 `controller.shutdown()`으로 바꾸고, 바로 위 `try/except`는 지운다:

```python
    finally:
        controller.shutdown()
        server.stop()
```

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest tests/test_studio_controller.py tests/test_studio_cli.py -q`
Expected: 모두 통과. 시간이 오래 걸리면(학습이 3~5초) 정상이다.

- [ ] **Step 6: 커밋**

```bash
git add src/jetson_app/studio/controller.py src/jetson_app/studio/cli.py tests/test_studio_controller.py
git commit -m "feat: studio controller with the six-step flow" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: 서버 라우트

**Files:**
- Modify: `src/jetson_app/studio/server.py`
- Replace: `tests/test_studio_server.py`

**Interfaces:**
- Consumes: Task 5 컨트롤러 메서드.
- Produces: POST `/api/connection`(연결 폼), `/api/probe/start|stop`, `/api/collect/start|stop|discard`, `/api/collect/import`(`{data_dir}`), `/api/analyze`(`{cycle_tag?, with_tags?}`), `/api/train`(`{tags}`), `/api/start`, `/api/stop`, `/api/retrain`. GET `/api/connection`, `/api/state`(+`control_enabled`, `token_required`), `/api/meta`, `/api/scores`.
- 제거: `/api/config`, `/api/recalibrate`.

- [ ] **Step 1: 실패하는 테스트 (서버 테스트 전체 교체)**

<!-- file: tests/test_studio_server.py -->
```python
import http.client
import json
import urllib.error
import urllib.request

import pytest

from jetson_app.studio.controller import StudioController
from jetson_app.studio.server import StudioServer
from studio_fakes import FakeIO, SourceFactory

CONNECTION = {"equipment_id": "nx5", "host": "localhost", "port": 1883, "subscribe_topics": ["dx1/data"]}


def _start(tmp_path, **kwargs):
    controller = StudioController(tmp_path, io=FakeIO(), source_factory=SourceFactory(), run_worker=False)
    server = StudioServer(controller, port=0, **kwargs)
    server.start()
    return controller, server


@pytest.fixture
def open_server(tmp_path):
    controller, server = _start(tmp_path)
    yield controller, server
    server.stop()


def _request(server, method, path, body=None, headers=None):
    url = f"http://127.0.0.1:{server.port}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8") or "null"), response
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        return e.code, (json.loads(raw) if raw.startswith("{") else raw), e


def test_pages_and_health(open_server):
    _, server = open_server

    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/") as response:
        assert response.status == 200 and "text/html" in response.headers["Content-Type"]
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/dashboard") as response:
        assert response.status == 200
    assert _request(server, "GET", "/healthz")[1] == {"ok": True}
    assert _request(server, "GET", "/nope")[0] == 404


def test_state_reports_stage_and_control_flags(open_server):
    _, server = open_server

    status, body, _ = _request(server, "GET", "/api/state")

    assert status == 200
    assert body["stage"] == "CONNECT" and body["step"] == 1
    assert body["control_enabled"] is True and body["token_required"] is False
    assert body["can"]["save_connection"] is True


def test_the_connection_round_trips_and_moves_to_the_probe_step(open_server):
    _, server = open_server

    assert _request(server, "GET", "/api/connection")[1] == {}
    assert _request(server, "POST", "/api/connection", CONNECTION)[0] == 200

    assert _request(server, "GET", "/api/connection")[1]["subscribe_topics"] == ["dx1/data"]
    assert _request(server, "GET", "/api/state")[1]["stage"] == "PROBE"
    assert _request(server, "GET", "/api/meta")[1]["state"] == "IDLE"
    assert _request(server, "GET", "/api/scores?seconds=60")[1]["points"] == []


def test_user_errors_come_back_as_400_with_the_message(open_server):
    _, server = open_server

    status, body, _ = _request(server, "POST", "/api/start", {})
    assert status == 400 and "연결 정보" in body["error"]

    status, body, _ = _request(server, "POST", "/api/collect/import", {"data_dir": "C:/definitely/not/here"})
    assert status == 400 and "연결 정보" in body["error"]

    _request(server, "POST", "/api/connection", CONNECTION)
    status, body, _ = _request(server, "POST", "/api/collect/import", {"data_dir": "C:/definitely/not/here"})
    assert status == 400 and "찾을 수 없습니다" in body["error"]

    status, body, _ = _request(server, "POST", "/api/analyze", {})
    assert status == 400 and "수집" in body["error"]

    status, body, _ = _request(server, "POST", "/api/connection", dict(CONNECTION, host="localhost:1883"))
    assert status == 400 and "포트" in body["error"]


def test_wrong_body_types_are_400_not_500(open_server):
    _, server = open_server
    _request(server, "POST", "/api/connection", CONNECTION)

    assert _request(server, "POST", "/api/train", {"tags": "pos"})[0] == 400
    assert _request(server, "POST", "/api/analyze", {"cycle_tag": 5})[0] == 400
    assert _request(server, "POST", "/api/collect/import", {"data_dir": ["x"]})[0] == 400


def test_bad_json_and_non_object_bodies_are_400(open_server):
    _, server = open_server
    url = f"http://127.0.0.1:{server.port}/api/start"

    for raw in (b"{not json", b"[1, 2]"):
        request = urllib.request.Request(url, data=raw, method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code == 400


def test_oversized_body_is_rejected_with_413(open_server):
    _, server = open_server
    conn = http.client.HTTPConnection("127.0.0.1", server.port)
    conn.putrequest("POST", "/api/analyze")
    conn.putheader("Content-Length", str(70_000))
    conn.endheaders()

    response = conn.getresponse()

    assert response.status == 413
    conn.close()


def test_post_with_a_foreign_origin_is_rejected(open_server):
    _, server = open_server

    status, body, _ = _request(server, "POST", "/api/start", {}, headers={"Origin": "http://evil.example"})

    assert status == 403


def test_post_with_a_foreign_host_header_is_rejected_on_loopback(open_server):
    _, server = open_server
    conn = http.client.HTTPConnection("127.0.0.1", server.port)

    conn.request("POST", "/api/start", body=b"{}", headers={"Host": "evil.example"})

    assert conn.getresponse().status == 403
    conn.close()


def test_token_is_required_for_post_when_configured(tmp_path):
    _, server = _start(tmp_path, token="s3cret")
    try:
        assert _request(server, "GET", "/api/state")[1]["token_required"] is True
        assert _request(server, "POST", "/api/connection", CONNECTION)[0] == 403
        assert _request(server, "POST", "/api/connection", CONNECTION, headers={"X-Studio-Token": "wrong"})[0] == 403
        assert _request(server, "POST", "/api/connection", CONNECTION, headers={"X-Studio-Token": "s3cret"})[0] == 200
    finally:
        server.stop()


def test_control_can_be_disabled_entirely(tmp_path):
    _, server = _start(tmp_path, control_enabled=False)
    try:
        status, body, _ = _request(server, "POST", "/api/connection", CONNECTION)
        assert status == 403 and "꺼져" in body["error"]
        assert _request(server, "GET", "/api/state")[1]["control_enabled"] is False
    finally:
        server.stop()


def test_a_second_server_cannot_bind_a_port_that_is_already_serving(open_server, tmp_path):
    _, server = open_server

    with pytest.raises(OSError):
        StudioServer(
            StudioController(tmp_path / "other", io=FakeIO(), source_factory=SourceFactory(), run_worker=False),
            port=server.port,
        )


def test_negative_content_length_is_rejected(open_server):
    _, server = open_server
    conn = http.client.HTTPConnection("127.0.0.1", server.port)
    conn.putrequest("POST", "/api/start")
    conn.putheader("Content-Length", "-5")
    conn.endheaders()

    assert conn.getresponse().status == 400
    conn.close()
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_studio_server.py -q`
Expected: FAIL (`/api/connection` 없음 등)

- [ ] **Step 3: 라우트를 바꾼다**

`server.py`의 `self._post_routes = {...}` 블록을 아래로 교체하고, 모듈 아래쪽 `_number` 함수 옆에 `_text`/`_text_list` 도우미를 추가한다.

```python
        self._post_routes = {
            "/api/connection": lambda body: controller.save_connection(body),
            "/api/probe/start": lambda body: controller.probe_start(),
            "/api/probe/stop": lambda body: controller.probe_stop(),
            "/api/collect/start": lambda body: controller.collect_start(),
            "/api/collect/stop": lambda body: controller.collect_stop(),
            "/api/collect/discard": lambda body: controller.discard(),
            "/api/collect/import": lambda body: controller.import_folder(_text(body, "data_dir")),
            "/api/analyze": lambda body: controller.analyze(
                _text(body, "cycle_tag", required=False) or None, with_tags=body.get("with_tags", True) is not False
            ),
            "/api/train": lambda body: controller.train(_text_list(body, "tags")),
            "/api/start": lambda body: controller.start(),
            "/api/stop": lambda body: controller.stop(),
            "/api/retrain": lambda body: controller.retrain(),
        }
```

GET 쪽 `/api/config` 분기는 이렇게 바꾼다:

```python
        elif url.path == "/api/connection":
            self._send_json(h, 200, c.connection())
```

모듈 끝에 추가:

```python
def _text(body: dict, key: str, required: bool = True) -> str:
    value = body.get(key, "" if required else None)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise StudioError(f"'{key}'는 문자열이어야 합니다.")
    return value


def _text_list(body: dict, key: str) -> list:
    value = body.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise StudioError(f"'{key}'는 문자열의 목록이어야 합니다.")
    return value
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_studio_server.py -q`
Expected: 모두 통과

- [ ] **Step 5: 커밋**

```bash
git add src/jetson_app/studio/server.py tests/test_studio_server.py
git commit -m "feat: studio server routes for the six-step flow" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 6단계 화면 (`studio.html`)

**Files:**
- Replace: `src/jetson_app/web/studio.html`, `tests/test_studio_page.py`

**Interfaces:**
- Consumes: Task 6의 API, `/api/state`의 `can`, `collect`, `probe`, `signals` 키.
- Produces: 요소 id — `equipment_id, subscribe_topics, host, port, btn-save-connection, btn-probe-start, btn-probe-stop, probe-info, probe-message, probe-tags, btn-collect-start, btn-collect-stop, btn-discard, import_dir, btn-import, collect-info, collect-cycles, btn-analyze, tags, cycle-tag, cycle-info, train-warning, btn-train, signals, dash, btn-start, btn-stop, btn-retrain`.

- [ ] **Step 1: 실패하는 테스트**

<!-- file: tests/test_studio_page.py -->
```python
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "src" / "jetson_app" / "web" / "studio.html"


def _html():
    return PAGE.read_text(encoding="utf-8")


def test_page_has_the_six_steps_and_main_controls():
    html = _html()

    for n in range(1, 7):
        assert f'data-step="{n}"' in html
    for control_id in (
        "btn-save-connection", "btn-probe-start", "btn-probe-stop", "btn-collect-start", "btn-collect-stop",
        "btn-discard", "btn-import", "btn-analyze", "btn-train", "btn-start", "btn-stop", "btn-retrain",
        "cycle-tag", "train-warning",
    ):
        assert f'id="{control_id}"' in html


def test_page_is_self_contained():
    html = _html()

    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html)
    assert "innerHTML" not in html                       # 서버가 준 문자열은 textContent로만 넣는다
    assert "X-Studio-Token" in html and "/api/state" in html and "/dashboard" in html


def test_buttons_follow_what_the_server_says_is_allowed():
    html = _html()
    render = html.split("function render(s)")[1].split("async function refresh")[0]

    assert "s.can" in render
    for api in ("/api/probe/start", "/api/collect/start", "/api/collect/import", "/api/analyze", "/api/train",
                "/api/start", "/api/retrain", "/api/connection"):
        assert api in html


def test_page_explains_the_limits_the_user_must_know():
    html = _html()

    assert "10" in html and "사이클" in html                # 최소 수집량 안내
    assert "축이 움직" in html                              # 멈춘 상태만 모으면 추천이 비는 이유
    assert "1GB" in html                                    # 수집 상한 도달 안내


def test_state_banner_is_cleared_when_its_condition_goes_away():
    html = _html()

    assert "function showBanner" in html and "stateMessage" in html
    assert 'banner("")' not in html.split("function render(s)")[1].split("async function refresh")[0]
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_studio_page.py -q`
Expected: FAIL (옛 화면)

- [ ] **Step 3: 화면을 쓴다**

<!-- file: src/jetson_app/web/studio.html -->
```html
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jetson-studio</title>
<style>
:root { color-scheme: light dark; --bg:#f6f7f9; --card:#fff; --ink:#1b1f24; --sub:#5b6570; --line:#d8dde3;
  --accent:#1f6feb; --ok:#1a7f37; --warn:#9a6700; --bad:#cf222e; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f1318; --card:#171c22; --ink:#e6e9ed; --sub:#97a1ac; --line:#2b333c;
  --accent:#58a6ff; --ok:#3fb950; --warn:#d29922; --bad:#f85149; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui, "Malgun Gothic", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 16px; }
h1 { font-size: 18px; margin: 4px 0 12px; }
.steps { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
.steps span { padding:4px 12px; border:1px solid var(--line); border-radius:999px; color:var(--sub); background:var(--card); }
.steps span.done { color:var(--ok); border-color:var(--ok); }
.steps span.now { color:#fff; background:var(--accent); border-color:var(--accent); }
#banner { display:none; padding:8px 12px; margin-bottom:12px; border-radius:6px; border:1px solid var(--bad); color:var(--bad); background:var(--card); }
#banner.warn { border-color:var(--warn); color:var(--warn); }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 16px; margin-bottom:12px; opacity:.6; }
.card.active { opacity:1; border-color:var(--accent); }
.card h2 { font-size:15px; margin:0 0 8px; }
label { display:block; margin:6px 0 2px; color:var(--sub); font-size:12px; }
input[type=text], input[type=number], select { width:100%; max-width:420px; padding:6px 8px; border:1px solid var(--line); border-radius:6px;
  background:var(--bg); color:var(--ink); }
.row { display:flex; gap:12px; flex-wrap:wrap; }
.row > div { flex:1 1 200px; }
button { padding:6px 14px; border:1px solid var(--accent); background:var(--accent); color:#fff; border-radius:6px; cursor:pointer; margin:8px 8px 0 0; }
button.secondary { background:transparent; color:var(--accent); }
button:disabled { opacity:.4; cursor:not-allowed; }
.table { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:12px; margin-top:8px; }
th, td { border-bottom:1px solid var(--line); padding:4px 8px; text-align:left; white-space:nowrap; }
td.reason { white-space:normal; color:var(--sub); }
.hint { color:var(--sub); font-size:12px; }
.info { margin-top:8px; }
.warnbox { color:var(--warn); margin-top:8px; }
pre { margin:8px 0 0; padding:8px; max-height:120px; overflow:auto; background:var(--bg); border:1px solid var(--line); border-radius:6px; font-size:11px; white-space:pre-wrap; word-break:break-all; }
iframe { width:100%; height:520px; border:1px solid var(--line); border-radius:6px; background:var(--card); }
</style>
</head>
<body>
<main>
<h1>jetson-studio</h1>
<div class="steps" id="steps"><span data-s="1">1 연결</span><span data-s="2">2 살펴보기</span><span data-s="3">3 전체 수집</span><span data-s="4">4 태그 추천</span><span data-s="5">5 학습</span><span data-s="6">6 감시</span></div>
<div id="banner"></div>

<section class="card" data-step="1">
  <h2>1. 연결</h2>
  <div class="row">
    <div><label for="equipment_id">장비 이름 (영문, 숫자, _ -)</label><input type="text" id="equipment_id" placeholder="nx5"></div>
    <div><label for="subscribe_topics">구독 토픽 (쉼표로 구분)</label><input type="text" id="subscribe_topics" placeholder="dx1/AxisData, dx1/ProcStart"></div>
  </div>
  <div class="row">
    <div><label for="host">브로커 주소</label><input type="text" id="host" value="localhost"></div>
    <div><label for="port">포트</label><input type="number" id="port" value="1883"></div>
  </div>
  <p class="hint">사이클 수를 정확히 세려면 사이클 시작 신호나 카운터가 있는 토픽도 같이 적으세요. 없으면 움직임의 반복으로 추정합니다.</p>
  <button id="btn-save-connection">연결 정보 저장</button>
</section>

<section class="card" data-step="2">
  <h2>2. 살펴보기</h2>
  <p class="hint">토픽으로 데이터가 들어오는지만 확인합니다. 아무것도 저장하지 않고, 끝내면 연결을 끊습니다(5분 뒤 자동 종료).</p>
  <button id="btn-probe-start">시험 시작</button>
  <button id="btn-probe-stop" class="secondary">시험 종료</button>
  <div id="probe-info" class="info"></div>
  <pre id="probe-message" hidden></pre>
  <div id="probe-tags" class="hint"></div>
</section>

<section class="card" data-step="3">
  <h2>3. 전체 수집</h2>
  <p class="hint">정상으로 운전하는 동안 들어오는 데이터를 원본 그대로 모읍니다. 정상 사이클이 10회 이상 들어가야 기준이 안정됩니다.
    수집하는 동안 축이 움직이는 사이클이 들어가야 태그를 추천할 수 있습니다(멈춘 상태만 모으면 추천이 비어 나옵니다).
    이상 상황은 섞이지 않게 하세요. 수집은 1GB에 닿으면 자동으로 끝납니다.</p>
  <button id="btn-collect-start">수집 시작</button>
  <button id="btn-collect-stop" class="secondary">수집 종료</button>
  <button id="btn-discard" class="secondary">수집한 데이터 지우기</button>
  <div id="collect-info" class="info"></div>
  <div id="collect-cycles" class="hint"></div>
  <label for="import_dir">또는 이미 저장해 둔 데이터 폴더(DX1 JSON)를 수집 데이터로 불러오기</label>
  <input type="text" id="import_dir" placeholder="C:\data\normal">
  <button id="btn-import" class="secondary">폴더 불러오기</button>
</section>

<section class="card" data-step="4">
  <h2>4. 태그 추천</h2>
  <p class="hint">수집한 데이터로 점수에 쓸 태그를 추천합니다. 추천 태그는 평소 조용하고 값의 범위가 넓은 신호(예: 서보 위치)입니다.</p>
  <button id="btn-analyze" class="secondary">태그 분석</button>
  <div class="table"><table id="tags" hidden>
    <thead><tr><th>선택</th><th>태그</th><th>종류</th><th>대기잡음</th><th>범위/잡음</th><th>판정과 이유</th></tr></thead><tbody></tbody>
  </table></div>
  <label for="cycle-tag">사이클 기준 신호</label>
  <select id="cycle-tag"><option value="">자동 선택</option></select>
  <div id="cycle-info" class="hint"></div>
</section>

<section class="card" data-step="5">
  <h2>5. 학습</h2>
  <p class="hint">고른 태그로 신호별 잡음 크기와 알람 기준을 계산합니다. 몇 초에서 몇 분 걸립니다.</p>
  <div id="train-warning" class="warnbox" hidden></div>
  <button id="btn-train">학습 시작</button>
  <div class="table"><table id="signals" hidden>
    <thead><tr><th>신호</th><th>태그</th><th>대기잡음 σ</th><th>알람 기준</th><th>움직임 기준</th></tr></thead><tbody></tbody>
  </table></div>
</section>

<section class="card" data-step="6">
  <h2>6. 감시</h2>
  <button id="btn-start">감시 시작</button>
  <button id="btn-stop" class="secondary">감시 중지</button>
  <button id="btn-retrain" class="secondary">다시 학습 (4단계로)</button>
  <div id="monitor-info" class="hint"></div>
  <iframe id="dash" title="점수 그래프"></iframe>
</section>
</main>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
let token = "";
try {
  token = new URLSearchParams(location.search).get("token") || sessionStorage.getItem("studioToken") || "";
  if (token) sessionStorage.setItem("studioToken", token);
} catch (e) { /* 저장소를 못 써도 화면은 동작한다 */ }

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Studio-Token": token },
    body: JSON.stringify(body),
  };
  let response;
  try { response = await fetch(path, options); }
  catch (e) { throw new Error("앱에 연결할 수 없습니다. jetson-studio가 실행 중인지 확인하세요."); }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || ("요청이 실패했습니다 (" + response.status + ")"));
  return data;
}

// 배너는 세 곳에서 온다: 방금 한 조작의 오류(actionError), 서버와의 연결 문제(connError), 서버 상태에서 읽은 안내(stateMessage).
// 상태 안내와 연결 문제는 조건이 사라지면 저절로 지워지고, 조작 오류는 다음 조작이 성공할 때 지워진다.
let actionError = "", connError = "", stateMessage = null;
function showBanner() {
  const shown = actionError ? { text: actionError, warn: false }
    : connError ? { text: connError, warn: false }
    : (stateMessage || { text: "", warn: false });
  const el = $("banner");
  el.textContent = shown.text;
  el.className = shown.warn ? "warn" : "";
  el.style.display = shown.text ? "block" : "none";
}
async function act(fn) {
  try { await fn(); actionError = ""; } catch (e) { actionError = e.message; }
  showBanner();
  refresh();
}
function cell(row, text, cls) {
  const td = document.createElement("td");
  td.textContent = text;
  if (cls) td.className = cls;
  row.appendChild(td);
  return td;
}
const fmt = (v) => (v === null || v === undefined) ? "-" : Number(v).toPrecision(3);
const fmtBytes = (n) => n >= 1048576 ? (n / 1048576).toFixed(1) + " MB" : (n / 1024).toFixed(1) + " KB";
function fmtDuration(seconds) {
  const s = Math.round(seconds);
  return Math.floor(s / 60) + "분 " + (s % 60) + "초";
}

let lastState = null, analysisShown = false;

function renderTags(tags) {
  const table = $("tags"), body = table.querySelector("tbody");
  body.replaceChildren();
  for (const t of tags) {
    const row = document.createElement("tr");
    const box = document.createElement("input");
    box.type = "checkbox"; box.value = t.tag; box.checked = t.recommended;
    box.disabled = t.kind !== "continuous";
    cell(row, "").appendChild(box);
    cell(row, t.tag); cell(row, t.kind); cell(row, fmt(t.sigma));
    cell(row, t.sigma === null ? "-" : "10^" + t.dynamic_range.toFixed(1));
    cell(row, (t.recommended ? "추천: " : "") + t.reason, "reason");
    body.appendChild(row);
  }
  table.hidden = tags.length === 0;
}
function selectedTags() {
  return Array.from($("tags").querySelectorAll("input:checked")).map((b) => b.value);
}
function renderCycles(cycles) {
  const select = $("cycle-tag"), keep = select.value;
  select.replaceChildren();
  const auto = document.createElement("option");
  auto.value = ""; auto.textContent = "자동 선택";
  select.appendChild(auto);
  for (const c of cycles.candidates) {
    const option = document.createElement("option");
    option.value = c.tag;
    option.textContent = c.tag + " (" + (c.method === "edge" ? "0/1 신호" : "카운터") + ", " + c.cycles + "사이클)";
    select.appendChild(option);
  }
  select.value = Array.from(select.options).some((o) => o.value === keep) ? keep : "";
  $("cycle-info").textContent = cycles.message;
  const warning = $("train-warning");
  warning.textContent = cycles.warning || "";
  warning.hidden = !cycles.warning;
}
async function runAnalyze(withTags) {
  const result = await api("/api/analyze", { cycle_tag: $("cycle-tag").value, with_tags: withTags });
  if (result.tags) renderTags(result.tags);
  renderCycles(result.cycles);
}
function renderSignals(signals) {
  const table = $("signals"), body = table.querySelector("tbody");
  body.replaceChildren();
  for (const s of signals) {
    const row = document.createElement("tr");
    cell(row, s.name); cell(row, s.tag); cell(row, fmt(s.sigma)); cell(row, fmt(s.limit)); cell(row, fmt(s.move_range));
    body.appendChild(row);
  }
  table.hidden = signals.length === 0;
}
function renderProbe(p) {
  const info = $("probe-info"), message = $("probe-message");
  if (!p || (!p.active && p.messages === 0)) {
    info.textContent = ""; message.hidden = true; $("probe-tags").textContent = "";
    return;
  }
  const parts = [p.active ? (p.connected ? "연결됨" : "연결 중이거나 끊김") : "종료됨",
    "수신 " + p.messages + "건", "초당 " + p.rate.toFixed(1) + "건"];
  if (p.last_rx_age !== null) parts.push("마지막 수신 " + p.last_rx_age.toFixed(1) + "초 전");
  if (p.topics.length) parts.push("토픽: " + p.topics.join(", "));
  info.textContent = parts.join(" · ");
  message.textContent = p.last_message;
  message.hidden = !p.last_message;
  const shown = p.tags.slice(0, 40).join(", ") + (p.tag_count > 40 ? " 외 " + (p.tag_count - 40) + "개" : "");
  $("probe-tags").textContent = p.tag_count ? ("숫자 태그 " + p.tag_count + "개: " + shown) : "";
}
function renderCollect(c) {
  const parts = [];
  if (c.active) parts.push(c.connected ? "수집 중" : "수집 중(브로커 연결이 끊겼습니다. 자동으로 다시 연결합니다)");
  else if (c.has_data) parts.push(c.interrupted ? "중단된 수집본" : "수집 완료");
  if (c.has_data || c.active) {
    parts.push(fmtDuration(c.seconds), "수신 " + c.messages + "건", fmtBytes(c.bytes) + " / " + fmtBytes(c.limit_bytes));
  }
  $("collect-info").textContent = parts.join(" · ");
  $("collect-cycles").textContent = c.cycles ? c.cycles.message : "";
}

function render(s) {
  lastState = s;
  const now = s.step;
  document.querySelectorAll("#steps span").forEach((el) => {
    const n = Number(el.dataset.s);
    el.className = n < now ? "done" : (n === now ? "now" : "");
  });
  document.querySelectorAll(".card").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.step) === now);
  });
  const control = s.control_enabled, can = s.can;
  const enable = (id, allowed) => { $(id).disabled = !control || !allowed; };
  enable("btn-save-connection", can.save_connection);
  enable("btn-probe-start", can.probe_start);
  enable("btn-probe-stop", can.probe_stop);
  enable("btn-collect-start", can.collect_start);
  enable("btn-collect-stop", can.collect_stop);
  enable("btn-discard", can.discard);
  enable("btn-import", can.import);
  enable("btn-analyze", can.analyze);
  enable("btn-train", can.train && selectedTags().length > 0);
  enable("btn-start", can.start);
  enable("btn-stop", can.stop);
  enable("btn-retrain", can.retrain);
  renderProbe(s.probe);
  renderCollect(s.collect);
  renderSignals(s.signals);
  $("monitor-info").textContent = s.running ? (s.connected ? "감시 중 · 브로커 연결됨" : "감시 중 · 브로커 연결이 끊겼습니다. 자동으로 다시 연결합니다.")
    : (s.model_ready ? "학습한 모델이 있습니다. 감시를 시작할 수 있습니다." : "");
  if (s.running && !$("dash").getAttribute("src")) $("dash").src = "/dashboard";
  if (!s.running) $("dash").removeAttribute("src");
  if (s.stage === "TAGS" && can.analyze && !analysisShown) {
    analysisShown = true;
    act(() => runAnalyze(true));
  }
  if (s.stage !== "TAGS") analysisShown = false;
  const p = s.probe, c = s.collect;
  if (!control) stateMessage = { text: "조작 기능이 꺼져 있습니다(읽기 전용). 이 장비에서 열거나 --token을 지정해 다시 시작하세요.", warn: true };
  else if (s.error) stateMessage = { text: s.error, warn: false };
  else if (c.no_data) stateMessage = { text: "30초 넘게 데이터가 들어오지 않습니다. 토픽 이름과 DX1 발행 상태를 확인하세요.", warn: true };
  else if (c.ended_reason === "size" && !c.active) stateMessage = { text: "수집이 크기 상한(1GB)에 닿아 자동으로 끝났습니다.", warn: true };
  else if (c.interrupted) stateMessage = { text: "이전 수집이 중간에 끊겼습니다(앱이 꺼졌을 수 있습니다). 그때까지 모인 데이터는 남아 있습니다.", warn: true };
  else if (p && p.expired && !p.active) stateMessage = { text: "시험이 5분이 지나 자동으로 끝났습니다.", warn: true };
  else stateMessage = null;
  showBanner();
}
async function refresh() {
  try { render(await api("/api/state")); connError = ""; }
  catch (e) { connError = e.message; }
  showBanner();
}

$("btn-save-connection").onclick = () => act(() => api("/api/connection", {
  equipment_id: $("equipment_id").value,
  host: $("host").value,
  port: $("port").value,
  subscribe_topics: $("subscribe_topics").value.split(",").map((t) => t.trim()).filter(Boolean),
}));
$("btn-probe-start").onclick = () => act(() => api("/api/probe/start", {}));
$("btn-probe-stop").onclick = () => act(() => api("/api/probe/stop", {}));
$("btn-collect-start").onclick = () => act(() => api("/api/collect/start", {}));
$("btn-collect-stop").onclick = () => act(() => api("/api/collect/stop", {}));
$("btn-discard").onclick = () => act(() => api("/api/collect/discard", {}));
$("btn-import").onclick = () => act(() => api("/api/collect/import", { data_dir: $("import_dir").value }));
$("btn-analyze").onclick = () => act(() => runAnalyze(true));
$("cycle-tag").onchange = () => act(() => runAnalyze(false));
$("tags").onchange = () => { if (lastState) render(lastState); };
$("btn-train").onclick = () => act(() => api("/api/train", { tags: selectedTags() }));
$("btn-start").onclick = () => act(() => api("/api/start", {}));
$("btn-stop").onclick = () => act(() => api("/api/stop", {}));
$("btn-retrain").onclick = () => act(() => api("/api/retrain", {}));

(async function init() {
  try {
    const form = await api("/api/connection");
    if (form.equipment_id) {
      $("equipment_id").value = form.equipment_id;
      $("host").value = form.host;
      $("port").value = form.port;
      $("subscribe_topics").value = form.subscribe_topics.join(", ");
    }
  } catch (e) { connError = e.message; showBanner(); }
  refresh();
  setInterval(refresh, 1000);
})();
</script>
</body>
</html>
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_studio_page.py -q`
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add src/jetson_app/web/studio.html tests/test_studio_page.py
git commit -m "feat: studio six-step screen" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: 끝에서 끝까지 시험, 실제 데이터 검증, 문서, exe

**Files:**
- Replace: `tests/test_studio_flow.py`, `tests/test_studio_real_data.py`
- Modify: `README.md`, `docs/DEVELOPER_OVERVIEW.md`, `docs/superpowers/specs/2026-10-15-studio-six-step-flow-design.md`(§6 라우트 반영)
- Rebuild: `dist/jetson-studio/jetson-studio.exe` (커밋 대상 아님)

- [ ] **Step 1: HTTP 전체 흐름 시험과 실제 run3 데이터 시험을 쓴다**

<!-- file: tests/test_studio_flow.py -->
```python
"""화면이 쓰는 HTTP API만으로 연결 → 살펴보기 → 수집 → 태그 추천 → 학습 → 감시 → 충격 알람까지 끝낸다(MQTT는 가짜)."""
import json
import urllib.request
from types import SimpleNamespace

from jetson_app.studio.controller import StudioController
from jetson_app.studio.server import StudioServer
from studio_fakes import FakeIO, SourceFactory, noise, payload


def _call(server, method, path, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=data, method=method)
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def test_whole_workflow_over_http(tmp_path):
    factory = SourceFactory()
    controller = StudioController(tmp_path, io=FakeIO(), source_factory=factory, run_worker=False)
    server = StudioServer(controller, port=0)
    server.start()
    try:
        assert _call(server, "GET", "/api/state")["stage"] == "CONNECT"
        _call(server, "POST", "/api/connection", {
            "equipment_id": "demo", "host": "localhost", "port": 1883, "subscribe_topics": ["dx1/data"],
        })
        assert _call(server, "GET", "/api/state")["stage"] == "PROBE"

        _call(server, "POST", "/api/probe/start", {})
        factory.last.emit(payload(0, pos=1.0, cyc=0))
        probe = _call(server, "GET", "/api/state")["probe"]
        assert probe["messages"] == 1 and probe["tags"] == ["cyc", "pos"]
        _call(server, "POST", "/api/probe/stop", {})

        _call(server, "POST", "/api/collect/start", {})
        for i in range(400):
            factory.last.emit(payload(i, pos=noise(i), cyc=1 if i % 100 >= 90 else 0))
        assert _call(server, "GET", "/api/state")["collect"]["messages"] == 400
        _call(server, "POST", "/api/collect/stop", {})
        assert _call(server, "GET", "/api/state")["stage"] == "TAGS"

        analysis = _call(server, "POST", "/api/analyze", {})
        assert {t["tag"] for t in analysis["tags"]} == {"pos", "cyc"}
        assert analysis["cycles"]["method"] == "edge" and analysis["cycles"]["warning"]

        _call(server, "POST", "/api/train", {"tags": ["pos"]})
        controller.join_training(timeout=30)
        state = _call(server, "GET", "/api/state")
        assert state["stage"] == "MONITOR" and state["signals"][0]["name"] == "pos"

        _call(server, "POST", "/api/start", {})
        for i in range(400, 700):
            value = noise(i) + (0.01 if i == 500 else 0.0)
            message = SimpleNamespace(payload=payload(i, pos=value))
            controller.pipeline.mqtt_subscriber._handle_message(None, None, message)
        controller.pump()
        points = _call(server, "GET", "/api/scores?seconds=3600&max_points=5000")["points"]
        assert sum(1 for p in points if p["g"]["pos"][1]) == 1

        _call(server, "POST", "/api/retrain", {})
        state = _call(server, "GET", "/api/state")
        assert state["stage"] == "TAGS" and state["running"] is False
    finally:
        server.stop()
```

<!-- file: tests/test_studio_real_data.py -->
```python
"""실제 NX5 데이터(JSONData/run3)로 6단계 흐름의 핵심을 확인한다. 데이터가 없는 체크아웃에서는 건너뛴다."""
import shutil
from pathlib import Path

import pytest

from jetson_app.studio.controller import StudioController
from studio_fakes import SourceFactory

RUN3 = Path(__file__).resolve().parents[2] / "JSONData" / "run3"
TAGS = [
    "NX5_AxisData:AxX_Act_Pos",
    "NX5_AxisData:AxZ_Act_Pos",
    "NX5_AxisData:AxCV_Act_Pos",
]
EXPECTED_SIGMA = {
    "axx_act_pos": 3.92e-5,
    "axz_act_pos": 3.53e-6,
    "axcv_act_pos": 3.44e-5,
}
CONNECTION = {
    "equipment_id": "nx5", "host": "localhost", "port": 1883,
    "subscribe_topics": ["dx1/AxisData", "dx1/ProcStart"],
}
needs_data = pytest.mark.skipif(not RUN3.is_dir(), reason="JSONData/run3 이 없는 체크아웃에서는 건너뜁니다")


class FakeIO:
    def connect(self, pipeline, host, port):
        pass

    def disconnect(self, pipeline):
        pass

    def is_connected(self, pipeline):
        return True


@needs_data
def test_run3_recommends_the_position_tags_counts_cycles_and_reproduces_the_noise_levels(tmp_path):
    controller = StudioController(tmp_path, io=FakeIO(), source_factory=SourceFactory(), run_worker=False)
    controller.save_connection(CONNECTION)
    controller.import_folder(str(RUN3))

    analysis = controller.analyze()
    recommended = {t["tag"] for t in analysis["tags"] if t["recommended"]}
    assert set(TAGS) <= recommended
    assert analysis["cycles"]["method"] == "edge" and 8 <= analysis["cycles"]["cycles"] <= 12

    controller.train(TAGS)
    controller.join_training(timeout=300)

    state = controller.state()
    assert state["stage"] == "MONITOR", state["error"]
    sigma = {s["name"]: s["sigma"] for s in state["signals"]}
    for name, expected in EXPECTED_SIGMA.items():
        assert sigma[name] == pytest.approx(expected, rel=0.1), name


@needs_data
def test_without_a_cycle_signal_the_period_estimate_still_finds_about_ten_cycles(tmp_path):
    only_axes = tmp_path / "axes_only"
    only_axes.mkdir()
    shutil.copy2(RUN3 / "DX_AxisData.json", only_axes / "DX_AxisData.json")
    controller = StudioController(tmp_path / "ws", io=FakeIO(), source_factory=SourceFactory(), run_worker=False)
    controller.save_connection(dict(CONNECTION, subscribe_topics=["dx1/AxisData"]))
    controller.import_folder(str(only_axes))

    cycles = controller.analyze(with_tags=False)["cycles"]

    assert cycles["method"] == "period" and cycles["exact"] is False
    assert 6 <= cycles["cycles"] <= 14, cycles["message"]
```

- [ ] **Step 2: 전체 테스트를 돌린다**

Run: `uv run pytest -q`
Expected: 전부 통과. 실제 데이터 시험이 실패하면 가정(추천 3개, 사이클 8~12, σ ±10%, 주기 추정 6~14)을 맞추려고 임계값을 만지기 전에 실제 값(`analysis["cycles"]`, `sigma`)을 출력해 원인을 확인하고, 필요하면 `trainer`의 끊김 표시(`reset_window` 빈 샘플)를 빼 본다.

- [ ] **Step 3: 문서를 고친다**

- `README.md`의 "jetson-studio" 절을 6단계로 고친다(연결, 살펴보기, 전체 수집, 태그 추천, 학습, 감시. 사이클 10회 권장과 경고만 한다는 점, 사이클 세는 방법 A/B/D, 1GB 상한, 폴더 불러오기, 수집 중 축이 움직여야 추천이 나온다는 점).
- `docs/DEVELOPER_OVERVIEW.md`의 상태 행·모듈 표(`studio/payloads|cycles|raw_source|probe|collector|trainer`)·테스트 수를 갱신한다.
- 스펙 §6 표를 구현한 라우트에 맞춘다: `GET /api/probe`, `GET /api/collect`는 `/api/state`의 `probe`, `collect`로 합쳤고, `/api/config`는 `/api/connection`, `/api/recalibrate`는 `/api/collect/discard`와 `/api/retrain`으로 나뉘었다고 적는다.

- [ ] **Step 4: exe를 다시 빌드해 확인한다**

실행 중인 `jetson-studio.exe`가 있으면 먼저 끈다(`taskkill /F /IM jetson-studio.exe`).
Run: `powershell -ExecutionPolicy Bypass -File packaging/build_exe.ps1`
그다음 `dist/jetson-studio/jetson-studio.exe --no-browser --workspace $env:TEMP/studio_ws_check`를 띄워 `curl http://localhost:8800/healthz`가 `{"ok": true}`, `curl http://localhost:8800/api/state`의 `stage`가 `CONNECT`인지 확인하고 종료한다.

- [ ] **Step 5: 커밋과 푸시**

```bash
git add tests/test_studio_flow.py tests/test_studio_real_data.py README.md docs/DEVELOPER_OVERVIEW.md docs/superpowers/specs/2026-10-15-studio-six-step-flow-design.md
git commit -m "test+docs: studio six-step end-to-end and real-data checks" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git push origin main
```

---

## 마지막 점검 (모든 Task 뒤)

- 새 눈으로 보는 리뷰어 한 명(가장 능력 있는 모델)에게 브랜치 전체를 검토시키고, Critical·Important는 테스트를 먼저 쓰고 고친다.
- 수동 확인(Task 11 상당)은 사용자와 실제 브로커(`192.168.250.200`)가 필요하다: 시험 → 짧은 수집 → 태그 추천 → 학습 → 감시. 이 계획의 자동 시험은 MQTT를 가짜로 대신한다.

## Self-Review (작성 후 점검 결과)

- **스펙 범위**: §3 단계별 동작 → Task 5·7, §4 구조 → Task 1~7, §5 사이클(A/B/D/E) → Task 1, §6 API → Task 6, §7 저장 위치 → Task 3, §8 오류 처리(연결 실패·끊김·상한·추천 없음·10 미만·학습 실패·재시작·수집 중 종료) → Task 3·5·7, §9 테스트 → 각 Task와 Task 8. §10 하지 않는 것은 구현하지 않는다.
- **스펙과 다르게 한 점**: 스펙 §6의 `GET /api/probe`, `GET /api/collect`는 `/api/state`에 합쳤다(한 번의 폴링으로 충분). Task 8 Step 3에서 스펙을 고친다. 수집 중 연결이 끊긴 구간을 `meta`에 기록하는 항목은 `connected` 표시와 재연결(paho 자동)로 줄였다.
- **자리표시자 점검**: "TBD/TODO" 없음. Task 4 Step 4의 조건부 안내는 실패 시 확인 방법이다.
- **타입 일관성**: `Collector.state()` 키, `ProbeSession.snapshot()` 키, `controller.state()["can"]` 키가 Task 5·7 화면 코드와 일치하는지 실행 중 확인한다(화면은 `s.can.<name>`, `s.collect.<name>`, `s.probe.<name>`을 쓴다).
