from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, List

from .buffer import Snapshot

_PRUNE_CHECK_INTERVAL = 10_000  # 대략 50ms 틱 기준 ~8분마다 오래된 데이터 정리 체크


class CalibrationError(ValueError):
    pass


class CalibrationState(Enum):
    CALIBRATING = "CALIBRATING"
    MONITORING = "MONITORING"


class StateStore:
    """CALIBRATING/MONITORING 상태를 파일로 영속화해, Jetson 재시작 시 이어서
    복구할 수 있게 한다. 모델 파일과 별도로 관리한다 — recalibrate는 모델 파일은
    남겨두고 이 마커만 CALIBRATING으로 되돌리므로, 재시작 시 반드시 이 마커를
    기준으로 판단해야 한다(모델 파일이 있다고 바로 MONITORING으로 재개하면 안 된다)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def read(self) -> CalibrationState:
        if not self._path.exists():
            return CalibrationState.CALIBRATING
        text = self._path.read_text(encoding="utf-8").strip()
        try:
            return CalibrationState(text)
        except ValueError:
            return CalibrationState.CALIBRATING

    def write(self, state: CalibrationState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(state.value, encoding="utf-8")


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

    def _read_all_locked(self) -> list[CalibrationSample]:
        """Read all samples from file. Caller must already hold self._lock."""
        if not self._path.exists():
            return []
        samples = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    samples.append(
                        CalibrationSample(timestamp=data["timestamp"], values=data["values"])
                    )
                except (ValueError, KeyError, TypeError) as e:
                    print(f"[CalibrationBufferWriter] 손상된 캘리브레이션 레코드 무시: {e}")
        return samples

    def read_all(self) -> list[CalibrationSample]:
        with self._lock:
            return self._read_all_locked()

    def count(self) -> int:
        with self._lock:
            return len(self._read_all_locked())

    def clear(self) -> None:
        with self._lock:
            if self._path.exists():
                self._path.unlink()

    def prune_older_than(self, cutoff: datetime) -> None:
        with self._lock:
            all_samples = self._read_all_locked()
            kept = [
                s for s in all_samples if datetime.fromisoformat(s.timestamp) >= cutoff
            ]
            if len(kept) == len(all_samples):
                return  # 만료된 샘플이 없으면 파일 재작성 자체를 건너뛴다
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                for s in kept:
                    f.write(json.dumps({"timestamp": s.timestamp, "values": s.values}) + "\n")
            tmp_path.replace(self._path)


TrainFn = Callable[[List[CalibrationSample]], None]


class CalibrationManager:
    """CALIBRATING/MONITORING 상태 전이와 캘리브레이션 데이터 수집을 담당."""

    def __init__(
        self,
        buffer_writer: CalibrationBufferWriter,
        min_samples: int,
        max_duration: timedelta,
        train_fn: TrainFn,
        state_store: StateStore,
    ) -> None:
        self._buffer_writer = buffer_writer
        self._min_samples = min_samples
        self._max_duration = max_duration
        self._train_fn = train_fn
        self._state_store = state_store
        self._lock = threading.Lock()
        self._tick_count = 0
        self.state = state_store.read()

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
            self._state_store.write(self.state)

    def handle_recalibrate_command(self) -> None:
        with self._lock:
            self._buffer_writer.clear()
            self.state = CalibrationState.CALIBRATING
            self._state_store.write(self.state)
