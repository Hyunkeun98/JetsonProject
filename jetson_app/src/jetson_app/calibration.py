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
                data = json.loads(line)
                samples.append(
                    CalibrationSample(timestamp=data["timestamp"], values=data["values"])
                )
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
            kept = [
                s
                for s in self._read_all_locked()
                if datetime.fromisoformat(s.timestamp) >= cutoff
            ]
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
