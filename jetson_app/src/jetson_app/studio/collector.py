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
