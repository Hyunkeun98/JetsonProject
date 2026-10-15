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
SLOW_REPORT_SECONDS = 30.0          # 정확한 사이클 신호가 없을 때의 주기 추정은 무거워서 드물게만 다시 계산한다
CAPTURE_FILE = "capture.jsonl"
META_FILE = "capture.meta"          # .json이 아니라서 데이터 폴더를 읽는 도구가 건너뛴다
DATA_SUFFIXES = (".json", ".jsonl")
NO_EQUIPMENT = "먼저 1단계에서 연결 정보를 저장하세요."


def _empty_state(limit_bytes: int) -> dict:
    return {
        "active": False, "has_data": False, "connected": False, "messages": 0, "bytes": 0,
        "limit_bytes": limit_bytes, "seconds": 0.0, "no_data": False, "interrupted": False,
        "ended_reason": None, "origin": None, "cycles": None, "counting": False,
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
        self._cond = threading.Condition(self._lock)    # 백그라운드 훑기가 끝나길 기다릴 때 쓴다
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
        self._generation = 0            # 수집본이 바뀔 때마다 늘려서, 늦게 끝난 옛 훑기 결과를 버린다
        self._scanning = False
        self._scan_cancel = None
        self._warned_feed = False

    # --- 준비 ---------------------------------------------------------------------------

    def set_equipment(self, equipment_id) -> None:
        with self._lock:
            if self._active:
                raise StudioError("수집 중에는 장비를 바꿀 수 없습니다. 먼저 수집을 종료하세요.")
            if equipment_id == self._equipment_id:
                return                  # 같은 장비를 다시 저장해도 이미 센 사이클 수를 버리지 않는다
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
        self._generation += 1

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
        opened = None
        try:
            with self._lock:
                self._wait_scan_locked()
                self._archive_current_locked()
                current = self.current_dir
                current.mkdir(parents=True, exist_ok=True)
                opened = open(current / CAPTURE_FILE, "ab")
                self._write_meta_locked(
                    host=host, port=port, topics=list(topics), started_at=_now_text(), stopped_at=None,
                    reason=None, messages=0, bytes=0, seconds=0.0,
                )
                self._file = opened
                self._messages = 0
                self._bytes = 0
                self._started = self._clock()
                self._last_rx = None
                self._ended_reason = None
                self._counter = CycleCounter()
                self._report = None
                self._generation += 1
                self._warned_feed = False
                self._source = source
                self._active = True            # 모든 준비가 끝난 뒤에 켠다(중간에 실패해도 반쯤 켜진 상태가 없다)
        except Exception as e:
            if opened is not None:
                try:
                    opened.close()
                except OSError:
                    pass
            self._disconnect(source)           # 연결만 남아 계속 구독하는 일이 없게 한다
            if isinstance(e, StudioError):
                raise
            raise StudioError(f"수집 파일을 만들지 못했습니다: {e}") from e

    def _on_payload(self, topic: str, payload: bytes) -> None:
        reason = None
        with self._lock:
            if not self._active:
                return
            line = payload.replace(b"\r", b" ").replace(b"\n", b" ").strip()
            if not line:
                return
            try:
                self._file.write(line + b"\n")
                self._file.flush()
            except OSError as e:
                print(f"[studio] 수집 파일을 쓰지 못해 수집을 멈춥니다: {e!r}")
                reason = "error"
            else:
                self._bytes += len(line) + 1
                self._messages += 1
                self._last_rx = self._clock()
                try:
                    self._counter.feed_payload(line)
                except Exception as e:           # 사이클 세기가 실패해도 원본은 이미 저장했다
                    if not self._warned_feed:
                        self._warned_feed = True
                        print(f"[studio] 사이클 세기에서 읽을 수 없는 payload를 건너뜁니다: {e!r}")
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
            except OSError as e:
                print(f"[studio] 수집 파일을 닫는 중 오류(무시): {e!r}")
            finally:
                self._file = None
            try:
                self._write_meta_locked(
                    stopped_at=_now_text(), reason=reason, messages=self._messages, bytes=self._bytes,
                    seconds=round(self._clock() - self._started, 1),
                )
            except OSError as e:
                print(f"[studio] 수집 정보를 저장하지 못했습니다(무시): {e!r}")
            self._report = None
        self._disconnect(source)

    @staticmethod
    def _disconnect(source) -> None:
        if source is None:
            return
        try:
            source.disconnect()
        except Exception as e:
            print(f"[studio] 수집 연결을 닫는 중 오류(무시): {e!r}")

    def discard(self) -> None:
        with self._lock:
            if self._active:
                raise StudioError("수집 중에는 지울 수 없습니다. 먼저 수집을 종료하세요.")
            self._wait_scan_locked()
            try:
                shutil.rmtree(self.current_dir)
            except FileNotFoundError:
                pass
            except OSError as e:
                raise StudioError(
                    f"수집한 데이터를 모두 지우지 못했습니다. 다른 프로그램이 파일을 쓰고 있는지 확인하세요. ({e})"
                ) from e
            self._forget_locked()

    def import_folder(self, folder) -> None:
        """저장해 둔 데이터 폴더의 *.json/*.jsonl을 수집본으로 복사해 온다(원본 폴더는 읽기만 한다)."""
        path = Path(str(folder).strip().strip('"').strip("'").strip())
        if not path.is_dir():
            raise StudioError(f"데이터 폴더를 찾을 수 없습니다: {path}")
        resolved = path.resolve()
        collections = self._root.resolve()
        if resolved == collections or collections in resolved.parents:
            raise StudioError("작업 폴더 안의 수집 데이터는 불러올 수 없습니다. 다른 폴더를 지정하세요.")
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in DATA_SUFFIXES)
        if not files:
            raise StudioError(f"데이터 폴더에 .json/.jsonl 파일이 없습니다: {path}")
        with self._lock:
            if self._equipment_id is None:
                raise StudioError(NO_EQUIPMENT)
            if self._active:
                raise StudioError("수집 중에는 불러올 수 없습니다. 먼저 수집을 종료하세요.")
            self._wait_scan_locked()
            self._archive_current_locked()
            current = self.current_dir
            current.mkdir(parents=True, exist_ok=True)
            total = 0
            for source in files:
                shutil.copy2(source, current / (source.stem + source.suffix.lower()))   # 분석 도구는 소문자 확장자만 읽는다
                total += source.stat().st_size
            self._forget_locked()               # 사이클 수는 백그라운드에서 센다
            self._ended_reason = "import"
            self._write_meta_locked(
                origin=str(path), started_at=_now_text(), stopped_at=_now_text(), reason="import",
                messages=None, bytes=total, seconds=0.0,
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
                report = self._report_locked(now)
                if report is None:
                    info["counting"] = True
                else:
                    info["cycles"] = report.to_dict()
                if not active and not info["seconds"] and self._counter is not None:
                    info["seconds"] = round(self._counter.duration_seconds, 1)
            return info

    def cycle_report(self, preferred_tag=None):
        """수집본의 사이클 수. 아직 안 셌으면 여기서 센다(오래 걸릴 수 있어 락을 잡지 않는다)."""
        with self._lock:
            counter = self._counter
            generation = self._generation
            directory = self.current_dir if counter is None else None
        if counter is None:
            counter = scan_directory(directory) if directory.exists() else CycleCounter()
            with self._lock:
                if self._counter is None and generation == self._generation:
                    self._counter = counter
                    self._report = None
                elif self._counter is not None:
                    counter = self._counter
        return counter.report(preferred_tag)

    # --- 내부 ---------------------------------------------------------------------------

    def _report_locked(self, now: float):
        """캐시된 사이클 보고서. 아직 센 적이 없으면 백그라운드에서 세기 시작하고 None."""
        if self._counter is None:
            self._start_scan_locked()
            return None
        slow = self._report is not None and self._report.method not in ("edge", "counter")
        interval = max(self._report_interval, SLOW_REPORT_SECONDS) if slow else self._report_interval
        if self._report is None or (self._active and now - self._report_at >= interval):
            self._report = self._counter.report()
            self._report_at = now
        return self._report

    def _start_scan_locked(self) -> None:
        if self._scanning:
            return
        self._scanning = True
        cancel = self._scan_cancel = threading.Event()
        threading.Thread(
            target=self._scan_in_background, args=(self.current_dir, self._generation, cancel), daemon=True
        ).start()

    def _wait_scan_locked(self) -> None:
        """파일을 지우거나 옮기기 전에 백그라운드 훑기를 멈추고 파일을 놓을 때까지 기다린다.
        Windows에서는 열려 있는 파일을 지울 수 없다. 기다리는 동안 락은 풀려서 훑기가 끝낼 수 있다."""
        while self._scanning:
            if self._scan_cancel is not None:
                self._scan_cancel.set()
            self._cond.wait(timeout=1.0)

    def _scan_in_background(self, directory: Path, generation: int, cancel: threading.Event) -> None:
        try:
            counter = scan_directory(directory, cancelled=cancel.is_set)
        except Exception as e:
            print(f"[studio] 사이클 세기를 하지 못했습니다: {e!r}")
            counter = CycleCounter()
        with self._lock:
            self._scanning = False
            if not cancel.is_set() and self._counter is None and generation == self._generation:
                self._counter = counter
                self._report = None
            self._cond.notify_all()

    def _data_files_locked(self) -> list:
        current = self.current_dir
        if not current.exists():
            return []
        return sorted(p for p in current.iterdir() if p.suffix in DATA_SUFFIXES and p.is_file())

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
        try:
            current.rename(destination)         # 폴더째 한 번에 옮긴다: 실패하면 반쯤 옮겨진 상태가 남지 않는다
        except OSError as e:
            raise StudioError(
                f"이전 수집본을 보관 폴더로 옮기지 못했습니다. 다른 프로그램(분석 중이거나 파일을 연 프로그램)이 "
                f"파일을 쓰고 있는지 확인하세요. ({e})"
            ) from e


def _now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
