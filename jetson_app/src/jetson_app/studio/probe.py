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
