from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse

from .dashboard_page import render_dashboard
from .score_history import ScoreHistory


class ScoreWebServer:
    """점수 이력을 읽기 전용으로 보여 주는 웹 서버(표준 라이브러리). 백그라운드 스레드에서 돈다."""

    def __init__(
        self,
        history: ScoreHistory,
        meta_fn: Callable[[], dict],
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        self._history = history
        self._meta_fn = meta_fn
        self._page = render_dashboard({"mode": "live"}).encode("utf-8")
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server._handle_get(self)

            def _method_not_allowed(self):
                self.send_response(405)
                self.send_header("Allow", "GET")
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_POST = do_PUT = do_DELETE = do_PATCH = _method_not_allowed

            def log_message(self, fmt, *args):  # 요청마다 콘솔을 어지럽히지 않는다
                pass

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self._thread = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join()

    def _handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        url = urlparse(handler.path)
        query = parse_qs(url.query)
        if url.path == "/":
            self._send(handler, 200, self._page, "text/html; charset=utf-8")
        elif url.path == "/healthz":
            self._send_json(handler, {"ok": True})
        elif url.path == "/api/meta":
            self._send_json(handler, self._meta_fn())
        elif url.path == "/api/scores":
            seconds = _number(query, "seconds", 300.0)
            max_points = _number(query, "max_points", 1200)
            self._send_json(handler, self._history.snapshot(seconds, int(max_points)))
        else:
            self._send(handler, 404, b"not found", "text/plain; charset=utf-8")

    def _send_json(self, handler, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(handler, 200, body, "application/json; charset=utf-8")

    @staticmethod
    def _send(handler, status: int, body: bytes, content_type: str) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.end_headers()
        handler.wfile.write(body)


def _number(query: dict, key: str, default: float) -> float:
    try:
        value = float(query[key][0])
    except (KeyError, IndexError, ValueError):
        return default
    if value != value or value in (float("inf"), float("-inf")):  # NaN, 무한대
        return default
    return value
