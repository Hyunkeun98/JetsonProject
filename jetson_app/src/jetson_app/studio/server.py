"""jetson-studio의 화면과 JSON API를 내보내는 서버(표준 라이브러리, 백그라운드 스레드)."""
from __future__ import annotations

import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..dashboard_page import render_dashboard
from .controller import StudioController
from .errors import StudioError

MAX_BODY_BYTES = 64 * 1024
DEFAULT_PORT = 8800  # 8080은 다른 프로그램과 자주 겹쳐서 피한다
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
_STUDIO_PAGE = Path(__file__).resolve().parent.parent / "web" / "studio.html"


class _Httpd(ThreadingHTTPServer):
    # Windows에서는 SO_REUSEADDR이 이미 쓰는 포트에 또 바인딩되게 해서, 앱을 두 번 켜도 오류가 나지 않고
    # 브라우저가 어느 쪽에 붙을지 알 수 없게 된다. Windows에서만 끈다(Linux는 재시작 직후 바인딩을 위해 켠다).
    allow_reuse_address = os.name != "nt"


class StudioServer:
    def __init__(
        self,
        controller: StudioController,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        token=None,
        control_enabled: bool = True,
    ) -> None:
        self._controller = controller
        self._token = token or None
        self._control_enabled = control_enabled
        self._loopback = host in LOOPBACK_HOSTS
        self._studio_page = _STUDIO_PAGE.read_bytes()
        self._dashboard_page = render_dashboard({"mode": "live"}).encode("utf-8")
        self._post_routes = {
            "/api/analyze": lambda body: controller.analyze(str(body.get("data_dir", ""))),
            "/api/config": lambda body: controller.save_config(body),
            "/api/start": lambda body: controller.start(),
            "/api/train": lambda body: controller.train(),
            "/api/recalibrate": lambda body: controller.recalibrate(),
            "/api/stop": lambda body: controller.stop(),
        }
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server._get(self)

            def do_POST(self):
                server._post(self)

            def _unsupported(self):
                server._send_json(self, 405, {"error": "지원하지 않는 요청입니다"})

            do_PUT = do_DELETE = do_PATCH = _unsupported

            def log_message(self, fmt, *args):
                pass

        self._httpd = _Httpd((host, port), Handler)
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

    # --- GET ----------------------------------------------------------------------------

    def _get(self, h) -> None:
        url = urlparse(h.path)
        query = parse_qs(url.query)
        c = self._controller
        if url.path == "/":
            self._send(h, 200, self._studio_page, "text/html; charset=utf-8")
        elif url.path == "/dashboard":
            self._send(h, 200, self._dashboard_page, "text/html; charset=utf-8")
        elif url.path == "/healthz":
            self._send_json(h, 200, {"ok": True})
        elif url.path == "/api/state":
            state = c.state()
            state["control_enabled"] = self._control_enabled
            state["token_required"] = self._token is not None
            self._send_json(h, 200, state)
        elif url.path == "/api/config":
            self._send_json(h, 200, c.form())
        elif url.path == "/api/meta":
            self._send_json(h, 200, c.meta())
        elif url.path == "/api/scores":
            self._send_json(
                h, 200, c.scores(_number(query, "seconds", 300.0), int(_number(query, "max_points", 1200)))
            )
        else:
            self._send(h, 404, b"not found", "text/plain; charset=utf-8")

    # --- POST ---------------------------------------------------------------------------

    def _post(self, h) -> None:
        if not self._control_enabled:
            return self._send_json(
                h, 403, {"error": "조작 기능이 꺼져 있습니다. 다른 PC에서 조작하려면 --token을 지정해 다시 시작하세요."}
            )
        if not self._origin_ok(h) or not self._token_ok(h):
            return self._send_json(h, 403, {"error": "허용되지 않은 요청입니다(토큰 또는 출처 확인 실패)."})
        route = self._post_routes.get(urlparse(h.path).path)
        if route is None:
            return self._send_json(h, 404, {"error": "없는 경로입니다"})
        try:
            length = int(h.headers.get("Content-Length") or 0)
        except ValueError:
            return self._send_json(h, 400, {"error": "Content-Length가 올바르지 않습니다"})
        if length < 0:
            return self._send_json(h, 400, {"error": "Content-Length가 올바르지 않습니다"})
        if length > MAX_BODY_BYTES:
            return self._send_json(h, 413, {"error": "요청이 너무 큽니다"})
        raw = h.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            return self._send_json(h, 400, {"error": "JSON 형식이 올바르지 않습니다"})
        if not isinstance(body, dict):
            return self._send_json(h, 400, {"error": "JSON 객체를 보내야 합니다"})
        try:
            result = route(body)
        except StudioError as e:
            return self._send_json(h, 400, {"error": str(e)})
        except Exception as e:
            print(f"[studio] 요청 처리 중 오류: {e!r}")
            return self._send_json(h, 500, {"error": "내부 오류가 났습니다. 콘솔 로그를 확인하세요."})
        self._send_json(h, 200, result if isinstance(result, dict) else {"ok": True})

    def _origin_ok(self, h) -> bool:
        host = h.headers.get("Host", "")
        hostname = urlparse("//" + host).hostname
        if self._loopback and hostname not in LOOPBACK_HOSTS:
            return False              # DNS rebinding 방어: 루프백 서버는 루프백 이름으로 온 요청만 받는다
        origin = h.headers.get("Origin")
        if origin is not None and urlparse(origin).netloc != host:
            return False              # 다른 사이트의 페이지에서 보낸 요청
        return True

    def _token_ok(self, h) -> bool:
        if self._token is None:
            return True
        supplied = h.headers.get("X-Studio-Token", "")
        return hmac.compare_digest(supplied.encode("utf-8"), self._token.encode("utf-8"))

    # --- 응답 ---------------------------------------------------------------------------

    def _send_json(self, h, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(h, status, body, "application/json; charset=utf-8")

    @staticmethod
    def _send(h, status: int, body: bytes, content_type: str) -> None:
        h.send_response(status)
        h.send_header("Content-Type", content_type)
        h.send_header("Content-Length", str(len(body)))
        h.send_header("Cache-Control", "no-store")
        h.send_header("X-Content-Type-Options", "nosniff")
        h.end_headers()
        h.wfile.write(body)


def _number(query: dict, key: str, default: float) -> float:
    try:
        value = float(query[key][0])
    except (KeyError, IndexError, ValueError):
        return default
    if value != value or value in (float("inf"), float("-inf")):
        return default
    return value
