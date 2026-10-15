from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import quote

from .controller import StudioController
from .server import DEFAULT_PORT, LOOPBACK_HOSTS, StudioServer


def default_workspace() -> Path:
    """exe로 실행하면 exe 옆, 그 밖에는 현재 폴더 아래의 workspace."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "workspace"
    return Path.cwd() / "workspace"


def control_enabled(web_host: str, token) -> bool:
    """루프백이 아닌 주소로 열 때는 토큰이 있어야 조작 기능을 켠다."""
    return web_host in LOOPBACK_HOSTS or bool(token)


def studio_url(web_host: str, port: int, token) -> str:
    shown_host = "localhost" if web_host in LOOPBACK_HOSTS else web_host
    url = f"http://{shown_host}:{port}/"
    return url + (f"?token={quote(token, safe='')}" if token else "")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="jetson-studio: 설정, 수집, 학습, 감시를 한 화면에서")
    parser.add_argument("--workspace", help="설정, 수집 데이터, 모델을 저장할 폴더(기본: exe 옆 또는 ./workspace)")
    parser.add_argument("--web-host", default="127.0.0.1", help="화면을 열 주소(기본 127.0.0.1, 이 장비에서만)")
    parser.add_argument("--web-port", type=int, default=DEFAULT_PORT, help="화면 포트(기본 8800)")
    parser.add_argument("--token", help="원격(루프백이 아닌 주소)에서 조작하려면 필요한 비밀번호")
    parser.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않는다")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace) if args.workspace else default_workspace()
    enabled = control_enabled(args.web_host, args.token)
    try:
        controller = StudioController(workspace)
        server = StudioServer(
            controller, host=args.web_host, port=args.web_port, token=args.token, control_enabled=enabled
        )
    except OSError as e:
        print(f"error: {args.web_host}:{args.web_port} 또는 작업 폴더를 열 수 없습니다: {e}", file=sys.stderr)
        raise SystemExit(2)

    server.start()
    url = studio_url(args.web_host, server.port, args.token)
    print(f"[studio] 화면: {url}")
    print(f"[studio] 작업 폴더: {workspace}")
    if not enabled:
        print(
            "[studio] 경고: 루프백이 아닌 주소인데 --token이 없어 조작 기능을 껐습니다(읽기 전용).",
            file=sys.stderr,
        )
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        try:
            controller.stop()
        except Exception as e:
            print(f"[studio] 종료 정리 중 오류(무시): {e}", file=sys.stderr)
        server.stop()


if __name__ == "__main__":
    main()
