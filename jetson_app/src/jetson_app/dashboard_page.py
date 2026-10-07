from __future__ import annotations

import json
from pathlib import Path

_PAGE = Path(__file__).parent / "web" / "dashboard.html"
_TOKEN = "/*__BOOT__*/null"


def render_dashboard(boot: dict) -> str:
    """화면 HTML에 시작 정보(모드, 파일 모드의 데이터)를 JSON으로 넣어 돌려준다."""
    text = json.dumps(boot, ensure_ascii=False, separators=(",", ":"))
    # <script> 안에 넣으므로 </script> 등으로 문서가 끊기지 않게 한다.
    text = text.replace("</", "<\\/").replace("<!--", "<\\!--")
    html = _PAGE.read_text(encoding="utf-8")
    if _TOKEN not in html:
        raise RuntimeError("dashboard.html에 시작 정보 자리표시자가 없습니다")
    return html.replace(_TOKEN, text, 1)
