import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "src" / "jetson_app" / "web" / "studio.html"


def _html():
    return PAGE.read_text(encoding="utf-8")


def test_page_has_the_four_steps_and_main_controls():
    html = _html()

    for marker in ('data-step="1"', 'data-step="2"', 'data-step="3"', 'data-step="4"'):
        assert marker in html
    for control_id in ("btn-analyze", "btn-save", "btn-start", "btn-train", "btn-recalibrate", "btn-stop"):
        assert f'id="{control_id}"' in html


def test_page_is_self_contained():
    html = _html()

    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html)
    assert "innerHTML" not in html                       # 서버가 준 문자열은 textContent로만 넣는다
    assert "X-Studio-Token" in html and "/api/state" in html and "/dashboard" in html
