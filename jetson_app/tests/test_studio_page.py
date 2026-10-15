import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "src" / "jetson_app" / "web" / "studio.html"


def _html():
    return PAGE.read_text(encoding="utf-8")


def test_page_has_the_six_steps_and_main_controls():
    html = _html()

    for n in range(1, 7):
        assert f'data-step="{n}"' in html
    for control_id in (
        "btn-save-connection", "btn-probe-start", "btn-probe-stop", "btn-collect-start", "btn-collect-stop",
        "btn-discard", "btn-import", "btn-analyze", "btn-train", "btn-start", "btn-stop", "btn-retrain",
        "cycle-tag", "train-warning",
    ):
        assert f'id="{control_id}"' in html


def test_page_is_self_contained():
    html = _html()

    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html)
    assert "innerHTML" not in html                       # 서버가 준 문자열은 textContent로만 넣는다
    assert "X-Studio-Token" in html and "/api/state" in html and "/dashboard" in html


def test_buttons_follow_what_the_server_says_is_allowed():
    html = _html()
    render = html.split("function render(s)")[1].split("async function refresh")[0]

    assert "s.can" in render
    for api in ("/api/probe/start", "/api/collect/start", "/api/collect/import", "/api/analyze", "/api/train",
                "/api/start", "/api/retrain", "/api/connection"):
        assert api in html


def test_page_explains_the_limits_the_user_must_know():
    html = _html()

    assert "10" in html and "사이클" in html                # 최소 수집량 안내
    assert "축이 움직" in html                              # 멈춘 상태만 모으면 추천이 비는 이유
    assert "1GB" in html                                    # 수집 상한 도달 안내


def test_state_banner_is_cleared_when_its_condition_goes_away():
    html = _html()

    assert "function showBanner" in html and "stateMessage" in html
    assert 'banner("")' not in html.split("function render(s)")[1].split("async function refresh")[0]
