"""설정 마법사: 연결 정보를 저장하고, 수집한 데이터의 태그를 추천하고, 고른 태그로 설정 YAML을 만들어 검증한다."""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from ..config import ConfigError, EquipmentConfig, load_equipment_config
from ..deviation_suggest import _signal_name, analyze_tags  # 신호 이름 규칙을 CLI 추천과 똑같이 맞춘다
from ..replay import ReplayError
from .errors import StudioError

SETTINGS_FILE = "studio.json"
DEFAULT_MIN_SAMPLES = 3000      # 설정 파일 형식상 필요한 값. 이 화면은 수집본으로 학습하므로 쓰이지 않는다
NO_CONNECTION = "먼저 1단계에서 연결 정보를 저장하세요."
_EQUIPMENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_HOST = re.compile(r"^[A-Za-z0-9._-]+$")


def analyze(data_dir: str) -> dict:
    """데이터 폴더의 숫자 태그를 분석해 화면에 보여 줄 표를 만든다. 폴더는 읽기만 한다."""
    path = Path(str(data_dir).strip().strip('"').strip("'").strip())
    try:
        reports = analyze_tags(path)
    except (ReplayError, OSError, ValueError) as e:
        raise StudioError(str(e)) from e
    return {
        "tags": [
            {
                "tag": r.tag,
                "kind": r.kind,
                "value_range": r.value_range,
                "sigma": r.sigma,
                "limit": r.limit,
                "dynamic_range": r.dynamic_range,
                "moving_fraction": r.moving_fraction,
                "recommended": r.recommended,
                "reason": r.reason,
            }
            for r in reports
        ]
    }


# --- 1단계: 연결 정보 -------------------------------------------------------------------------


def _check_topic_filter(topic: str) -> None:
    """MQTT 토픽 필터 문법: '#'는 마지막 단계에 혼자, '+'는 한 단계 전체여야 한다."""
    levels = topic.split("/")
    bad = "\x00" in topic or len(topic.encode("utf-8")) > 65535
    for index, level in enumerate(levels):
        if "#" in level and (level != "#" or index != len(levels) - 1):
            bad = True
        if "+" in level and level != "+":
            bad = True
    if bad:
        raise StudioError(f"토픽 '{topic}'은 올바른 MQTT 토픽이 아닙니다('#'는 마지막에 혼자, '+'는 한 단계 전체여야 합니다).")


def _clean_connection(form: dict) -> dict:
    equipment_id = str(form.get("equipment_id", "")).strip()
    if not _EQUIPMENT_ID.match(equipment_id):
        raise StudioError("장비 이름은 영문, 숫자, _, - 만 쓸 수 있습니다(1~40자).")
    host = str(form.get("host", "")).strip()
    if not host:
        raise StudioError("브로커 주소를 입력하세요(예: localhost).")
    if not _HOST.match(host):
        raise StudioError("브로커 주소에는 포트나 공백을 넣을 수 없습니다. 포트는 아래 칸에 따로 적으세요.")
    try:
        port = int(form.get("port"))
    except (TypeError, ValueError):
        raise StudioError("포트는 1~65535 사이의 숫자여야 합니다.") from None
    if not 1 <= port <= 65535:
        raise StudioError("포트는 1~65535 사이의 숫자여야 합니다.")
    raw_topics = form.get("subscribe_topics") or []
    if not isinstance(raw_topics, list):
        raise StudioError("구독 토픽은 목록으로 보내야 합니다(쉼표로 구분해 적으세요).")
    topics = [str(t).strip() for t in raw_topics if str(t).strip()]
    if not topics:
        raise StudioError("구독할 토픽을 하나 이상 입력하세요(예: dx1/AxisData).")
    for topic in topics:
        _check_topic_filter(topic)
    return {"equipment_id": equipment_id, "host": host, "port": port, "topics": topics}


def save_connection(workspace, form: dict) -> dict:
    clean = _clean_connection(form)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / SETTINGS_FILE).write_text(json.dumps(clean, ensure_ascii=False), encoding="utf-8")
    return clean


def load_settings(workspace):
    path = Path(workspace) / SETTINGS_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return _clean_connection(
            {
                "equipment_id": data.get("equipment_id"),
                "host": data.get("host"),
                "port": data.get("port"),
                "subscribe_topics": data.get("topics"),
            }
        )
    except StudioError:
        return None


def saved_connection(workspace) -> dict:
    """저장된 연결 정보를 화면 입력칸 값으로 돌려준다. 아직 없으면 빈 딕셔너리."""
    settings = load_settings(workspace)
    if settings is None:
        return {}
    return {
        "equipment_id": settings["equipment_id"],
        "host": settings["host"],
        "port": settings["port"],
        "subscribe_topics": list(settings["topics"]),
    }


# --- 5단계: 고른 태그로 설정 만들기 -----------------------------------------------------------------


def _config_path(workspace, equipment_id: str) -> Path:
    return Path(workspace) / "configs" / f"{equipment_id}.yaml"


def _config_dict(settings: dict, tags: list) -> dict:
    used = set()
    signals = {_signal_name(tag, used): tag for tag in tags}
    eid = settings["equipment_id"]
    return {
        "equipment_id": eid,
        "mqtt": {
            "subscribe_topics": settings["topics"],
            "publish_topic": f"jetson/{eid}/anomaly",
            "command_topic": f"jetson/{eid}/cmd",
        },
        "deviation": {"signals": signals},
        "resample_interval_ms": 100,
        "max_lateness_ms": 500,
        "window_size": 30,
        "calibration": {"max_duration": "7d", "min_samples": DEFAULT_MIN_SAMPLES},
    }


def save_config(workspace, tags) -> EquipmentConfig:
    """저장된 연결 정보와 고른 태그로 설정 YAML을 만들어 기존 로더로 검증한 뒤에만 저장한다."""
    settings = load_settings(workspace)
    if settings is None:
        raise StudioError(NO_CONNECTION)
    chosen = []
    for tag in tags or []:
        tag = str(tag).strip()
        if tag and tag not in chosen:
            chosen.append(tag)
    if not chosen:
        raise StudioError("점수를 낼 태그를 하나 이상 선택하세요.")
    final = _config_path(workspace, settings["equipment_id"])
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.tmp")
    tmp.write_text(
        yaml.safe_dump(_config_dict(settings, chosen), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    try:
        config = load_equipment_config(tmp)
    except ConfigError as e:
        tmp.unlink()
        raise StudioError(f"설정이 올바르지 않습니다: {e}") from e
    tmp.replace(final)
    return config


def load_config(workspace):
    """연결 정보에 맞는 설정 YAML이 있으면 읽고, 없으면 None."""
    settings = load_settings(workspace)
    if settings is None:
        return None
    path = _config_path(workspace, settings["equipment_id"])
    if not path.exists():
        return None
    try:
        return load_equipment_config(path)
    except ConfigError as e:
        raise StudioError(f"저장된 설정을 읽지 못했습니다: {e}") from e


def saved_tags(workspace) -> list:
    config = load_config(workspace)
    if config is None or config.deviation is None:
        return []
    return [s.tag for s in config.deviation.signals]
