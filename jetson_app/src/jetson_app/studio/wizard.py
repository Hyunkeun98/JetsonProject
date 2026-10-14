"""설정 마법사: 정상 데이터를 훑어 태그를 추천하고, 선택한 태그로 설정 YAML을 만들어 검증한 뒤 저장한다."""
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
DEFAULT_MIN_SAMPLES = 3000
_EQUIPMENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_HOST = re.compile(r"^[A-Za-z0-9._-]+$")


def analyze(data_dir: str) -> dict:
    """정상 데이터 폴더의 숫자 태그를 분석해 화면에 보여 줄 표를 만든다. 폴더는 읽기만 한다."""
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


def _clean_form(form: dict) -> dict:
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
    topics = [str(t).strip() for t in form.get("subscribe_topics") or [] if str(t).strip()]
    if not topics:
        raise StudioError("구독할 토픽을 하나 이상 입력하세요(예: dx1/AxisData).")
    tags = []
    for tag in form.get("tags") or []:
        tag = str(tag).strip()
        if tag and tag not in tags:
            tags.append(tag)
    if not tags:
        raise StudioError("점수를 낼 태그를 하나 이상 선택하세요.")
    try:
        min_samples = int(form.get("min_samples", DEFAULT_MIN_SAMPLES))
    except (TypeError, ValueError):
        min_samples = 0
    if min_samples < 1:
        raise StudioError("학습에 필요한 수집 개수는 1 이상이어야 합니다.")
    return {
        "equipment_id": equipment_id,
        "host": host,
        "port": port,
        "topics": topics,
        "tags": tags,
        "min_samples": min_samples,
    }


def _config_dict(clean: dict) -> dict:
    used = set()
    signals = {_signal_name(tag, used): tag for tag in clean["tags"]}
    eid = clean["equipment_id"]
    return {
        "equipment_id": eid,
        "mqtt": {
            "subscribe_topics": clean["topics"],
            "publish_topic": f"jetson/{eid}/anomaly",
            "command_topic": f"jetson/{eid}/cmd",
        },
        "deviation": {"signals": signals},
        "resample_interval_ms": 100,
        "max_lateness_ms": 500,
        "window_size": 30,
        "calibration": {"max_duration": "7d", "min_samples": clean["min_samples"]},
    }


def save_config(workspace: Path, form: dict) -> EquipmentConfig:
    """입력을 검사하고 설정 YAML을 만들어 기존 설정 로더로 검증한 뒤에만 작업 폴더에 저장한다."""
    clean = _clean_form(form)
    config_dir = Path(workspace) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    name = f"{clean['equipment_id']}.yaml"
    final = config_dir / name
    tmp = config_dir / f".{name}.tmp"
    tmp.write_text(
        yaml.safe_dump(_config_dict(clean), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    try:
        config = load_equipment_config(tmp)
    except ConfigError as e:
        tmp.unlink()
        raise StudioError(f"설정이 올바르지 않습니다: {e}") from e
    tmp.replace(final)
    (Path(workspace) / SETTINGS_FILE).write_text(
        json.dumps({"config": name, "host": clean["host"], "port": clean["port"]}), encoding="utf-8"
    )
    return config


def load_settings(workspace: Path):
    path = Path(workspace) / SETTINGS_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if not isinstance(data, dict) or not {"config", "host", "port"} <= set(data):
        return None
    return data


def saved_form(workspace: Path) -> dict:
    """저장된 설정을 화면 입력칸 값으로 되돌린다. 아직 저장한 것이 없으면 빈 딕셔너리."""
    settings = load_settings(workspace)
    if settings is None:
        return {}
    try:
        config = load_equipment_config(Path(workspace) / "configs" / settings["config"])
    except ConfigError:
        return {}
    signals = config.deviation.signals if config.deviation else ()
    return {
        "equipment_id": config.equipment_id,
        "host": settings["host"],
        "port": settings["port"],
        "subscribe_topics": list(config.subscribe_topics),
        "tags": [s.tag for s in signals],
        "min_samples": config.calibration.min_samples,
    }
