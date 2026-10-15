"""DX1 payload({"records": [...]})에서 숫자 값을 꺼내는 공용 도구."""
from __future__ import annotations

import json
import math
from typing import Iterator


def iter_records(payload: bytes) -> Iterator[dict]:
    try:
        data = json.loads(payload)
    except ValueError:
        return
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return
    for record in records:
        if isinstance(record, dict):
            yield record


def numeric_value(value):
    """숫자(불리언은 0/1)이면 그 값, 아니면 None. NaN과 무한대는 버린다."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None
