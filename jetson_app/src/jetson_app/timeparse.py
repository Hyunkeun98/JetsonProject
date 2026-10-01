from __future__ import annotations

import re
from datetime import datetime, timezone

_TS_PATTERN = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,9}))?"
    r"(Z|[+-]\d{2}:?\d{2})$"
)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def parse_dx1_timestamp(text: object) -> int | None:
    """DX1(SpeeDBee Synapse)이 내보내는 ISO 8601 타임스탬프를 epoch 나노초(int)로 바꾼다.

    DX1 형식(`2026-10-01T00:47:01.718651520+0000`: 나노초 9자리, 콜론 없는 오프셋)은
    Python 3.8의 `datetime.fromisoformat`이 읽지 못해 직접 파싱한다. 형식이 맞지 않거나
    존재하지 않는 날짜면 None을 반환한다(호출부가 해당 record를 버린다).
    """
    if not isinstance(text, str):
        return None
    match = _TS_PATTERN.match(text.strip())
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(g) for g in match.groups()[:6])
    fraction = match.group(7) or ""
    offset_text = match.group(8)
    try:
        moment = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None
    if offset_text == "Z":
        offset_seconds = 0
    else:
        digits = offset_text[1:].replace(":", "")
        offset_seconds = int(digits[:2]) * 3600 + int(digits[2:]) * 60
        if offset_text[0] == "-":
            offset_seconds = -offset_seconds
    delta = moment - _EPOCH
    epoch_seconds = delta.days * 86400 + delta.seconds - offset_seconds
    return epoch_seconds * 1_000_000_000 + int(fraction.ljust(9, "0"))


def format_epoch_ns(epoch_ns: int) -> str:
    """epoch 나노초를 Python 3.8의 `datetime.fromisoformat`이 읽을 수 있는
    UTC ISO 8601 문자열(마이크로초 정밀도, `+00:00`)로 바꾼다."""
    seconds, nanos = divmod(epoch_ns, 1_000_000_000)
    moment = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=nanos // 1000)
    return moment.isoformat()
