from __future__ import annotations

import math
import threading
from collections import deque
from datetime import datetime

from .publisher import GroupOutput

DEFAULT_HISTORY_SECONDS = 3600
MIN_SECONDS = 10
MIN_POINTS = 50
MAX_POINTS = 5000


class ScoreHistory:
    """최근 점수를 메모리에 쌓는 링 버퍼(웹 화면용). 시각은 DX1 이벤트 시각(epoch ms)이다.
    행: (t_ms, {그룹: (점수, 알람, 원인 태그)}). 재시작하면 사라진다."""

    def __init__(self, capacity: int, group_names: tuple[str, ...]) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._rows: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.group_names = tuple(group_names)

    @property
    def capacity(self) -> int:
        return self._rows.maxlen

    def record(self, t_ms: int, groups: dict) -> None:
        row = (t_ms, {n: (g.score, bool(g.alarm), g.top_tag) for n, g in groups.items()})
        with self._lock:
            self._rows.append(row)

    def snapshot(self, seconds: float, max_points: int) -> dict:
        """최근 seconds초를 최대 max_points점으로 줄여 돌려준다. 구간마다 그룹별 최댓값 점수를 남겨
        짧은 튐이 사라지지 않게 하고, 구간 안에 알람이 있었으면 알람으로 표시한다."""
        seconds = max(MIN_SECONDS, seconds)
        max_points = min(MAX_POINTS, max(MIN_POINTS, int(max_points)))
        with self._lock:
            rows = list(self._rows)
        if not rows:
            return {"now_ms": None, "bucket_ms": 0, "points": []}
        now_ms = rows[-1][0]
        start_ms = now_ms - int(seconds * 1000)
        rows = [r for r in rows if r[0] >= start_ms]
        if len(rows) <= max_points:
            points = [
                {"t": t, "g": {n: [s, a, top] for n, (s, a, top) in g.items()}} for t, g in rows
            ]
            return {"now_ms": now_ms, "bucket_ms": 0, "points": points}
        bucket_ms = max(1, math.ceil(seconds * 1000 / max_points))
        points = []
        current_bucket = None
        point = None
        for t, groups in rows:
            bucket = (t - start_ms) // bucket_ms
            if bucket != current_bucket:
                current_bucket = bucket
                point = {"t": t, "g": {}}
                points.append(point)
            merged = point["g"]
            for name, (score, alarm, top) in groups.items():
                best = merged.get(name)
                if best is None:
                    merged[name] = [score, alarm, top]
                else:
                    if score > best[0]:
                        best[0], best[2] = score, top
                    best[1] = best[1] or alarm
        return {"now_ms": now_ms, "bucket_ms": bucket_ms, "points": points}


class HistoryRecordingPublisher:
    """결과 발행기를 감싸 같은 결과를 이력에도 기록한다. 원래 발행을 먼저 하고, 이력 기록의
    예외는 삼켜서 점수/발행 경로에 영향을 주지 않는다."""

    def __init__(self, inner, history: ScoreHistory) -> None:
        self._inner = inner
        self._history = history

    def publish(self, timestamp, anomaly_score, alarm, top_deviant_tag, groups=None) -> None:
        self._inner.publish(timestamp, anomaly_score, alarm, top_deviant_tag, groups)
        try:
            outputs = groups or {
                "all": GroupOutput(score=anomaly_score, alarm=alarm, top_tag=top_deviant_tag)
            }
            t_ms = int(round(datetime.fromisoformat(timestamp).timestamp() * 1000))
            self._history.record(t_ms, outputs)
        except Exception as e:  # 화면용 기록이 점수 경로를 막으면 안 된다
            print(f"[ScoreHistory] 이력 기록 실패(무시): {e}")
