from __future__ import annotations


class DropCounter:
    """버려진 항목의 누적 개수를 세고, 메시지마다 로그가 쏟아지지 않도록
    첫 발생과 이후 `log_every`건 단위로만 한 줄씩 출력한다."""

    def __init__(self, label: str, log_every: int = 100) -> None:
        self._label = label
        self._log_every = log_every
        self.total = 0

    def add(self, n: int = 1) -> None:
        before = self.total
        self.total += n
        if before == 0 or before // self._log_every != self.total // self._log_every:
            print(f"[{self._label}] 누적 {self.total}건")
