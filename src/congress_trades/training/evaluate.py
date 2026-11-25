"""Student-vs-teacher agreement metrics (until a hand-labeled gold set exists)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Agreement:
    n: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    stance_match: int = 0
    direction_match: int = 0
    signal_tp: int = 0
    signal_fp: int = 0
    signal_fn: int = 0
    invalid: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def add(self, ref: dict[str, Any], pred: dict[str, Any] | None) -> None:
        self.n += 1
        if pred is None:
            self.invalid += 1
            pred = {}
        r = {str(t).upper() for t in ref.get("tickers", [])}
        p = {str(t).upper() for t in pred.get("tickers", [])}
        self.tp += len(r & p)
        self.fp += len(p - r)
        self.fn += len(r - p)
        self.stance_match += ref.get("stance") == pred.get("stance")
        self.direction_match += ref.get("direction") == pred.get("direction")
        rs, ps = bool(ref.get("policy_signal")), bool(pred.get("policy_signal"))
        self.signal_tp += rs and ps
        self.signal_fp += ps and not rs
        self.signal_fn += rs and not ps

    @staticmethod
    def _f1(tp: int, fp: int, fn: int) -> float:
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0

    def summary(self) -> dict[str, float]:
        lat = sorted(self.latencies_ms)
        return {
            "n": self.n,
            "ticker_f1": self._f1(self.tp, self.fp, self.fn),
            "ticker_precision": self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0,
            "ticker_recall": self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0,
            "stance_acc": self.stance_match / self.n if self.n else 0.0,
            "direction_acc": self.direction_match / self.n if self.n else 0.0,
            "signal_f1": self._f1(self.signal_tp, self.signal_fp, self.signal_fn),
            "valid_json": 1 - self.invalid / self.n if self.n else 0.0,
            "p50_ms": lat[len(lat) // 2] if lat else 0.0,
        }
