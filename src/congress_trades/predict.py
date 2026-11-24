"""Prediction layer (skeleton).

Planned: combine quantitative features (filing frequency, disclosure lag, trade size,
committee overlap, price momentum) with the LLM's qualitative read into a per-ticker signal,
then evaluate it with a walk-forward backtest against SPY.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class Signal:
    ticker: str
    as_of: date
    direction: Literal["long", "short", "neutral"]
    score: float
    rationale: str


def generate_signals(as_of: date) -> list[Signal]:
    raise NotImplementedError("Signal generation is on the roadmap.")
