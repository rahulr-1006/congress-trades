"""Standard performance metrics over a series of periodic returns."""

from __future__ import annotations

import math
from collections.abc import Sequence


def total_return(returns: Sequence[float]) -> float:
    growth = 1.0
    for r in returns:
        growth *= 1 + r
    return growth - 1


def sharpe(returns: Sequence[float], periods_per_year: int = 252, rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = [r - rf / periods_per_year for r in returns]
    mean = sum(excess) / len(excess)
    var = sum((r - mean) ** 2 for r in excess) / (len(excess) - 1)
    return 0.0 if var == 0 else mean / math.sqrt(var) * math.sqrt(periods_per_year)


def max_drawdown(returns: Sequence[float]) -> float:
    """Largest peak-to-trough decline of the equity curve, as a positive fraction."""
    equity, peak, worst = 1.0, 1.0, 0.0
    for r in returns:
        equity *= 1 + r
        peak = max(peak, equity)
        worst = max(worst, 1 - equity / peak)
    return worst


def hit_rate(trade_returns: Sequence[float]) -> float:
    return sum(r > 0 for r in trade_returns) / len(trade_returns) if trade_returns else 0.0
