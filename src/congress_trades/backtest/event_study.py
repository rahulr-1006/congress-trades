"""Event study: did the stock beat the benchmark after a signal?

For each event (ticker, decision date, direction):
  entry  = adjusted close on the first trading day *strictly after* the decision date.
           Hearings run during market hours and transcript publish times aren't known to the
           minute, so trading same-day would risk lookahead. This is deliberately conservative.
  CAR(h) = direction * [(P[entry+h] / P[entry] - 1) - (B[entry+h] / B[entry] - 1)]
where B is the benchmark (SPY by default). Positive CAR = the signal was right.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date

from ..market.prices import PriceSeries

HORIZONS = (1, 5, 20)


@dataclass(frozen=True)
class Event:
    ticker: str
    decision_date: date
    direction: int  # +1 long, -1 short
    source: str = ""  # e.g. "CHRG-118hhrg12345#42"


@dataclass
class HorizonStats:
    horizon: int
    returns: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.returns)

    @property
    def mean(self) -> float:
        return sum(self.returns) / self.n if self.n else 0.0

    @property
    def t_stat(self) -> float:
        if self.n < 2:
            return 0.0
        var = sum((r - self.mean) ** 2 for r in self.returns) / (self.n - 1)
        return 0.0 if var == 0 else self.mean / math.sqrt(var / self.n)

    @property
    def hit_rate(self) -> float:
        return sum(r > 0 for r in self.returns) / self.n if self.n else 0.0


@dataclass
class StudyResult:
    horizons: dict[int, HorizonStats]
    skipped_no_prices: list[Event]
    skipped_out_of_range: list[Event]


def abnormal_return(
    px: PriceSeries, bench: PriceSeries, decision: date, horizon: int, direction: int
) -> float | None:
    days = sorted(d for d in px if d in bench)
    i = bisect.bisect_right(days, decision)  # first trading day strictly after decision
    if i + horizon >= len(days):
        return None
    d0, d1 = days[i], days[i + horizon]
    stock = px[d1] / px[d0] - 1
    market = bench[d1] / bench[d0] - 1
    return direction * (stock - market)


def run_event_study(
    events: Iterable[Event],
    prices: Mapping[str, PriceSeries],
    benchmark: PriceSeries,
    horizons: tuple[int, ...] = HORIZONS,
) -> StudyResult:
    result = StudyResult({h: HorizonStats(h) for h in horizons}, [], [])
    for ev in events:
        px = prices.get(ev.ticker)
        if not px:
            result.skipped_no_prices.append(ev)
            continue
        any_ok = False
        for h in horizons:
            r = abnormal_return(px, benchmark, ev.decision_date, h, ev.direction)
            if r is not None:
                result.horizons[h].returns.append(r)
                any_ok = True
        if not any_ok:
            result.skipped_out_of_range.append(ev)
    return result
