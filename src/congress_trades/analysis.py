"""Analytics over stored filings."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from .models import Filing


@dataclass(frozen=True)
class FilerStats:
    name: str
    state_district: str
    filings: int
    last_filed: date


def most_active(filings: Iterable[Filing], limit: int = 10) -> list[FilerStats]:
    """Rank members by number of PTRs filed (a proxy for trading activity)."""
    counts: Counter[tuple[str, str]] = Counter()
    latest: dict[tuple[str, str], date] = {}
    for f in filings:
        key = (f.name, f.state_district)
        counts[key] += 1
        if key not in latest or f.filing_date > latest[key]:
            latest[key] = f.filing_date
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -latest[kv[0]].toordinal(), kv[0]))
    return [
        FilerStats(name=n, state_district=sd, filings=c, last_filed=latest[(n, sd)])
        for (n, sd), c in ranked[:limit]
    ]
