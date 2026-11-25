"""Cache-aware price loading: only hit the network for ranges not already in SQLite."""

from __future__ import annotations

from datetime import date

from ..store import Store
from .prices import PriceSeries, fetch_prices


def load_prices(store: Store, symbol: str, start: date, end: date) -> PriceSeries:
    cov = store.price_coverage(symbol)
    if not cov or start < cov[0] or end > cov[1]:
        lo = min(start, cov[0]) if cov else start
        hi = max(end, cov[1]) if cov else end
        store.save_prices(symbol, fetch_prices(symbol, lo, hi), lo, hi)
    return {d: p for d, p in store.prices(symbol).items() if start <= d <= end}
