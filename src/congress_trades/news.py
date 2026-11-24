"""News ingestion (skeleton).

Planned: pull recent headlines per ticker / per member from RSS and news APIs so the
LLM can weigh a trade against what was public at the time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str
    published: datetime
    url: str
    tickers: tuple[str, ...] = ()


def fetch_news(ticker: str, limit: int = 20) -> list[NewsItem]:
    """Return recent headlines mentioning ``ticker``."""
    raise NotImplementedError("News ingestion is on the roadmap.")
