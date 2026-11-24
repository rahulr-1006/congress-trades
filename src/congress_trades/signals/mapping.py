"""Entity -> tradeable instrument mapping.

Not yet implemented. Planned: resolve company names to tickers via SEC's
``company_tickers.json``, and sectors to sector ETFs (e.g. Semiconductors -> SOXX).
"""

from __future__ import annotations


def resolve_ticker(name: str) -> str | None:
    raise NotImplementedError
