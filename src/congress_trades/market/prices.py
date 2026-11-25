"""Daily split/dividend-adjusted prices from Yahoo Finance's public chart endpoint (no key).

Prices are cached in SQLite, so a backtest only hits the network once per ticker.
"""

from __future__ import annotations

import json
import urllib.error
from datetime import date, datetime, timedelta, timezone

from .. import http

CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1={p1}&period2={p2}&interval=1d&events=div,split"
)

PriceSeries = dict[date, float]  # trading date -> adjusted close


def _yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")  # BRK.B -> BRK-B


def parse_chart(payload: bytes) -> PriceSeries:
    data = json.loads(payload)["chart"]
    if data.get("error") or not data.get("result"):
        return {}
    res = data["result"][0]
    offset = res["meta"].get("gmtoffset", 0)
    adj = (res.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    closes = adj or res["indicators"]["quote"][0]["close"]
    out: PriceSeries = {}
    for ts, px in zip(res.get("timestamp", []), closes, strict=False):
        if px is not None:
            out[datetime.fromtimestamp(ts + offset, tz=timezone.utc).date()] = float(px)
    return out


def fetch_prices(symbol: str, start: date, end: date) -> PriceSeries:
    def epoch(d: date) -> int:
        return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())

    url = CHART_URL.format(
        symbol=_yahoo_symbol(symbol), p1=epoch(start), p2=epoch(end + timedelta(days=1))
    )
    try:
        return parse_chart(http.get(url, min_interval=0.5))
    except urllib.error.HTTPError as e:
        if e.code in (400, 404):  # unknown/unsupported symbol
            return {}
        raise
