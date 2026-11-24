"""Point-in-time backtester (not yet implemented).

Requirements:
- Replays hearings in chronological order through the *same* pipeline used live
  (ingest -> llm tasks -> scoring), so there is no separate "backtest-only" code path.
- A signal from a hearing held on day T may only trade at T's close at the earliest
  (or T+1 open if the transcript was published later): no lookahead.
- Uses only prices available at decision time; outputs metrics from ``metrics.py``.
"""

from __future__ import annotations

from datetime import date


def run_backtest(start: date, end: date) -> dict[str, float]:
    raise NotImplementedError
