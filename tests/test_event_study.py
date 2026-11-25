from datetime import date, timedelta

import pytest

from congress_trades.backtest.event_study import Event, abnormal_return, run_event_study

DAYS = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
BENCH = {d: 100.0 * (1.01**i) for i, d in enumerate(DAYS)}  # +1%/day
STOCK = {d: 100.0 * (1.02**i) for i, d in enumerate(DAYS)}  # +2%/day


def test_entry_is_strictly_after_decision_date():
    # Decision on day 0 -> enter day 1 close; 1-day horizon = day1 -> day2.
    r = abnormal_return(STOCK, BENCH, DAYS[0], 1, +1)
    assert r == pytest.approx(0.02 - 0.01)


def test_no_lookahead_on_decision_day_price():
    # A huge jump on the decision day itself must not be captured.
    stock = dict(STOCK)
    stock[DAYS[5]] *= 10
    later = {d: (p if d <= DAYS[5] else p * 10) for d, p in stock.items()}
    r = abnormal_return(later, BENCH, DAYS[5], 1, +1)
    assert r == pytest.approx(0.01)


def test_short_direction_flips_sign():
    assert abnormal_return(STOCK, BENCH, DAYS[0], 5, -1) < 0


def test_out_of_range_returns_none():
    assert abnormal_return(STOCK, BENCH, DAYS[-2], 5, +1) is None


def test_run_event_study_aggregates_and_tracks_skips():
    events = [
        Event("UP", DAYS[0], +1),
        Event("UP", DAYS[3], -1),
        Event("GONE", DAYS[0], +1),
        Event("UP", DAYS[-1], +1),
    ]
    res = run_event_study(events, {"UP": STOCK}, BENCH, horizons=(1, 5))
    assert res.horizons[1].n == 2
    assert res.horizons[1].hit_rate == 0.5
    assert [e.ticker for e in res.skipped_no_prices] == ["GONE"]
    assert len(res.skipped_out_of_range) == 1
