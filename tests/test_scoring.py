from datetime import date

from congress_trades.backtest.splits import split
from congress_trades.market.tickers import Listing, MentionIndex
from congress_trades.models import Hearing, Statement
from congress_trades.signals.scoring import llm_direction, llm_v0, mention_baseline

IDX = MentionIndex([Listing("NVDA", "NVIDIA Corporation - Common Stock", "Q"),
                    Listing("INTC", "Intel Corporation - Common Stock", "Q")])
H = Hearing("CHRG-1", "t", "Commerce", "HOUSE", 118, date(2023, 6, 1), date(2023, 9, 1))
ST = Statement("CHRG-1", 3, "Mr. X", "Nvidia shipped chips. Nvidia again. Intel too.")


def test_baseline_one_long_per_ticker_per_hearing():
    ev = mention_baseline([(H, ST), (H, ST)], IDX, "held")
    assert sorted(e.ticker for e in ev) == ["INTC", "NVDA"]
    assert all(e.direction == 1 and e.decision_date == date(2023, 6, 1) for e in ev)


def test_published_timing_uses_publish_date():
    assert mention_baseline([(H, ST)], IDX, "published")[0].decision_date == date(2023, 9, 1)


def test_llm_direction_rules():
    assert llm_direction({"signal": {"policy_signal": True, "direction": "negative"}}) == -1
    assert llm_direction({"signal": {"policy_signal": False}, "sentiment": {"score": 0.8}}) == 1
    assert llm_direction({"sentiment": {"score": 0.2}}) == 0


def test_llm_v0_drops_hallucinated_tickers():
    outputs = {("CHRG-1", 3): {
        "entities": {"tickers": ["NVDA", "MSFT"]},  # MSFT is not in the text
        "signal": {"policy_signal": True, "direction": "negative"},
    }}
    ev = llm_v0([(H, ST)], outputs, IDX, "held")
    assert [(e.ticker, e.direction) for e in ev] == [("NVDA", -1)]


def test_split_by_model_cutoff():
    old = H
    new = Hearing("CHRG-2", "t", "c", "HOUSE", 118, date(2024, 3, 1))
    dev, holdout = split([old, new])
    assert dev == [old] and holdout == [new]
