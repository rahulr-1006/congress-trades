import json
from datetime import date, datetime, timezone
from pathlib import Path

from congress_trades.market.prices import parse_chart
from congress_trades.market.tickers import MentionIndex, clean_name, parse_symbol_file

LISTINGS = parse_symbol_file(
    (Path(__file__).parent / "fixtures" / "nasdaqlisted.txt").read_text(), "Symbol", "Q"
)


def test_symbol_file_drops_etfs_warrants_and_test_issues():
    assert [x.symbol for x in LISTINGS] == ["NVDA", "INTC", "MU", "GOOGL", "GOOG", "TGT"]


def test_clean_name():
    assert clean_name("NVIDIA Corporation - Common Stock") == "nvidia"
    assert clean_name("Micron Technology, Inc. - Common Stock") == "micron technology"


def test_mentions_find_names_aliases_and_skip_common_words():
    idx = MentionIndex(LISTINGS)
    text = ("CHIPS Act money went to Intel and Micron Technology, while Nvidia and Google "
            "lobbied. Our target is lower prices.")
    hits = idx.find(text)
    assert hits.keys() == {"INTC", "MU", "NVDA", "GOOGL"}
    assert hits["MU"] == "Micron Technology"


def test_mentions_skip_acronyms_and_dictionary_words():
    from congress_trades.market.tickers import Listing

    idx = MentionIndex([Listing("STEM", "Stem, Inc. Common Stock", "N"),
                        Listing("MSTR", "Strategy Inc - Class A Common Stock", "Q")])
    assert idx.find("Our STEM workforce needs a Strategy.") == {}


def test_mentions_require_capitalization():
    assert MentionIndex(LISTINGS).find("intel agencies said") == {}


def test_parse_chart_prefers_adjusted_close():
    ts = int(datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc).timestamp())
    payload = {"chart": {"error": None, "result": [{
        "meta": {"gmtoffset": -18000},
        "timestamp": [ts, ts + 86400],
        "indicators": {"quote": [{"close": [100.0, 101.0]}],
                       "adjclose": [{"adjclose": [99.0, None]}]},
    }]}}
    assert parse_chart(json.dumps(payload).encode()) == {date(2024, 1, 2): 99.0}


def test_parse_chart_handles_unknown_symbol():
    payload = {"chart": {"result": None, "error": {"code": "Not Found"}}}
    assert parse_chart(json.dumps(payload).encode()) == {}


def test_mentions_skip_people():
    from congress_trades.market.tickers import Listing, speaker_surnames

    idx = MentionIndex([Listing("ROG", "Rogers Corporation Common Stock", "N")])
    assert idx.find("Thank you, Chairman Rogers.") == {}
    assert idx.find("Rogers said so.", speaker_surnames(["Mr. Rogers"])) == {}
    assert idx.find("Rogers said so.") == {"ROG": "Rogers"}
