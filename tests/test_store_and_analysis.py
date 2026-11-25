import sqlite3
from datetime import date

from congress_trades.analysis import most_active
from congress_trades.ingest.house_ptr import parse_ptrs
from congress_trades.models import Statement
from congress_trades.store import Store


def test_upsert_is_idempotent(sample_xml):
    store = Store()
    filings = list(parse_ptrs(sample_xml))
    assert store.upsert_filings(filings) == 3
    assert store.upsert_filings(filings) == 0


def test_since_filter(sample_xml):
    store = Store()
    store.upsert_filings(parse_ptrs(sample_xml))
    assert [f.doc_id for f in store.filings(since=date(2025, 6, 1))] == ["20026002"]


def test_most_active_ranking(sample_xml):
    top = most_active(parse_ptrs(sample_xml))
    assert [(s.name, s.filings) for s in top] == [("Jane Doe", 2), ("John Smith", 1)]


def test_price_cache_merges_coverage():
    store = Store()
    store.save_prices("X", {date(2024, 1, 2): 1.0}, date(2024, 1, 1), date(2024, 1, 31))
    store.save_prices("X", {date(2024, 3, 1): 2.0}, date(2024, 3, 1), date(2024, 3, 31))
    assert store.price_coverage("X") == (date(2024, 1, 1), date(2024, 3, 31))
    assert len(store.prices("X")) == 2


def test_output_log_and_lookup():
    store = Store()
    st = Statement("P", 1, "Mr. A", "text")
    store.log_output(st, "entities", "m.gguf", {"tickers": []}, 812.5,
                     {"prompt_tokens": 300, "completion_tokens": 40})
    assert store.has_output(st, "entities", "m.gguf")
    assert not store.has_output(st, "entities", "other.gguf")
    assert store.latency_rows()[0]["completion_tokens"] == 40


def test_migrates_old_database(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE hearings (package_id TEXT PRIMARY KEY, title TEXT NOT NULL, "
                 "committee TEXT NOT NULL, chamber TEXT NOT NULL, congress INTEGER NOT NULL, "
                 "held_on TEXT NOT NULL, raw_text TEXT)")
    conn.commit()
    conn.close()
    store = Store(db)
    cols = {r["name"] for r in store.conn.execute("PRAGMA table_info(hearings)")}
    assert {"published_on", "committee_id"} <= cols
