from datetime import date

from congress_trades.analysis import most_active
from congress_trades.house import parse_ptrs
from congress_trades.store import Store


def test_upsert_is_idempotent(sample_xml):
    store = Store()
    filings = list(parse_ptrs(sample_xml))
    assert store.upsert(filings) == 3
    assert store.upsert(filings) == 0
    assert store.count() == 3


def test_since_filter(sample_xml):
    store = Store()
    store.upsert(parse_ptrs(sample_xml))
    assert [f.doc_id for f in store.filings(since=date(2025, 6, 1))] == ["20026002"]


def test_most_active_ranking(sample_xml):
    top = most_active(parse_ptrs(sample_xml))
    assert [(s.name, s.filings) for s in top] == [("Jane Doe", 2), ("John Smith", 1)]
    assert top[0].last_filed == date(2025, 7, 2)
