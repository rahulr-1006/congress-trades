from datetime import date

from congress_trades.house import parse_ptrs


def test_parse_keeps_only_valid_ptrs(sample_xml):
    filings = list(parse_ptrs(sample_xml))
    assert [f.doc_id for f in filings] == ["20026001", "20026002", "20026003"]


def test_parse_fields(sample_xml):
    f = next(parse_ptrs(sample_xml))
    assert f.name == "Jane Doe"
    assert f.state == "CA"
    assert f.filing_date == date(2025, 3, 14)
    assert f.pdf_url.endswith("/ptr-pdfs/2025/20026001.pdf")
