from datetime import date
from pathlib import Path

from congress_trades.ingest.hearings import (
    build_hearing,
    parse_committee,
    parse_mods,
    parse_statements,
)

FIX = Path(__file__).parent / "fixtures"
TEXT = (FIX / "sample_hearing.txt").read_text()
MODS = (FIX / "sample_mods.xml").read_bytes()


def test_parse_committee_prefers_subcommittee_and_joins_wrapped_lines():
    assert parse_committee(TEXT) == "Subcommittee On Innovation, Data, And Commerce"


def test_parse_statements_attributes_speakers():
    sts = parse_statements("CHRG-test", TEXT)
    assert [s.speaker for s in sts] == ["Mr. Bilirakis", "Ms. Schakowsky", "Dr. Smith"]
    assert "Intel and Micron fast enough." in sts[0].text
    assert "\n" not in sts[0].text
    assert "Christopher Walker" in sts[1].text  # non-speaker line folds into previous turn
    assert sts[-1].text == "I appreciate the opportunity to testify."


def test_parse_mods():
    meta = parse_mods(MODS)
    assert meta["held_dates"] == ["2023-06-14"]
    assert meta["date_ingested"] == "2024-02-01"
    assert meta["committees"] == [("hsif00", "Energy and Commerce")]
    assert meta["chamber"] == "HOUSE" and meta["congress"] == "118"


def test_build_hearing_uses_held_and_published_dates():
    h = build_hearing("CHRG-x", parse_mods(MODS), TEXT)
    assert h.held_on == date(2023, 6, 14)
    assert h.published_on == date(2024, 2, 1)
    assert h.committee == "Energy and Commerce" and h.committee_id == "hsif00"


def test_build_hearing_falls_back_to_transcript_committee_and_issue_date():
    meta = {**parse_mods(MODS), "committees": [], "held_dates": [], "date_ingested": ""}
    h = build_hearing("CHRG-x", meta, TEXT)
    assert h.held_on == date(2023, 6, 20)
    assert h.published_on is None
    assert h.committee.startswith("Subcommittee On Innovation")
