from pathlib import Path

import pytest

from congress_trades.ingest.hearings import build_hearing, parse_committee, parse_statements

TEXT = (Path(__file__).parent / "fixtures" / "sample_hearing.txt").read_text()


def test_parse_committee_prefers_subcommittee():
    assert parse_committee(TEXT) == "Subcommittee On Innovation, Data, And Commerce"


def test_parse_statements_attributes_speakers():
    sts = parse_statements("CHRG-test", TEXT)
    assert [s.speaker for s in sts] == ["Mr. Bilirakis", "Ms. Schakowsky", "Dr. Smith"]
    assert "Intel and Micron fast enough." in sts[0].text
    assert "\n" not in sts[0].text
    # Non-speaker lines are folded into the previous statement, adjournment ends parsing
    assert "Christopher Walker" in sts[1].text
    assert sts[-1].text == "I appreciate the opportunity to testify."


@pytest.mark.parametrize("held, issued, expected", [(["2025-12-16"], "2026-01-02", "2025-12-16"),
                                                   ([], "2026-01-02", "2026-01-02")])
def test_build_hearing_dates(held, issued, expected):
    summary = {"packageId": "CHRG-x", "title": "T", "chamber": "HOUSE", "congress": "119",
               "heldDates": held, "dateIssued": issued}
    assert build_hearing(summary, TEXT).held_on.isoformat() == expected
