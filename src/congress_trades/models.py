from __future__ import annotations

from dataclasses import dataclass
from datetime import date

PTR_PDF_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"


@dataclass(frozen=True)
class Hearing:
    package_id: str
    title: str
    committee: str
    chamber: str
    congress: int
    held_on: date


@dataclass(frozen=True)
class Statement:
    """One continuous speaker turn within a hearing."""

    package_id: str
    seq: int
    speaker: str
    text: str


@dataclass(frozen=True)
class Filing:
    """One Periodic Transaction Report (PTR) filed with the House Clerk."""

    doc_id: str
    first: str
    last: str
    state_district: str
    year: int
    filing_date: date

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}".strip()

    @property
    def state(self) -> str:
        return self.state_district[:2]

    @property
    def pdf_url(self) -> str:
        return PTR_PDF_URL.format(year=self.year, doc_id=self.doc_id)
