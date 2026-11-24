"""Client for the House Clerk's public financial disclosure index.

The Clerk publishes a yearly zip (``{year}FD.zip``) containing an XML index of
every disclosure filed that year. Filings with ``FilingType == "P"`` are
Periodic Transaction Reports: the STOCK Act filings that list individual trades.
"""

from __future__ import annotations

import io
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from datetime import datetime

from ..models import Filing

INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
PTR_FILING_TYPE = "P"
USER_AGENT = "congress-trades/0.1 (+https://github.com/rahulr-1006/congress-trades)"


def download_index(year: int, timeout: float = 30.0) -> bytes:
    """Download the raw XML disclosure index for ``year``."""
    req = urllib.request.Request(INDEX_URL.format(year=year), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        xml_name = next(n for n in zf.namelist() if n.lower().endswith(".xml"))
        return zf.read(xml_name)


def _text(el: ET.Element, tag: str) -> str:
    return (el.findtext(tag) or "").strip()


def parse_ptrs(xml_bytes: bytes) -> Iterator[Filing]:
    """Yield every PTR filing in a disclosure index, skipping malformed rows."""
    root = ET.fromstring(xml_bytes)
    for m in root.iter("Member"):
        if _text(m, "FilingType") != PTR_FILING_TYPE:
            continue
        try:
            filed = datetime.strptime(_text(m, "FilingDate"), "%m/%d/%Y").date()
            year = int(_text(m, "Year"))
        except ValueError:
            continue
        doc_id = _text(m, "DocID")
        if not doc_id:
            continue
        yield Filing(
            doc_id=doc_id,
            first=_text(m, "First"),
            last=_text(m, "Last"),
            state_district=_text(m, "StateDst"),
            year=year,
            filing_date=filed,
        )


def fetch_ptrs(year: int) -> list[Filing]:
    return list(parse_ptrs(download_index(year)))
