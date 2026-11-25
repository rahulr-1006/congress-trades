"""Congressional hearing transcripts from GovInfo (GPO), no API key required.

- Package lists come from the public sitemaps (``CHRG_{year}_sitemap.xml``).
- Transcripts come from ``/content/pkg/{id}/html/{id}.htm`` (preformatted text where every
  speaker turn starts on an indented line like ``    Mr. Keating. Thank you, Mr. Chair.``).
- Metadata comes from MODS XML: held date, committee, and ``dateIngested``, the date the
  transcript was published on GovInfo. That date matters for backtests, because transcripts
  usually appear months after the hearing.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import date

from .. import http
from ..models import Hearing, Statement

SITEMAP_URL = "https://www.govinfo.gov/sitemap/CHRG_{year}_sitemap.xml"
CONTENT_URL = "https://www.govinfo.gov/content/pkg/{pid}/html/{pid}.htm"
MODS_URL = "https://www.govinfo.gov/metadata/pkg/{pid}/mods.xml"
MODS_NS = {"m": "http://www.loc.gov/mods/v3"}

SPEAKER_RE = re.compile(
    r"^ {4}("
    r"(?:Mr|Ms|Mrs|Dr)\. [A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)?"
    r"|(?:Chairman|Chairwoman|Chair|Senator|Secretary|Ambassador|General|Admiral) [A-Z][A-Za-z'\-]+"
    r"|The (?:Chairman|Chairwoman|Chair)"
    r")\. (.*)$"
)
COMMITTEE_RE = re.compile(r"^((?:SUB)?COMMITTEE ON [A-Z ,&'\-]+?)\s*$")
PACKAGE_RE = re.compile(r"CHRG-\d+[a-z]+\d+")


def list_packages(year: int) -> list[str]:
    """All hearing package IDs GovInfo files under ``year``."""
    xml = http.get(SITEMAP_URL.format(year=year)).decode()
    return sorted(set(PACKAGE_RE.findall(xml)))


def parse_mods(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)

    def first(path: str) -> str:
        el = root.find(path, MODS_NS)
        return (el.text or "").strip() if el is not None else ""

    committees = [
        (c.get("authorityId", ""), (c.findtext("m:name[@type='authority-short']", "", MODS_NS)))
        for c in root.iterfind(".//m:extension/m:congCommittee", MODS_NS)
    ]
    return {
        "title": first(".//m:titleInfo/m:title"),
        "held_dates": [e.text.strip() for e in root.iterfind(".//m:extension/m:heldDate", MODS_NS)
                       if e.text],
        "date_issued": first(".//m:originInfo/m:dateIssued"),
        "date_ingested": first(".//m:extension/m:dateIngested"),
        "chamber": first(".//m:extension/m:chamber"),
        "congress": first(".//m:extension/m:congress"),
        "committees": committees,
    }


def html_to_text(raw: bytes) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", raw.decode("utf-8", errors="replace")))


def parse_committee(text: str) -> str:
    """Fallback: most specific committee named in the transcript header (subcommittee wins)."""
    header = [ln.strip() for ln in text.splitlines()[:120]]
    found = []
    for i, line in enumerate(header):
        if not (m := COMMITTEE_RE.match(line)):
            continue
        name = m.group(1)
        # Long names wrap: "SUBCOMMITTEE ON INTERIOR, ENVIRONMENT, AND" / "RELATED AGENCIES"
        nxt = header[i + 1] if i + 1 < len(header) else ""
        if name.endswith((" AND", ",", " OF", " ON", " THE")) and nxt.isupper():
            name = f"{name} {nxt}"
        found.append(name.title())
    subs = [c for c in found if c.startswith("Subcommittee")]
    return (subs or found or [""])[0]


def parse_statements(package_id: str, text: str) -> list[Statement]:
    """Split a transcript into speaker-attributed statements, preserving order."""
    statements: list[Statement] = []
    speaker: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if speaker and buf:
            body = " ".join(" ".join(buf).split())
            statements.append(Statement(package_id, len(statements), speaker, body))

    for line in text.splitlines():
        m = SPEAKER_RE.match(line)
        if m:
            flush()
            speaker, buf = m.group(1), [m.group(2)]
        elif speaker is not None:
            if line.strip().startswith("[Whereupon"):
                flush()
                speaker, buf = None, []
            else:
                buf.append(line.strip())
    flush()
    return statements


def build_hearing(package_id: str, meta: dict, text: str) -> Hearing:
    held = (meta["held_dates"] or [meta["date_issued"]])[0]
    ingested = meta.get("date_ingested")
    committee_id, committee = (meta["committees"] or [("", "")])[0]
    return Hearing(
        package_id=package_id,
        title=meta["title"],
        committee=committee or parse_committee(text),
        committee_id=committee_id,
        chamber=meta["chamber"],
        congress=int(meta["congress"] or 0),
        held_on=date.fromisoformat(held[:10]),
        published_on=date.fromisoformat(ingested[:10]) if ingested else None,
    )


def fetch_meta(package_id: str) -> dict:
    return parse_mods(http.get(MODS_URL.format(pid=package_id)))


def fetch_hearing(
    package_id: str, meta: dict | None = None
) -> tuple[Hearing, list[Statement], str]:
    meta = meta or fetch_meta(package_id)
    text = html_to_text(http.get(CONTENT_URL.format(pid=package_id)))
    return build_hearing(package_id, meta, text), parse_statements(package_id, text), text
