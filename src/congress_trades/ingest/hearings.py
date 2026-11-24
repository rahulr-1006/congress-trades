"""Congressional hearing transcripts from the GPO's GovInfo API (``CHRG`` collection).

GovInfo publishes official hearing transcripts as preformatted text, with every speaker turn
starting on an indented line such as ``    Mr. Keating. Thank you, Mr. Chair.``. This module
lists hearings, downloads transcripts, and splits them into attributed statements.

Set ``GOVINFO_API_KEY`` (free at https://api.data.gov/signup); falls back to ``DEMO_KEY``,
which is heavily rate-limited.
"""

from __future__ import annotations

import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date

from ..models import Hearing, Statement

API = "https://api.govinfo.gov"
USER_AGENT = "congress-trades/0.1 (+https://github.com/rahulr-1006/congress-trades)"

SPEAKER_RE = re.compile(
    r"^ {4}("
    r"(?:Mr|Ms|Mrs|Dr)\. [A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)?"
    r"|(?:Chairman|Chairwoman|Chair|Senator|Secretary|Ambassador|General|Admiral) [A-Z][A-Za-z'\-]+"
    r"|The (?:Chairman|Chairwoman|Chair)"
    r")\. (.*)$"
)
COMMITTEE_RE = re.compile(r"^((?:SUB)?COMMITTEE ON [A-Z ,&'\-]+?)\s*$")


class RateLimited(RuntimeError):
    pass


@dataclass(frozen=True)
class HearingRef:
    package_id: str
    title: str
    date_issued: str


def _api_key() -> str:
    return os.environ.get("GOVINFO_API_KEY", "DEMO_KEY")


def _get(path: str, **params: str) -> bytes:
    params["api_key"] = _api_key()
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimited(
                "GovInfo rate limit hit. Get a free key at https://api.data.gov/signup "
                "and export GOVINFO_API_KEY=..."
            ) from e
        raise


def list_hearings(since: date, limit: int = 20) -> list[HearingRef]:
    """Hearings published or modified on GovInfo since ``since`` (newest first)."""
    path = f"/collections/CHRG/{since.isoformat()}T00:00:00Z"
    data = json.loads(_get(path, pageSize=str(limit), offsetMark="*"))
    return [
        HearingRef(p["packageId"], p["title"], p.get("dateIssued", ""))
        for p in data.get("packages", [])
    ]


def fetch_transcript(package_id: str) -> tuple[dict, str]:
    """Return (package summary metadata, plain-text transcript)."""
    summary = json.loads(_get(f"/packages/{package_id}/summary"))
    raw = _get(f"/packages/{package_id}/htm").decode("utf-8", errors="replace")
    text = html.unescape(re.sub(r"<[^>]+>", "", raw))
    return summary, text


def parse_committee(text: str) -> str:
    """The most specific committee named in the transcript header (subcommittee wins)."""
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


def build_hearing(summary: dict, text: str) -> Hearing:
    held = (summary.get("heldDates") or [summary.get("dateIssued")])[0]
    return Hearing(
        package_id=summary["packageId"],
        title=summary.get("title", ""),
        committee=parse_committee(text),
        chamber=summary.get("chamber", ""),
        congress=int(summary.get("congress", 0) or 0),
        held_on=date.fromisoformat(held),
    )


def fetch_hearing(package_id: str) -> tuple[Hearing, list[Statement], str]:
    summary, text = fetch_transcript(package_id)
    return build_hearing(summary, text), parse_statements(package_id, text), text
