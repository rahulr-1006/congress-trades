"""SQLite persistence: raw transcripts, parsed statements, model outputs, and signal audit log."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .models import Filing, Hearing, Statement

SCHEMA = """
CREATE TABLE IF NOT EXISTS hearings (
    package_id TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    committee  TEXT NOT NULL,
    chamber    TEXT NOT NULL,
    congress   INTEGER NOT NULL,
    held_on    TEXT NOT NULL,
    raw_text   TEXT
);
CREATE TABLE IF NOT EXISTS statements (
    package_id TEXT NOT NULL REFERENCES hearings(package_id),
    seq        INTEGER NOT NULL,
    speaker    TEXT NOT NULL,
    text       TEXT NOT NULL,
    PRIMARY KEY (package_id, seq)
);
-- Every model call is logged with the exact model + task so results are reproducible.
CREATE TABLE IF NOT EXISTS model_outputs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    package_id TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    task       TEXT NOT NULL,
    model      TEXT NOT NULL,
    output     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS filings (
    doc_id         TEXT PRIMARY KEY,
    first          TEXT NOT NULL,
    last           TEXT NOT NULL,
    state_district TEXT NOT NULL,
    year           INTEGER NOT NULL,
    filing_date    TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # --- hearings -------------------------------------------------------
    def has_hearing(self, package_id: str) -> bool:
        q = "SELECT 1 FROM hearings WHERE package_id = ?"
        return self.conn.execute(q, (package_id,)).fetchone() is not None

    def save_hearing(self, h: Hearing, statements: Iterable[Statement], raw_text: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO hearings VALUES (?, ?, ?, ?, ?, ?, ?)",
                (h.package_id, h.title, h.committee, h.chamber, h.congress,
                 h.held_on.isoformat(), raw_text),
            )
            self.conn.execute("DELETE FROM statements WHERE package_id = ?", (h.package_id,))
            self.conn.executemany(
                "INSERT INTO statements VALUES (?, ?, ?, ?)",
                [(s.package_id, s.seq, s.speaker, s.text) for s in statements],
            )

    def hearings(self) -> list[Hearing]:
        rows = self.conn.execute("SELECT * FROM hearings ORDER BY held_on DESC")
        return [
            Hearing(r["package_id"], r["title"], r["committee"], r["chamber"], r["congress"],
                    date.fromisoformat(r["held_on"]))
            for r in rows
        ]

    def statements(self, package_id: str, min_chars: int = 0) -> list[Statement]:
        rows = self.conn.execute(
            "SELECT * FROM statements WHERE package_id = ? AND length(text) >= ? ORDER BY seq",
            (package_id, min_chars),
        )
        return [Statement(r["package_id"], r["seq"], r["speaker"], r["text"]) for r in rows]

    def log_output(self, st: Statement, task: str, model: str, output: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO model_outputs (package_id, seq, task, model, output, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (st.package_id, st.seq, task, model, json.dumps(output),
                 datetime.now(timezone.utc).isoformat()),
            )

    # --- PTR filings ----------------------------------------------------
    def upsert_filings(self, filings: Iterable[Filing]) -> int:
        before = self.filing_count()
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO filings VALUES (?, ?, ?, ?, ?, ?)",
                [(f.doc_id, f.first, f.last, f.state_district, f.year, f.filing_date.isoformat())
                 for f in filings],
            )
        return self.filing_count() - before

    def filing_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]

    def filings(self, since: date | None = None) -> list[Filing]:
        sql, args = "SELECT * FROM filings", ()
        if since:
            sql, args = sql + " WHERE filing_date >= ?", (since.isoformat(),)
        rows = self.conn.execute(sql + " ORDER BY filing_date DESC, doc_id", args)
        return [
            Filing(r["doc_id"], r["first"], r["last"], r["state_district"], r["year"],
                   date.fromisoformat(r["filing_date"]))
            for r in rows
        ]
