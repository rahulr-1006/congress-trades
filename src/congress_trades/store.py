"""SQLite persistence for filings."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from .models import Filing

SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    doc_id         TEXT PRIMARY KEY,
    chamber        TEXT NOT NULL DEFAULT 'house',
    first          TEXT NOT NULL,
    last           TEXT NOT NULL,
    state_district TEXT NOT NULL,
    year           INTEGER NOT NULL,
    filing_date    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_filings_date ON filings(filing_date);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def upsert(self, filings: Iterable[Filing]) -> int:
        """Insert filings, ignoring ones already stored. Returns number of new rows."""
        before = self.count()
        self.conn.executemany(
            "INSERT OR IGNORE INTO filings "
            "(doc_id, first, last, state_district, year, filing_date) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (f.doc_id, f.first, f.last, f.state_district, f.year, f.filing_date.isoformat())
                for f in filings
            ],
        )
        self.conn.commit()
        return self.count() - before

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]

    def filings(self, since: date | None = None) -> list[Filing]:
        sql = "SELECT * FROM filings"
        args: tuple = ()
        if since:
            sql += " WHERE filing_date >= ?"
            args = (since.isoformat(),)
        sql += " ORDER BY filing_date DESC, doc_id"
        return [
            Filing(
                doc_id=r["doc_id"],
                first=r["first"],
                last=r["last"],
                state_district=r["state_district"],
                year=r["year"],
                filing_date=date.fromisoformat(r["filing_date"]),
            )
            for r in self.conn.execute(sql, args)
        ]

    def close(self) -> None:
        self.conn.close()
