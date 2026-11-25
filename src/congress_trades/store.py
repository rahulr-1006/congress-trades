"""SQLite persistence: transcripts, statements, model outputs (audit + latency), prices, tickers."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .market.prices import PriceSeries
from .market.tickers import Listing
from .models import Filing, Hearing, Statement

SCHEMA = """
CREATE TABLE IF NOT EXISTS hearings (
    package_id   TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    committee    TEXT NOT NULL,
    chamber      TEXT NOT NULL,
    congress     INTEGER NOT NULL,
    held_on      TEXT NOT NULL,
    raw_text     TEXT,
    published_on TEXT,
    committee_id TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS skipped_packages (
    package_id TEXT PRIMARY KEY,
    reason     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS statements (
    package_id TEXT NOT NULL REFERENCES hearings(package_id),
    seq        INTEGER NOT NULL,
    speaker    TEXT NOT NULL,
    text       TEXT NOT NULL,
    PRIMARY KEY (package_id, seq)
);
-- Every model call: what was asked, which model answered, what it said, and what it cost.
CREATE TABLE IF NOT EXISTS model_outputs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    package_id        TEXT NOT NULL,
    seq               INTEGER NOT NULL,
    task              TEXT NOT NULL,
    model             TEXT NOT NULL,
    output            TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    latency_ms        REAL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS idx_outputs_stmt ON model_outputs(package_id, seq, task, model);
CREATE TABLE IF NOT EXISTS tickers (
    symbol   TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    exchange TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prices (
    symbol    TEXT NOT NULL,
    day       TEXT NOT NULL,
    adj_close REAL NOT NULL,
    PRIMARY KEY (symbol, day)
);
CREATE TABLE IF NOT EXISTS price_coverage (
    symbol TEXT PRIMARY KEY,
    start  TEXT NOT NULL,
    end    TEXT NOT NULL
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

# Columns added after the first release; applied to older databases on open.
MIGRATIONS = {
    "hearings": {"published_on": "TEXT", "committee_id": "TEXT NOT NULL DEFAULT ''"},
    "model_outputs": {"latency_ms": "REAL", "prompt_tokens": "INTEGER",
                      "completion_tokens": "INTEGER"},
}


def _d(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        for table, cols in MIGRATIONS.items():
            have = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for col, ddl in cols.items():
                if col not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
        self.conn.commit()

    # --- hearings -------------------------------------------------------
    def has_hearing(self, package_id: str) -> bool:
        for table in ("hearings", "skipped_packages"):
            q = f"SELECT 1 FROM {table} WHERE package_id = ?"
            if self.conn.execute(q, (package_id,)).fetchone():
                return True
        return False

    def skip_package(self, package_id: str, reason: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO skipped_packages VALUES (?, ?)",
                              (package_id, reason))

    def save_hearing(self, h: Hearing, statements: Iterable[Statement], raw_text: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO hearings (package_id, title, committee, chamber, "
                "congress, held_on, raw_text, published_on, committee_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (h.package_id, h.title, h.committee, h.chamber, h.congress,
                 h.held_on.isoformat(), raw_text,
                 h.published_on.isoformat() if h.published_on else None, h.committee_id),
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
                    date.fromisoformat(r["held_on"]), _d(r["published_on"]), r["committee_id"])
            for r in rows
        ]

    def hearing(self, package_id: str) -> Hearing | None:
        return next((h for h in self.hearings() if h.package_id == package_id), None)

    def statements(self, package_id: str, min_chars: int = 0) -> list[Statement]:
        rows = self.conn.execute(
            "SELECT * FROM statements WHERE package_id = ? AND length(text) >= ? ORDER BY seq",
            (package_id, min_chars),
        )
        return [Statement(r["package_id"], r["seq"], r["speaker"], r["text"]) for r in rows]

    # --- model outputs --------------------------------------------------
    def log_output(
        self, st: Statement, task: str, model: str, output: dict[str, Any],
        latency_ms: float | None = None, usage: dict[str, int] | None = None,
    ) -> None:
        usage = usage or {}
        with self.conn:
            self.conn.execute(
                "INSERT INTO model_outputs (package_id, seq, task, model, output, created_at, "
                "latency_ms, prompt_tokens, completion_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (st.package_id, st.seq, task, model, json.dumps(output),
                 datetime.now(timezone.utc).isoformat(), latency_ms,
                 usage.get("prompt_tokens"), usage.get("completion_tokens")),
            )

    def has_output(self, st: Statement, task: str, model: str) -> bool:
        q = ("SELECT 1 FROM model_outputs WHERE package_id = ? AND seq = ? AND task = ? "
             "AND model = ?")
        return self.conn.execute(q, (st.package_id, st.seq, task, model)).fetchone() is not None

    def outputs(self, model: str | None = None) -> dict[tuple[str, int], dict[str, dict]]:
        """Latest output per (package_id, seq) -> {task: output}."""
        sql, args = "SELECT * FROM model_outputs", ()
        if model:
            sql, args = sql + " WHERE model = ?", (model,)
        out: dict[tuple[str, int], dict[str, dict]] = {}
        for r in self.conn.execute(sql + " ORDER BY id", args):
            out.setdefault((r["package_id"], r["seq"]), {})[r["task"]] = json.loads(r["output"])
        return out

    def latency_rows(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT model, task, latency_ms, prompt_tokens, completion_tokens FROM model_outputs "
            "WHERE latency_ms IS NOT NULL"
        ))

    # --- tickers --------------------------------------------------------
    def save_listings(self, listings: Iterable[Listing]) -> int:
        with self.conn:
            self.conn.execute("DELETE FROM tickers")
            self.conn.executemany("INSERT OR IGNORE INTO tickers VALUES (?, ?, ?)",
                                  [(x.symbol, x.name, x.exchange) for x in listings])
        return self.conn.execute("SELECT COUNT(*) FROM tickers").fetchone()[0]

    def listings(self) -> list[Listing]:
        return [Listing(r["symbol"], r["name"], r["exchange"])
                for r in self.conn.execute("SELECT * FROM tickers ORDER BY rowid")]

    # --- prices ---------------------------------------------------------
    def price_coverage(self, symbol: str) -> tuple[date, date] | None:
        r = self.conn.execute("SELECT * FROM price_coverage WHERE symbol = ?", (symbol,)).fetchone()
        return (date.fromisoformat(r["start"]), date.fromisoformat(r["end"])) if r else None

    def save_prices(self, symbol: str, series: PriceSeries, start: date, end: date) -> None:
        cov = self.price_coverage(symbol)
        if cov:
            start, end = min(start, cov[0]), max(end, cov[1])
        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO prices VALUES (?, ?, ?)",
                                  [(symbol, d.isoformat(), px) for d, px in series.items()])
            self.conn.execute("INSERT OR REPLACE INTO price_coverage VALUES (?, ?, ?)",
                              (symbol, start.isoformat(), end.isoformat()))

    def prices(self, symbol: str) -> PriceSeries:
        rows = self.conn.execute("SELECT day, adj_close FROM prices WHERE symbol = ?", (symbol,))
        return {date.fromisoformat(r["day"]): r["adj_close"] for r in rows}

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
