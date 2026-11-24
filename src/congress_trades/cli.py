"""Command-line interface.

    congress-trades ingest --year 2025
    congress-trades top --limit 15 --since 2025-06-01
    congress-trades recent --limit 20
    congress-trades analyze --model models/qwen2.5-7b-instruct-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from . import house
from .analysis import most_active
from .store import Store

DEFAULT_DB = Path("data/trades.db")


def _cmd_ingest(args: argparse.Namespace) -> int:
    store = Store(args.db)
    total_new = 0
    for year in args.year:
        filings = house.fetch_ptrs(year)
        new = store.upsert(filings)
        total_new += new
        print(f"{year}: {len(filings)} PTRs found, {new} new")
    print(f"Done. {store.count()} filings in {args.db} (+{total_new}).")
    return 0


def _cmd_top(args: argparse.Namespace) -> int:
    store = Store(args.db)
    rows = most_active(store.filings(since=args.since), limit=args.limit)
    if not rows:
        print("No filings stored. Run `congress-trades ingest` first.")
        return 1
    width = max(len(r.name) for r in rows)
    print(f"{'#':>3}  {'Member':<{width}}  {'Dist':<5} {'PTRs':>5}  Last filed")
    for i, r in enumerate(rows, 1):
        print(f"{i:>3}  {r.name:<{width}}  {r.state_district:<5} {r.filings:>5}  {r.last_filed}")
    return 0


def _cmd_recent(args: argparse.Namespace) -> int:
    store = Store(args.db)
    for f in store.filings()[: args.limit]:
        print(f"{f.filing_date}  {f.name:<28} {f.state_district:<5} {f.pdf_url}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from . import llm

    filings = Store(args.db).filings()[: args.limit]
    if not filings:
        print("No filings stored. Run `congress-trades ingest` first.")
        return 1
    result = llm.analyze(llm.load_model(str(args.model)), filings)
    print(result.summary)
    for item in result.watchlist:
        print(f"  - {item.get('member')}: {item.get('reason')}")
    print(f"confidence: {result.confidence:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="congress-trades",
        description="Track STOCK Act trade disclosures filed by members of Congress.",
    )
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite path (default: %(default)s)")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="Download PTR filings from the House Clerk")
    ing.add_argument("--year", type=int, nargs="+", default=[date.today().year])
    ing.set_defaults(func=_cmd_ingest)

    top = sub.add_parser("top", help="Members ranked by number of PTRs filed")
    top.add_argument("--limit", type=int, default=10)
    top.add_argument("--since", type=date.fromisoformat, default=None, help="YYYY-MM-DD")
    top.set_defaults(func=_cmd_top)

    rec = sub.add_parser("recent", help="Most recent filings with links to the source PDF")
    rec.add_argument("--limit", type=int, default=20)
    rec.set_defaults(func=_cmd_recent)

    ana = sub.add_parser("analyze", help="Have a local llama.cpp model analyze recent filings")
    ana.add_argument("--model", type=Path, required=True, help="Path to a GGUF model file")
    ana.add_argument("--limit", type=int, default=50, help="Number of recent filings to include")
    ana.set_defaults(func=_cmd_analyze)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
