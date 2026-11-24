"""Command-line interface.

    congress-trades hearings fetch --since 2026-01-01 --limit 10
    congress-trades hearings list
    congress-trades extract CHRG-119hhrg64507 --model models/llama-3.1-8b-instruct-q4_k_m.gguf
    congress-trades ptr ingest --year 2025
    congress-trades ptr top
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from .analysis import most_active
from .ingest import hearings as hearings_src
from .ingest import house_ptr
from .store import Store

DEFAULT_DB = Path("data/congress.db")


def _hearings_fetch(args: argparse.Namespace, store: Store) -> int:
    try:
        refs = hearings_src.list_hearings(args.since, limit=args.limit)
    except hearings_src.RateLimited as e:
        print(e)
        return 2
    for ref in refs:
        if store.has_hearing(ref.package_id):
            print(f"skip  {ref.package_id} (already stored)")
            continue
        try:
            hearing, statements, raw = hearings_src.fetch_hearing(ref.package_id)
        except hearings_src.RateLimited as e:
            print(e)
            return 2
        if not statements:
            print(f"skip  {ref.package_id} (no transcript text published yet)")
            continue
        store.save_hearing(hearing, statements, raw)
        print(f"saved {hearing.package_id}  {hearing.held_on}  {len(statements):>4} statements  "
              f"{hearing.committee or '?'}: {hearing.title[:60]}")
    return 0


def _hearings_list(args: argparse.Namespace, store: Store) -> int:
    for h in store.hearings():
        n = len(store.statements(h.package_id))
        print(f"{h.held_on}  {h.package_id:<20} {n:>4} stmts  {h.title[:70]}")
    return 0


def _extract(args: argparse.Namespace, store: Store) -> int:
    from .llm import LlamaCppEngine, run_task

    hearing = next((h for h in store.hearings() if h.package_id == args.package_id), None)
    if hearing is None:
        print(f"{args.package_id} not stored; run `hearings fetch` first.")
        return 1
    engine = LlamaCppEngine(str(args.model))
    model_id = args.model.name
    for st in store.statements(hearing.package_id, min_chars=args.min_chars)[: args.limit]:
        for task in args.tasks:
            out = run_task(engine, task, hearing, st)
            store.log_output(st, task, model_id, out)
            print(f"[{st.seq:>3}] {st.speaker:<18} {task:<9} {out}")
    return 0


def _ptr_ingest(args: argparse.Namespace, store: Store) -> int:
    for year in args.year:
        filings = house_ptr.fetch_ptrs(year)
        print(f"{year}: {len(filings)} PTRs, {store.upsert_filings(filings)} new")
    return 0


def _ptr_top(args: argparse.Namespace, store: Store) -> int:
    rows = most_active(store.filings(since=args.since), limit=args.limit)
    if not rows:
        print("No filings stored. Run `congress-trades ptr ingest` first.")
        return 1
    width = max(len(r.name) for r in rows)
    print(f"{'#':>3}  {'Member':<{width}}  {'Dist':<5} {'PTRs':>5}  Last filed")
    for i, r in enumerate(rows, 1):
        print(f"{i:>3}  {r.name:<{width}}  {r.state_district:<5} {r.filings:>5}  {r.last_filed}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="congress-trades",
        description="Trading signals from congressional hearings, extracted by a local LLM.",
    )
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite path (default: %(default)s)")
    sub = p.add_subparsers(dest="command", required=True)

    hp = sub.add_parser("hearings", help="Hearing transcript ingestion").add_subparsers(
        dest="action", required=True
    )
    f = hp.add_parser("fetch", help="Download new hearing transcripts from GovInfo")
    f.add_argument("--since", type=date.fromisoformat, default=date.today() - timedelta(days=30))
    f.add_argument("--limit", type=int, default=10)
    f.set_defaults(func=_hearings_fetch)
    hp.add_parser("list", help="List stored hearings").set_defaults(func=_hearings_list)

    ex = sub.add_parser("extract", help="Run local llama.cpp extraction over a stored hearing")
    ex.add_argument("package_id")
    ex.add_argument("--model", type=Path, required=True, help="Path to a GGUF model")
    ex.add_argument("--tasks", nargs="+", default=["entities", "sentiment", "signal"])
    ex.add_argument("--limit", type=int, default=25, help="Max statements to process")
    ex.add_argument("--min-chars", type=int, default=200, help="Skip short procedural remarks")
    ex.set_defaults(func=_extract)

    pp = sub.add_parser("ptr", help="House trade disclosures (PTRs)").add_subparsers(
        dest="action", required=True
    )
    pi = pp.add_parser("ingest", help="Download PTR filings from the House Clerk")
    pi.add_argument("--year", type=int, nargs="+", default=[date.today().year])
    pi.set_defaults(func=_ptr_ingest)
    pt = pp.add_parser("top", help="Members ranked by number of PTRs filed")
    pt.add_argument("--limit", type=int, default=10)
    pt.add_argument("--since", type=date.fromisoformat, default=None)
    pt.set_defaults(func=_ptr_top)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args, Store(args.db))


if __name__ == "__main__":
    sys.exit(main())
