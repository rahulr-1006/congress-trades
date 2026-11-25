"""Command-line interface. Run ``congress-trades -h`` for the full list."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from .analysis import most_active
from .backtest import splits
from .backtest.event_study import HORIZONS, run_event_study
from .ingest import hearings as src
from .ingest import house_ptr
from .market import tickers as tk
from .market.loader import load_prices
from .signals import scoring
from .store import Store

DEFAULT_DB = Path("data/congress.db")
MARKET_COMMITTEES = [
    "Financial Services", "Banking", "Energy and Commerce", "Commerce", "Armed Services",
    "Finance", "Ways and Means", "Energy and Natural Resources", "Agriculture",
]


# ---------------------------------------------------------------- hearings
def _save(store: Store, pid: str, meta: dict | None = None) -> str:
    hearing, statements, raw = src.fetch_hearing(pid, meta)
    if not statements:
        store.skip_package(pid, "no transcript text")
        return f"skip  {pid} (no transcript text)"
    store.save_hearing(hearing, statements, raw)
    return (f"saved {pid}  held {hearing.held_on}  published {hearing.published_on}  "
            f"{len(statements):>4} stmts  {hearing.committee}: {hearing.title[:50]}")


def _hearings_fetch(args, store: Store) -> int:
    for pid in args.package_ids:
        print(_save(store, pid))
    return 0


def _hearings_backfill(args, store: Store) -> int:
    wanted = [c.lower() for c in args.committee]
    pids = src.list_packages(args.year)
    print(f"{args.year}: {len(pids)} hearing packages on GovInfo")
    saved = 0
    for pid in pids:
        if saved >= args.limit:
            break
        if store.has_hearing(pid):
            continue
        meta = src.fetch_meta(pid)
        names = " | ".join(name for _, name in meta["committees"]).lower()
        if wanted and not any(w in names for w in wanted):
            store.skip_package(pid, f"committee: {names or '?'}")
            continue
        msg = _save(store, pid, meta)
        saved += msg.startswith("saved")
        print(msg)
    print(f"Saved {saved} hearings.")
    return 0


def _hearings_list(args, store: Store) -> int:
    for h in store.hearings():
        n = len(store.statements(h.package_id))
        print(f"{h.held_on}  pub {h.published_on}  {h.package_id:<20} {n:>4}  "
              f"{h.committee[:22]:<22} {h.title[:50]}")
    return 0


# ---------------------------------------------------------------- tickers / prices
def _mention_index(store: Store) -> tk.MentionIndex:
    listings = store.listings()
    if not listings:
        sys.exit("No tickers stored. Run `congress-trades tickers sync` first.")
    return tk.MentionIndex(listings)


def _tickers_sync(args, store: Store) -> int:
    print(f"Stored {store.save_listings(tk.fetch_listings())} listed common stocks.")
    return 0


def _mentions(args, store: Store) -> int:
    index = _mention_index(store)
    stmts = store.statements(args.package_id)
    names = tk.speaker_surnames(s.speaker for s in stmts)
    hit = 0
    for st in stmts:
        found = index.find(st.text, names)
        if found:
            hit += 1
            print(f"[{st.seq:>3}] {st.speaker:<18} {found}")
    print(f"{hit}/{len(stmts)} statements mention a listed company.")
    return 0


def _prices(args, store: Store) -> int:
    series = load_prices(store, args.symbol.upper(), args.start, args.end)
    days = sorted(series)
    if not days:
        print("No prices returned.")
        return 1
    print(f"{args.symbol.upper()}: {len(days)} days {days[0]} -> {days[-1]}, "
          f"last adj close {series[days[-1]]:.2f}")
    return 0


# ---------------------------------------------------------------- LLM
def _extract(args, store: Store) -> int:
    from .llm import LlamaCppEngine, run_task

    pids = [h.package_id for h in store.hearings()] if args.package_id == "all" else [
        args.package_id]
    index = _mention_index(store) if args.prefilter else None
    engine = LlamaCppEngine(str(args.model))
    model_id = args.model.name
    for pid in pids:
        hearing = store.hearing(pid)
        if hearing is None:
            print(f"{pid} not stored; run `hearings fetch {pid}` first.")
            return 1
        stmts = store.statements(pid, min_chars=args.min_chars)
        if index:
            names = tk.speaker_surnames(s.speaker for s in store.statements(pid))
            stmts = [s for s in stmts if index.find(s.text, names)]
        print(f"{pid}: {len(stmts)} statements to process")
        for st in stmts[: args.limit]:
            for task in args.tasks:
                if store.has_output(st, task, model_id):
                    continue
                t0 = time.perf_counter()
                try:
                    out = run_task(engine, task, hearing, st)
                except ValueError as e:
                    print(f"[{st.seq:>3}] {task}: invalid output ({e})")
                    continue
                ms = (time.perf_counter() - t0) * 1000
                store.log_output(st, task, model_id, out, ms, engine.last_usage)
                print(f"[{st.seq:>3}] {st.speaker:<18} {task:<9} {ms:>6.0f}ms {out}")
    return 0


def _latency(args, store: Store) -> int:
    rows = store.latency_rows()
    if not rows:
        print("No timed model calls yet. Run `extract` first.")
        return 1
    groups: dict[tuple[str, str], list] = {}
    for r in rows:
        groups.setdefault((r["model"], r["task"]), []).append(r)
    print(f"{'model':<38} {'task':<10} {'n':>4} {'p50 ms':>8} {'p95 ms':>8} {'out tok/s':>10}")
    for (model, task), rs in sorted(groups.items()):
        lat = sorted(r["latency_ms"] for r in rs)
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        toks = [r["completion_tokens"] / (r["latency_ms"] / 1000) for r in rs
                if r["completion_tokens"]]
        tps = f"{statistics.mean(toks):.1f}" if toks else "-"
        print(f"{model[:38]:<38} {task:<10} {len(rs):>4} {statistics.median(lat):>8.0f} "
              f"{p95:>8.0f} {tps:>10}")
    return 0


# ---------------------------------------------------------------- validation
def _validate(args, store: Store) -> int:
    dev, holdout = splits.split(store.hearings())
    pool = {"dev": dev, "holdout": holdout, "all": dev + holdout}[args.split]
    if args.split != "dev":
        print("WARNING: evaluating on the holdout set. Don't tune on what you see here.\n")
    items = [(h, st) for h in pool for st in store.statements(h.package_id)]
    index = _mention_index(store)
    if args.scorer == "baseline":
        events = scoring.mention_baseline(items, index, args.timing)
    else:
        events = scoring.llm_v0(items, store.outputs(args.model), index, args.timing,
                                args.sentiment_threshold)
    if not events:
        print("No events generated. (LLM scorer needs `extract` outputs for these hearings.)")
        return 1
    lo = min(e.decision_date for e in events) - timedelta(days=7)
    hi = min(max(e.decision_date for e in events) + timedelta(days=45), date.today())
    bench = load_prices(store, args.benchmark, lo, hi)
    prices = {t: load_prices(store, t, lo, hi) for t in sorted({e.ticker for e in events})}
    res = run_event_study(events, prices, bench)

    print(f"scorer={args.scorer} timing={args.timing} split={args.split} "
          f"hearings={len(pool)} events={len(events)} benchmark={args.benchmark}")
    print(f"skipped: {len(res.skipped_no_prices)} no price data (delisted?), "
          f"{len(res.skipped_out_of_range)} too recent\n")
    print(f"{'horizon':>8} {'n':>5} {'mean CAR':>10} {'t-stat':>8} {'hit rate':>9}")
    for h in HORIZONS:
        s = res.horizons[h]
        print(f"{h:>6}d {s.n:>6} {s.mean * 100:>9.2f}% {s.t_stat:>8.2f} {s.hit_rate:>8.1%}")
    return 0


# ---------------------------------------------------------------- PTRs
def _ptr_ingest(args, store: Store) -> int:
    for year in args.year:
        filings = house_ptr.fetch_ptrs(year)
        print(f"{year}: {len(filings)} PTRs, {store.upsert_filings(filings)} new")
    return 0


def _ptr_top(args, store: Store) -> int:
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
    iso = date.fromisoformat

    hp = sub.add_parser("hearings", help="Hearing transcripts").add_subparsers(
        dest="action", required=True)
    f = hp.add_parser("fetch", help="Download specific hearings by GovInfo package ID")
    f.add_argument("package_ids", nargs="+")
    f.set_defaults(func=_hearings_fetch)
    b = hp.add_parser("backfill", help="Bulk-download a year of hearings from market committees")
    b.add_argument("--year", type=int, required=True)
    b.add_argument("--committee", nargs="*", default=MARKET_COMMITTEES,
                   help="Substring filters on committee name (empty = all committees)")
    b.add_argument("--limit", type=int, default=25)
    b.set_defaults(func=_hearings_backfill)
    hp.add_parser("list", help="List stored hearings").set_defaults(func=_hearings_list)

    tp = sub.add_parser("tickers", help="US-listed ticker universe").add_subparsers(
        dest="action", required=True)
    tp.add_parser("sync", help="Download Nasdaq/NYSE listings").set_defaults(func=_tickers_sync)

    m = sub.add_parser("mentions", help="Show listed companies mentioned in a hearing")
    m.add_argument("package_id")
    m.set_defaults(func=_mentions)

    pr = sub.add_parser("prices", help="Load (and cache) daily adjusted prices")
    pr.add_argument("symbol")
    pr.add_argument("--start", type=iso, default=date.today() - timedelta(days=365))
    pr.add_argument("--end", type=iso, default=date.today())
    pr.set_defaults(func=_prices)

    ex = sub.add_parser("extract", help="Run local llama.cpp extraction over stored hearings")
    ex.add_argument("package_id", help="GovInfo package ID, or 'all'")
    ex.add_argument("--model", type=Path, required=True, help="Path to a GGUF model")
    ex.add_argument("--tasks", nargs="+", default=["entities", "sentiment", "signal"])
    ex.add_argument("--limit", type=int, default=25, help="Max statements per hearing")
    ex.add_argument("--min-chars", type=int, default=200, help="Skip short procedural remarks")
    ex.add_argument("--no-prefilter", dest="prefilter", action="store_false",
                    help="Also process statements that mention no listed company")
    ex.set_defaults(func=_extract)

    sub.add_parser("latency", help="Per-model/task latency and throughput").set_defaults(
        func=_latency)

    v = sub.add_parser("validate", help="Event-study backtest of signals vs. a benchmark")
    v.add_argument("--scorer", choices=["baseline", "llm"], default="baseline")
    v.add_argument("--timing", choices=["held", "published"], default="published")
    v.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    v.add_argument("--model", default=None, help="Filter LLM outputs to this model file name")
    v.add_argument("--benchmark", default="SPY")
    v.add_argument("--sentiment-threshold", type=float, default=0.5)
    v.set_defaults(func=_validate)

    pp = sub.add_parser("ptr", help="House trade disclosures (PTRs)").add_subparsers(
        dest="action", required=True)
    pi = pp.add_parser("ingest", help="Download PTR filings from the House Clerk")
    pi.add_argument("--year", type=int, nargs="+", default=[date.today().year])
    pi.set_defaults(func=_ptr_ingest)
    pt = pp.add_parser("top", help="Members ranked by number of PTRs filed")
    pt.add_argument("--limit", type=int, default=10)
    pt.add_argument("--since", type=iso, default=None)
    pt.set_defaults(func=_ptr_top)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args, Store(args.db))


if __name__ == "__main__":
    sys.exit(main())
