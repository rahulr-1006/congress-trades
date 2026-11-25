"""Scoring rules: model output -> events for the event study.

Two scorers so the LLM always has something to beat:

- ``mention_baseline``: every company mentioned in a hearing is a long. Measures the "being
  talked about in Congress" effect with no language understanding at all.
- ``llm_v0``: tickers must appear in both the LLM's entity output *and* the deterministic
  mention pre-filter (this drops hallucinated tickers). Direction comes from the signal task
  (positive/negative), falling back to sentiment when |score| >= threshold.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from ..backtest.event_study import Event
from ..market.tickers import MentionIndex, speaker_surnames
from ..models import Hearing, Statement


def decision_date(h: Hearing, timing: str) -> date | None:
    """'held' = trade as if processing the live hearing; 'published' = when the transcript hit
    GovInfo, which is what a transcript-only pipeline could actually act on."""
    return h.held_on if timing == "held" else h.published_on


def _surnames_by_hearing(items: list[tuple[Hearing, Statement]]) -> dict[str, set[str]]:
    speakers: dict[str, set[str]] = {}
    for h, st in items:
        speakers.setdefault(h.package_id, set()).add(st.speaker)
    return {pid: speaker_surnames(sps) for pid, sps in speakers.items()}


def mention_baseline(
    items: Iterable[tuple[Hearing, Statement]], index: MentionIndex, timing: str
) -> list[Event]:
    items = list(items)
    names = _surnames_by_hearing(items)
    events, seen = [], set()
    for h, st in items:
        d = decision_date(h, timing)
        for ticker in index.find(st.text, names[h.package_id]):
            if d and (h.package_id, ticker) not in seen:  # one event per hearing per ticker
                seen.add((h.package_id, ticker))
                events.append(Event(ticker, d, +1, f"{h.package_id}#{st.seq}"))
    return events


def llm_direction(outputs: dict[str, dict], sentiment_threshold: float = 0.5) -> int:
    sig = outputs.get("signal", {})
    if sig.get("policy_signal") and sig.get("direction") in ("positive", "negative"):
        return 1 if sig["direction"] == "positive" else -1
    score = float(outputs.get("sentiment", {}).get("score", 0) or 0)
    if abs(score) >= sentiment_threshold:
        return 1 if score > 0 else -1
    return 0


def llm_v0(
    items: Iterable[tuple[Hearing, Statement]],
    outputs: dict[tuple[str, int], dict[str, dict]],
    index: MentionIndex,
    timing: str,
    sentiment_threshold: float = 0.5,
) -> list[Event]:
    items = list(items)
    names = _surnames_by_hearing(items)
    events = []
    for h, st in items:
        out = outputs.get((st.package_id, st.seq))
        d = decision_date(h, timing)
        if not out or not d:
            continue
        direction = llm_direction(out, sentiment_threshold)
        if direction == 0:
            continue
        llm_tickers = {t.upper() for t in out.get("entities", {}).get("tickers", [])}
        grounded = llm_tickers & set(index.find(st.text, names[h.package_id]))
        events += [Event(t, d, direction, f"{h.package_id}#{st.seq}") for t in sorted(grounded)]
    return events
