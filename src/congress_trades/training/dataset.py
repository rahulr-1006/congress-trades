"""Build chat-format JSONL datasets for ``mlx_lm lora`` from teacher outputs.

- Candidates: dev-period statements (pre-cutoff) that mention a listed company.
- Splits are by *hearing* (hash of package id), so no hearing leaks across train/valid.
- Label cleaning: teacher tickers not grounded in the statement text are removed before
  training, so students don't learn the teacher's hallucinations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..backtest import splits
from ..llm.tasks import messages_for
from ..market.tickers import MentionIndex, speaker_surnames
from ..models import Hearing, Statement
from ..store import Store

TASK = "combined"


def is_valid_hearing(package_id: str, valid_frac: float) -> bool:
    bucket = int(hashlib.sha256(package_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < valid_frac


def candidates(store: Store, index: MentionIndex, min_chars: int = 200
               ) -> list[tuple[Hearing, Statement]]:
    dev, _ = splits.split(store.hearings())
    out = []
    for h in dev:
        stmts = store.statements(h.package_id)
        names = speaker_surnames(s.speaker for s in stmts)
        out += [(h, s) for s in stmts if len(s.text) >= min_chars and index.find(s.text, names)]
    return out


def clean_label(label: dict[str, Any], grounded: Iterable[str]) -> dict[str, Any]:
    keep = set(grounded)
    out = dict(label)
    out["tickers"] = [t for t in label.get("tickers", []) if str(t).upper() in keep]
    return out


@dataclass
class BuildStats:
    train: int = 0
    valid: int = 0
    unlabeled: int = 0
    tickers_dropped: int = 0


def build(store: Store, index: MentionIndex, teacher: str, out_dir: Path,
          valid_frac: float = 0.15) -> BuildStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = store.outputs(teacher)
    stats = BuildStats()
    rows: dict[str, list[str]] = {"train": [], "valid": []}
    items = candidates(store, index)
    names_by_pid: dict[str, set[str]] = {}
    for h, st in items:
        label = labels.get((st.package_id, st.seq), {}).get(TASK)
        if label is None:
            stats.unlabeled += 1
            continue
        if h.package_id not in names_by_pid:
            names_by_pid[h.package_id] = speaker_surnames(
                s.speaker for s in store.statements(h.package_id))
        cleaned = clean_label(label, index.find(st.text, names_by_pid[h.package_id]))
        stats.tickers_dropped += len(label.get("tickers", [])) - len(cleaned["tickers"])
        msgs = messages_for(TASK, h, st) + [
            {"role": "assistant", "content": json.dumps(cleaned, ensure_ascii=False)}]
        split = "valid" if is_valid_hearing(h.package_id, valid_frac) else "train"
        rows[split].append(json.dumps({"messages": msgs}, ensure_ascii=False))
    for split, lines in rows.items():
        (out_dir / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
        setattr(stats, split, len(lines))
    return stats
