"""Scoring rule: model output -> trade signal.

Not yet implemented. Planned rule (v0):
    score = sentiment.score * w_sent + direction_sign * w_policy * speaker_weight
where ``speaker_weight`` is higher for committee chairs / ranking members, and the signal is
BUY above +threshold, SELL below -threshold, HOLD otherwise; confidence = |score| clipped to 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

Action = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class Signal:
    ticker: str
    action: Action
    confidence: float
    created_at: datetime
    package_id: str
    statement_seq: int
    text_span: str
    model_output: dict[str, Any]


def score(entities: dict, sentiment: dict, policy: dict, speaker: str) -> list[Signal]:
    raise NotImplementedError("Scoring rule not implemented yet (see module docstring).")
