"""Time-based dev / holdout split.

Llama 3.1's pretraining data ends around December 2023. On older hearings the model may
already "know" what happened to a company afterwards, which can leak into its stance scores.
So everything before the cutoff is dev (tune prompts, thresholds, fine-tune), and only
hearings after it are a true holdout. Run the holdout rarely; every look at it makes it a
little less held out.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from ..models import Hearing

MODEL_KNOWLEDGE_CUTOFF = date(2023, 12, 31)


def split(hearings: Iterable[Hearing], cutoff: date = MODEL_KNOWLEDGE_CUTOFF):
    dev, holdout = [], []
    for h in hearings:
        (dev if h.held_on <= cutoff else holdout).append(h)
    return dev, holdout
