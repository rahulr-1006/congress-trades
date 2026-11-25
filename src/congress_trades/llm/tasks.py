"""Extraction task specs. Each task = a prompt file + the JSON keys its output must contain."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from ..models import Hearing, Statement
from .engine import ChatModel

_STR_LIST = {"type": "array", "items": {"type": "string"}, "maxItems": 8}  # bounds runaway lists
_FIELDS: dict[str, dict] = {
    "companies": _STR_LIST,
    "tickers": _STR_LIST,
    "sectors": _STR_LIST,
    "affected": _STR_LIST,
    "stance": {"type": "string", "enum": ["supportive", "hostile", "neutral", "mixed"]},
    "score": {"type": "number"},
    "target": {"type": "string"},
    "policy_signal": {"type": "boolean"},
    "signal_type": {"type": "string"},
    "direction": {"type": "string", "enum": ["positive", "negative", "none"]},
    "summary": {"type": "string"},
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    prompt_file: str
    required_keys: tuple[str, ...]
    extra_keys: tuple[str, ...] = ()

    @property
    def json_schema(self) -> dict:
        """Schema for grammar-constrained decoding: llama.cpp compiles it to a GBNF grammar,
        so every required key is guaranteed to be present with the right type."""
        keys = self.required_keys + self.extra_keys
        return {"type": "object", "properties": {k: _FIELDS[k] for k in keys},
                "required": list(keys)}

    @property
    def system_prompt(self) -> str:
        return resources.files(__package__).joinpath("prompts", self.prompt_file).read_text()


TASKS: dict[str, TaskSpec] = {
    t.name: t
    for t in (
        TaskSpec("entities", "entities.md", ("companies", "tickers", "sectors")),
        TaskSpec("sentiment", "sentiment.md", ("stance", "score"), ("target",)),
        TaskSpec("signal", "signal.md", ("policy_signal", "direction", "summary"),
                 ("signal_type", "affected")),
        # All three in one call: shares one prefill, ~1/3 the calls. Used for distillation.
        TaskSpec("combined", "combined.md", ("companies", "tickers", "sectors", "stance", "score",
                                             "policy_signal", "signal_type", "direction",
                                             "summary")),
    )
}
MAX_STATEMENT_CHARS = 6000  # ~1.5k tokens; keeps prompts inside small context windows


def format_statement(hearing: Hearing, st: Statement) -> str:
    return (
        f"Hearing: {hearing.title}\n"
        f"Committee: {hearing.committee} ({hearing.chamber})\n"
        f"Date: {hearing.held_on}\n"
        f"Speaker: {st.speaker}\n\n"
        f"Statement:\n{st.text[:MAX_STATEMENT_CHARS]}"
    )


def parse_json(text: str, required: tuple[str, ...]) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model output is not JSON: {text[:200]!r}")
    data = json.loads(text[start : end + 1])
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Model output missing keys {missing}: {data}")
    return data


def messages_for(task: str, hearing: Hearing, st: Statement) -> list[dict[str, str]]:
    return [{"role": "system", "content": TASKS[task].system_prompt},
            {"role": "user", "content": format_statement(hearing, st)}]


def run_task(
    model: ChatModel, task: str, hearing: Hearing, st: Statement, constrained: bool = True
) -> dict[str, Any]:
    spec = TASKS[task]
    schema = spec.json_schema if constrained else None
    raw = model.chat(spec.system_prompt, format_statement(hearing, st), schema=schema)
    return parse_json(raw, spec.required_keys)


def split_combined(out: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """View a combined output as the three single-task outputs (for scoring/eval)."""
    return {
        "entities": {k: out.get(k, []) for k in ("companies", "tickers", "sectors")},
        "sentiment": {"stance": out.get("stance"), "score": out.get("score", 0)},
        "signal": {k: out.get(k) for k in ("policy_signal", "signal_type", "direction",
                                           "summary")},
    }
