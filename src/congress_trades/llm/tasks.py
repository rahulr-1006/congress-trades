"""Extraction task specs. Each task = a prompt file + the JSON keys its output must contain."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from ..models import Hearing, Statement
from .engine import ChatModel


@dataclass(frozen=True)
class TaskSpec:
    name: str
    prompt_file: str
    required_keys: tuple[str, ...]

    @property
    def system_prompt(self) -> str:
        return resources.files(__package__).joinpath("prompts", self.prompt_file).read_text()


TASKS: dict[str, TaskSpec] = {
    t.name: t
    for t in (
        TaskSpec("entities", "entities.md", ("companies", "tickers", "sectors")),
        TaskSpec("sentiment", "sentiment.md", ("stance", "score")),
        TaskSpec("signal", "signal.md", ("policy_signal", "direction", "summary")),
    )
}


def format_statement(hearing: Hearing, st: Statement) -> str:
    return (
        f"Hearing: {hearing.title}\n"
        f"Committee: {hearing.committee} ({hearing.chamber})\n"
        f"Date: {hearing.held_on}\n"
        f"Speaker: {st.speaker}\n\n"
        f"Statement:\n{st.text}"
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


def run_task(model: ChatModel, task: str, hearing: Hearing, st: Statement) -> dict[str, Any]:
    spec = TASKS[task]
    raw = model.chat(spec.system_prompt, format_statement(hearing, st))
    return parse_json(raw, spec.required_keys)
