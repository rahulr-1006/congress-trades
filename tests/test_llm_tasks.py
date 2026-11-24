import json
from datetime import date

import pytest

from congress_trades.llm import TASKS, run_task
from congress_trades.llm.tasks import parse_json
from congress_trades.models import Hearing, Statement

HEARING = Hearing("CHRG-x", "Chips", "Subcommittee On Commerce", "HOUSE", 119, date(2025, 12, 16))
STMT = Statement("CHRG-x", 0, "Ms. Schakowsky", "Nvidia continues to ship chips abroad.")


class FakeModel:
    def __init__(self, reply: dict):
        self.reply, self.calls = reply, []

    def chat(self, system, user, *, max_tokens=1024):
        self.calls.append((system, user))
        return "Here you go: " + json.dumps(self.reply)


def test_every_task_has_a_prompt_file():
    for spec in TASKS.values():
        assert "JSON" in spec.system_prompt


def test_run_task_passes_context_and_parses_output():
    model = FakeModel({"companies": ["Nvidia"], "tickers": ["NVDA"], "sectors": ["Semis"]})
    out = run_task(model, "entities", HEARING, STMT)
    assert out["tickers"] == ["NVDA"]
    system, user = model.calls[0]
    assert system == TASKS["entities"].system_prompt
    assert "Speaker: Ms. Schakowsky" in user and "Committee: Subcommittee On Commerce" in user


def test_missing_keys_rejected():
    with pytest.raises(ValueError, match="missing keys"):
        parse_json('{"stance": "hostile"}', ("stance", "score"))
