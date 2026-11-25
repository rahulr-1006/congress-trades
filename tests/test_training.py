import json
from datetime import date

from congress_trades.llm.tasks import TASKS, split_combined
from congress_trades.market.tickers import Listing, MentionIndex
from congress_trades.models import Hearing, Statement
from congress_trades.store import Store
from congress_trades.training import dataset
from congress_trades.training.evaluate import Agreement

IDX = MentionIndex([Listing("NVDA", "NVIDIA Corporation - Common Stock", "Q")])


def test_combined_task_registered():
    assert "policy_signal" in TASKS["combined"].required_keys
    assert "JSON" in TASKS["combined"].system_prompt


def test_split_combined():
    parts = split_combined({"tickers": ["NVDA"], "stance": "hostile", "score": -0.5,
                            "policy_signal": True, "direction": "negative", "summary": "s"})
    assert parts["entities"]["tickers"] == ["NVDA"]
    assert parts["sentiment"]["score"] == -0.5
    assert parts["signal"]["direction"] == "negative"


def test_valid_split_is_deterministic_by_hearing():
    ids = [f"CHRG-118hhrg{i}" for i in range(400)]
    first = [dataset.is_valid_hearing(i, 0.15) for i in ids]
    assert first == [dataset.is_valid_hearing(i, 0.15) for i in ids]
    assert 0.08 < sum(first) / len(first) < 0.22


def test_clean_label_drops_ungrounded_tickers():
    out = dataset.clean_label({"tickers": ["NVDA", "MSFT"], "stance": "neutral"}, {"NVDA"})
    assert out["tickers"] == ["NVDA"] and out["stance"] == "neutral"


def test_build_writes_chat_jsonl(tmp_path):
    store = Store()
    h = Hearing("CHRG-118hhrg1", "Chips", "Commerce", "HOUSE", 118, date(2023, 3, 1))
    st = Statement(h.package_id, 0, "Mr. A", "Nvidia " + "x " * 120)
    store.save_hearing(h, [st], "raw")
    store.log_output(st, "combined", "teacher.gguf",
                     {"tickers": ["NVDA", "AMD"], "stance": "neutral"})
    stats = dataset.build(store, IDX, "teacher.gguf", tmp_path, valid_frac=0.0)
    assert (stats.train, stats.valid, stats.tickers_dropped) == (1, 0, 1)
    row = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert json.loads(row["messages"][2]["content"])["tickers"] == ["NVDA"]


def test_agreement_metrics():
    a = Agreement()
    a.add({"tickers": ["NVDA"], "stance": "hostile", "direction": "negative",
           "policy_signal": True},
          {"tickers": ["NVDA", "INTC"], "stance": "hostile", "direction": "none",
           "policy_signal": True})
    a.add({"tickers": [], "stance": "neutral"}, None)
    s = a.summary()
    assert s["n"] == 2 and s["valid_json"] == 0.5
    assert s["ticker_precision"] == 0.5 and s["ticker_recall"] == 1.0
    assert s["stance_acc"] == 0.5 and s["signal_f1"] == 1.0
