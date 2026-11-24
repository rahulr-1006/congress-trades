from datetime import datetime

import pytest

from congress_trades.house import parse_ptrs
from congress_trades.llm import analyze, build_prompt, parse_response
from congress_trades.news import NewsItem


class FakeModel:
    def __init__(self, reply: str):
        self.reply = reply
        self.messages: list[dict] = []

    def create_chat_completion(self, messages, **kwargs):
        self.messages = messages
        return {"choices": [{"message": {"content": self.reply}}]}


def test_build_prompt_includes_filings_and_news(sample_xml):
    news = [NewsItem("Chip stocks rally", "Reuters", datetime(2025, 7, 1), "https://x")]
    prompt = build_prompt(list(parse_ptrs(sample_xml)), news)
    assert "Jane Doe (CA12)" in prompt
    assert "[Reuters] Chip stocks rally" in prompt


def test_parse_response_tolerates_surrounding_text():
    a = parse_response('Sure! {"summary": "s", "watchlist": [], "confidence": 0.4} done')
    assert a.summary == "s" and a.confidence == 0.4


def test_parse_response_rejects_non_json():
    with pytest.raises(ValueError):
        parse_response("no json here")


def test_analyze_with_fake_model(sample_xml):
    model = FakeModel('{"summary": "ok", "watchlist": [{"member": "Jane Doe", "reason": "2 PTRs"}],'
                      ' "confidence": 0.7}')
    result = analyze(model, list(parse_ptrs(sample_xml)))
    assert result.watchlist[0]["member"] == "Jane Doe"
    assert model.messages[0]["role"] == "system"
