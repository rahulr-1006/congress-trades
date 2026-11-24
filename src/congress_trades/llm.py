"""Local LLM analysis via llama.cpp.

Runs a quantized GGUF model on-device through ``llama-cpp-python``: no API keys and no data
leaves the machine. The model reads recent filings (and news, once wired up) and returns
structured JSON that downstream prediction code can consume.

Install with ``pip install -e ".[llm]"`` and pass ``--model path/to/model.gguf``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .models import Filing
from .news import NewsItem

SYSTEM_PROMPT = """You are a financial analyst studying stock trades disclosed by members of \
the U.S. Congress under the STOCK Act. Given recent filings and news, identify patterns worth \
investigating (clusters of activity, committee-relevant timing, unusual filers). Respond with \
JSON only: {"summary": str, "watchlist": [{"member": str, "reason": str}], "confidence": float}"""


class ChatModel(Protocol):
    def create_chat_completion(self, messages: list[dict], **kwargs: Any) -> dict: ...


@dataclass(frozen=True)
class Analysis:
    summary: str
    watchlist: list[dict]
    confidence: float


def build_prompt(filings: Sequence[Filing], news: Sequence[NewsItem] = ()) -> str:
    lines = ["Recent Periodic Transaction Reports:"]
    lines += [f"- {f.filing_date} {f.name} ({f.state_district}) doc={f.doc_id}" for f in filings]
    if news:
        lines.append("\nRecent news:")
        lines += [f"- {n.published:%Y-%m-%d} [{n.source}] {n.title}" for n in news]
    return "\n".join(lines)


def parse_response(text: str) -> Analysis:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model did not return JSON: {text[:200]!r}")
    data = json.loads(text[start : end + 1])
    return Analysis(
        summary=str(data.get("summary", "")),
        watchlist=list(data.get("watchlist", [])),
        confidence=float(data.get("confidence", 0.0)),
    )


def load_model(model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1) -> ChatModel:
    """Load a GGUF model with llama.cpp (Metal on Apple Silicon when n_gpu_layers=-1)."""
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise SystemExit('llama-cpp-python not installed. Run: pip install -e ".[llm]"') from e
    return Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)


def analyze(
    model: ChatModel, filings: Sequence[Filing], news: Sequence[NewsItem] = ()
) -> Analysis:
    resp = model.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(filings, news)},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return parse_response(resp["choices"][0]["message"]["content"])
