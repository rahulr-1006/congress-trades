# congress-trades

![CI](https://github.com/rahulr-1006/congress-trades/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![llama.cpp](https://img.shields.io/badge/inference-llama.cpp-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Trading signals from U.S. congressional hearings, extracted by a **local LLM running on
llama.cpp**.

When a committee chair grills a CEO, floats new regulation, or pushes for a funding bill,
markets can react. This project ingests hearing transcripts, has an on-device model extract
companies, stance, and policy signals from every statement, and turns those into auditable
BUY/SELL/HOLD signals that can be backtested against price data.

## Pipeline

```
 GovInfo hearing transcripts ─┐                         ┌─ entities  (companies, tickers, sectors)
 (live audio → whisper.cpp)   ├─► parse speakers ─► llama.cpp ─┼─ sentiment (stance, score)
 House PTR disclosures ───────┘   & committee     (local GGUF) └─ signal    (policy/regulatory)
                                        │                                  │
                                        ▼                                  ▼
                                 SQLite: transcripts, statements,  ──► scoring ──► signals ──► backtest
                                 model outputs (audit log)               ticker mapping       Sharpe / DD / hit rate
```

| Module | Status | Responsibility |
| --- | --- | --- |
| `ingest/hearings.py` | ✅ | GovInfo transcripts + MODS metadata (held date, publish date, committee); no API key |
| `ingest/house_ptr.py` | ✅ | House Clerk STOCK Act trade disclosures |
| `market/tickers.py` | ✅ | 5.5k US-listed stocks; O(n) n-gram company-mention pre-filter |
| `market/prices.py` | ✅ | Daily adjusted closes, cached in SQLite |
| `llm/engine.py` | ✅ | llama.cpp inference (Metal/CUDA offload, JSON-constrained output, token usage) |
| `llm/tasks.py` + `llm/prompts/` | ✅ | Extraction task specs, kept separate from the engine |
| `signals/scoring.py` | ✅ | Mention baseline + LLM scorer (LLM tickers must be grounded in the text) |
| `backtest/event_study.py` | ✅ | Market-adjusted returns at 1/5/20 days, t-stats, hit rate; no lookahead |
| `backtest/splits.py` | ✅ | Dev/holdout split at the model's knowledge cutoff |
| `store.py` | ✅ | SQLite: transcripts, statements, model outputs + latency (audit trail), prices |
| Fine-tuning | 🚧 | LoRA fine-tune → merge → GGUF → quantization sweep ([plan](docs/FINETUNING.md)) |

Full scope: [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) · Compliance: [docs/COMPLIANCE.md](docs/COMPLIANCE.md)

## Quickstart

```bash
git clone https://github.com/rahulr-1006/congress-trades.git
cd congress-trades
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"

congress-trades tickers sync                         # US-listed stock universe
congress-trades hearings backfill --year 2023 --limit 25   # market-relevant committees
congress-trades hearings list
congress-trades mentions CHRG-118hhrg52357           # which companies came up?
```

```
saved CHRG-118hhrg52357  held 2023-02-07  published 2023-09-06   454 stmts  Financial Services: COMBATTING THE ECONOMIC THREAT FROM CHINA
[  8] Chairman McHenry   {'LRCX': 'LAM RESEARCH'}
[124] Mr. Hill           {'INTC': 'Intel'}
[168] Mr. Foster         {'TSM': 'TSMC'}
[320] Mr. Torres         {'META': 'Facebook'}
```

No API keys anywhere: transcripts come from GovInfo's public content/sitemap endpoints, ticker
lists from Nasdaq Trader, prices from Yahoo's public chart endpoint.

### Run the local model

Download any instruct-tuned GGUF (e.g. Llama 3.1 8B Instruct Q4_K_M, ~5 GB) into `models/`:

```bash
congress-trades extract CHRG-119hhrg64507 \
    --model models/llama-3.1-8b-instruct-q4_k_m.gguf \
    --tasks entities sentiment signal
congress-trades latency     # p50/p95 latency and tokens/sec per model and task
```

Real output from Llama 3.1 8B (Q4_K_M) on an M1 Pro, fully offline, on a Dec 2025 House
Foreign Affairs subcommittee hearing:

```
[  1] Mr. Keating  entities  {'companies': ['Nvidia'], 'tickers': ['NVDA'], 'sectors': ['Semiconductors']}
[  1] Mr. Keating  sentiment {'stance': 'hostile', 'score': -0.85, 'target': "Russia and China's hybrid warfare in Europe"}
[  1] Mr. Keating  signal    {'policy_signal': True, 'signal_type': 'regulation', 'direction': 'negative',
                              'affected': ['Nvidia'], 'summary': 'The White House approved the export of
                              advanced Nvidia H200 chips to China ...'}
[  2] Mr. Self     signal    {'policy_signal': False, 'signal_type': 'none', 'direction': 'none',
                              'affected': [], 'summary': 'None, procedural remarks and greetings'}
```

Only statements that mention a listed company are sent to the model (`--no-prefilter` to
disable). On a typical hearing that's ~2% of statements, which turns a ~30 min hearing into
under a minute. Every output is written to `model_outputs` with the model, task, latency, and
token counts, so any signal can be traced to the exact statement and model that produced it.

Already have the model in Ollama? Its blobs are GGUF files, so you can reuse one without
downloading anything:

```bash
ollama show llama3.1 --modelfile | grep ^FROM      # prints the blob path
ln -s <that path> models/llama-3.1-8b-instruct-q4_k_m.gguf
```

**Hardware:** 8B @ Q4_K_M needs ~6 GB RAM and runs well on Apple Silicon (Metal) or a
consumer GPU. No internet needed after the model is downloaded.

### Swapping models and prompts

- Model: pass a different `--model` path.
- Prompts: edit `src/congress_trades/llm/prompts/*.md`. Add a task by registering a `TaskSpec`.
- Engine: anything with a `chat(system, user) -> str` method works (that's how the tests run
  without weights).

## Validation

```bash
congress-trades validate --scorer baseline --timing published --split dev
congress-trades validate --scorer llm      --timing published --split dev --model <file>.gguf
```

Each signal becomes an event; the event study measures the stock's return minus SPY's, signed
by the signal's direction, from the first close *after* the decision date.

Design choices that keep the numbers honest:

- **Two timing modes.** `held` trades on the hearing date (upper bound: assumes live
  processing). `published` trades when the transcript hit GovInfo, which is what a
  transcript-only system can actually do. The gap is large: 2023 transcripts appeared
  1.5 to 18 months after the hearing.
- **Knowledge-cutoff split.** Llama 3.1's training data runs through ~Dec 2023, so it may
  "remember" how older stories played out. Hearings up to the cutoff are dev; later ones are
  the holdout.
- **Baselines first.** `baseline` goes long every mentioned company. The LLM has to beat it.
- **Grounding.** LLM tickers only count if the deterministic pre-filter also found that
  company in the statement, which drops hallucinated tickers.
- **Survivorship is reported.** Delisted companies have no prices; the report says how many
  events were skipped.

First baseline run (23 hearings from 2023, 96 events, `published` timing):

```
 horizon     n   mean CAR   t-stat  hit rate
     1d     96     -0.14%    -0.82    41.7%
     5d     96     -0.07%    -0.16    58.3%
    20d     96      0.02%     0.03    49.0%
```

No edge from mentions alone, which is the expected null result and the bar to beat.

## Roadmap

- [x] Hearing transcript ingestion with speaker, committee, held + publish dates
- [x] Ticker universe, mention pre-filter, price cache
- [x] llama.cpp inference layer with separate, swappable task prompts
- [x] Audit log of every model output with latency and token usage
- [x] Event-study validation with baseline, timing modes, and cutoff-aware split
- [ ] Hand-labeled gold set for extraction accuracy (precision/recall per task)
- [ ] LoRA fine-tune of Llama 3 → GGUF → quantization sweep (accuracy vs. latency vs. memory), see [docs/FINETUNING.md](docs/FINETUNING.md)
- [ ] Live hearing audio → text with whisper.cpp
- [ ] C++ event-driven backtest core (pybind11)
- [ ] Cross-reference signals with members' own PTR trades

## Development

```bash
ruff check .
pytest
```

## License

MIT. For research and educational purposes only; not investment advice.
