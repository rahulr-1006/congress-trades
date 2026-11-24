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
| `ingest/hearings.py` | ✅ | GovInfo client; splits transcripts into speaker-attributed statements |
| `ingest/house_ptr.py` | ✅ | House Clerk STOCK Act trade disclosures |
| `llm/engine.py` | ✅ | llama.cpp inference (Metal/CUDA offload, JSON-constrained output) |
| `llm/tasks.py` + `llm/prompts/` | ✅ | Extraction task specs, kept separate from the engine |
| `store.py` | ✅ | SQLite: raw text, statements, every model output (audit trail) |
| `backtest/metrics.py` | ✅ | Total return, Sharpe, max drawdown, hit rate |
| `signals/` | 🚧 | Entity → ticker mapping, scoring rule |
| `backtest/engine.py` | 🚧 | Point-in-time, lookahead-free replay of the live pipeline |

Full scope: [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) · Compliance: [docs/COMPLIANCE.md](docs/COMPLIANCE.md)

## Quickstart

```bash
git clone https://github.com/rahulr-1006/congress-trades.git
cd congress-trades
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"

export GOVINFO_API_KEY=...            # free: https://api.data.gov/signup

congress-trades hearings fetch --since 2026-01-01 --limit 10
congress-trades hearings list
```

```
saved CHRG-119hhrg64507  2025-12-16   110 statements  Subcommittee On Europe: HYBRID WARFARE IN EUROPE...
```

### Run the local model

Download any instruct-tuned GGUF (e.g. Llama 3.1 8B Instruct Q4_K_M, ~5 GB) into `models/`:

```bash
congress-trades extract CHRG-119hhrg64507 \
    --model models/llama-3.1-8b-instruct-q4_k_m.gguf \
    --tasks entities sentiment signal
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

Every output is written to the `model_outputs` table with the model name and task, so any
future signal can be traced back to the exact statement and model that produced it.

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

## Roadmap

- [x] Hearing transcript ingestion with speaker + committee attribution
- [x] llama.cpp inference layer with separate, swappable task prompts
- [x] Audit log of every model output
- [x] Performance metrics
- [ ] Entity → ticker resolution (SEC ticker list, sector ETFs)
- [ ] Scoring rule and signal table
- [ ] Historical price ingestion + point-in-time backtester
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
