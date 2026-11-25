# Requirements

Scope document for the project. Status markers: ✅ implemented · 🚧 stubbed with interface · ⬜ not started.

## 1. Goal
Detect tradeable signals from congressional hearing content (statements, tone, named
companies/sectors) before or as the market reacts.

## 2. Data ingestion
| Requirement | Status |
| --- | --- |
| Hearing transcripts from GovInfo (`CHRG` collection), keyless, with held + publish dates | ✅ `ingest/hearings.py` |
| Speaker identity per statement | ✅ |
| Committee, date, chamber | ✅ |
| Topic tags | ⬜ (planned: LLM task) |
| Live video/audio (C-SPAN, committee streams) → speech-to-text (whisper.cpp) for when transcripts lag | ⬜ |
| House PTR trade disclosures (secondary signal / validation) | ✅ `ingest/house_ptr.py` |

## 3. Local LLM inference (llama.cpp)
| Requirement | Status |
| --- | --- |
| Quantized open-weight model (Llama 3.x / Qwen 2.5 class, GGUF) via llama.cpp | ✅ `llm/engine.py` |
| Fully offline inference | ✅ |
| Entity extraction (companies, tickers, sectors) | ✅ `llm/prompts/entities.md` |
| Sentiment / stance classification | ✅ `llm/prompts/sentiment.md` |
| Policy / regulatory signal flagging + summary | ✅ `llm/prompts/signal.md` |
| Prompt specs separate from inference engine | ✅ `llm/tasks.py` (`TaskSpec`) |

## 4. Signal generation
| Requirement | Status |
| --- | --- |
| Map entities → tickers (grounded against listed universe) | ✅ `market/tickers.py` |
| Scoring rule → direction (baseline + LLM v0) | ✅ `signals/scoring.py` |
| Audit log: source hearing, text span, raw model output | ✅ `model_outputs` table; 🚧 signals table |

## 5. Backtesting
| Requirement | Status |
| --- | --- |
| Historical hearings paired with historical prices | ✅ `hearings backfill`, `market/prices.py` |
| Same pipeline live and in backtest; no lookahead | ✅ `backtest/event_study.py` |
| Returns, Sharpe, max drawdown, hit rate | ✅ `backtest/metrics.py` |

## 6. Infra / systems
- **Compute:** an 8B model at Q4_K_M needs ~5–6 GB RAM; runs on Apple Silicon via Metal or
  on CPU. Larger models (70B) need ~40 GB+.
- **Storage:** SQLite (`data/congress.db`) holds raw transcripts, parsed statements, every
  model output (with model name + task), and PTR filings.
- **Modularity:** models are swapped with `--model`; prompts are swapped by editing files in
  `llm/prompts/`; any object with a `chat()` method can replace the llama.cpp engine.
- **Systems/C++:** ⬜ C++ event-driven backtest core exposed via pybind11; ⬜ direct
  llama.cpp server integration with batched inference.

## 7. Risk / compliance
See [COMPLIANCE.md](COMPLIANCE.md).
