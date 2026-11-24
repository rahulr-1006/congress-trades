# congress-trades

![CI](https://github.com/rahulr-1006/congress-trades/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![llama.cpp](https://img.shields.io/badge/LLM-llama.cpp-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A local-LLM pipeline that reads live U.S. congressional stock trade disclosures and market news,
then turns them into trading signals.

Members of Congress must disclose trades over $1,000 within 45 days under the
[STOCK Act](https://en.wikipedia.org/wiki/STOCK_Act). Those **Periodic Transaction Reports
(PTRs)** are public but buried in PDFs. This project ingests them continuously, has a quantized
model running on-device through [llama.cpp](https://github.com/ggml-org/llama.cpp) reason over
the filings alongside current news, and combines that read with quantitative features to produce
per-ticker predictions.

## Architecture

```
 House Clerk PTRs ─┐
 (Senate eFD)      ├──► ingest ──► SQLite ──► llama.cpp (local GGUF) ──► predict ──► signals
 News / RSS ───────┘                  │         structured JSON           ▲
                                      └──── quant features ───────────────┘
```

| Module | Status | Responsibility |
| --- | --- | --- |
| `house.py` | ✅ | Downloads the House Clerk's disclosure index and parses PTR filings |
| `store.py` | ✅ | SQLite persistence with idempotent upserts |
| `analysis.py` | ✅ | Filing statistics (most active filers, etc.) |
| `llm.py` | ✅ | llama.cpp wrapper: prompt building, local inference, structured JSON output |
| `news.py` | 🚧 | Headline ingestion per ticker/member |
| `predict.py` | 🚧 | Feature engineering + signal generation + backtesting |
| `cli.py` | ✅ | `ingest`, `top`, `recent`, `analyze` |

## Quickstart

```bash
git clone https://github.com/rahulr-1006/congress-trades.git
cd congress-trades
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

congress-trades ingest --year 2025 2026   # pull PTR filings from the House Clerk
congress-trades top --limit 5             # most active filers
congress-trades recent --limit 5          # latest filings + source PDF links
```

```
  #  Member                  Dist   PTRs  Last filed
  1  Marjorie Taylor Greene  GA14     25  2025-12-16
  2  Rob Bresnahan           PA08     18  2025-12-19
  3  Suzan K. DelBene        WA01     15  2025-12-08
  4  Kelly Louise Morrison   MN03     14  2025-12-19
  5  Harold Dallas Rogers    KY05     12  2026-01-16
```

### Local LLM analysis

```bash
pip install -e ".[llm]"   # llama-cpp-python (Metal-accelerated on Apple Silicon)
# drop any instruct-tuned GGUF into models/, e.g. Qwen2.5-7B-Instruct Q4_K_M
congress-trades analyze --model models/qwen2.5-7b-instruct-q4_k_m.gguf
```

Everything runs on-device: no API keys, no rate limits, no data sent to a third party.

## Roadmap

- [x] House PTR ingestion and storage
- [x] Most-active-filer leaderboard
- [x] llama.cpp inference layer with structured output
- [ ] Parse individual transactions (ticker, buy/sell, amount range) out of PTR PDFs
- [ ] Scheduled polling for new filings (live feed)
- [ ] News ingestion and ticker/member entity linking
- [ ] Senate eFD ingestion
- [ ] Quant features: disclosure lag, cluster buying, committee overlap, momentum
- [ ] Signal model + walk-forward backtest vs. SPY with realistic disclosure delays
- [ ] Dashboard

## Development

```bash
ruff check .
pytest
```

Tests use a fake model, so CI never needs to download weights.

## Data sources

- [U.S. House Clerk: Financial Disclosure Reports](https://disclosures-clerk.house.gov/FinancialDisclosure)

For research and educational purposes only. Not investment advice.

## License

MIT
