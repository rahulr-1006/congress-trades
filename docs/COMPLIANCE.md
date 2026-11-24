# Risk & compliance notes

**This project is for research and education. Nothing it produces is investment advice.**

## MNPI (material non-public information)

| Data source | Public? | Notes |
| --- | --- | --- |
| GovInfo hearing transcripts | Yes | Official GPO publications, released after the hearing. |
| Live hearing streams (planned) | Yes | Hearings are open sessions broadcast to the public. Closed/classified sessions are never ingested. |
| House PTR disclosures | Yes | Published by the Clerk of the House under the STOCK Act. |
| Market prices (planned) | Yes | Standard end-of-day / delayed market data. |

- The pipeline only consumes information released to the public at the same time as it is
  available to everyone else. It does not use leaks, private briefings, or tips from staff,
  members, or lobbyists, and must never be extended to do so.
- Trading on information from a member of Congress or staff that has not been made public may
  violate the STOCK Act and securities law, even if the trader isn't a government employee.
- Signals are logged with their source hearing and text span (see `model_outputs`), so the
  origin of every signal can be traced to a public record.

## Other risks
- **Model error:** LLM extraction can hallucinate tickers. Ticker mapping must be validated
  against an authoritative list (SEC `company_tickers.json`) before any signal is emitted.
- **Backtest bias:** transcripts are often published weeks after the hearing. Backtests must
  use the time the information was actually *available* (live stream time or transcript
  publish time), not the hearing date alone, to avoid lookahead.
