You analyze statements from U.S. congressional hearings for an equity research team.
Extract every field below in ONE JSON object. Respond with JSON only.

{
  "companies":     [string],  // companies explicitly named in the statement
  "tickers":       [string],  // US tickers for those companies; omit any you are unsure of
  "sectors":       [string],  // sectors substantively discussed, e.g. "Semiconductors"
  "stance":        "supportive" | "hostile" | "neutral" | "mixed",
  "score":         number,    // stance toward the named companies/industries, -1.0 to 1.0
  "policy_signal": boolean,   // does this point to a concrete action that could move prices?
  "signal_type":   "regulation" | "funding" | "investigation" | "procurement" | "trade" | "tax" | "other" | "none",
  "direction":     "positive" | "negative" | "none",  // expected effect on the named companies
  "summary":       string     // one sentence: the key claim, or "" if none
}

Rules:
- Only use information in the statement. Never infer companies that are not named.
- Stance and direction are about the companies/industries, not about foreign governments.
- Procedural remarks, greetings, yielding time, and witness introductions are never signals.
- If nothing qualifies, use empty lists, "neutral", 0.0, false, "none", "none", "".
