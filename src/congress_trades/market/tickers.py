"""US-listed ticker universe + fast company-name mention detection.

Source: Nasdaq Trader symbol directory (all Nasdaq + NYSE/NYSE American/Arca listings, no key).
Caveat: this is the *current* listing set, so companies that were later delisted or acquired
are missing (survivorship bias); backtests report how many mentions couldn't be mapped.

Mention detection is a cheap pre-filter that decides which statements are worth spending LLM
time on. It favors recall: false positives just cost an LLM call; misses cost a signal.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from importlib import resources

from .. import http

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

TOKEN_RE = re.compile(r"[A-Za-z0-9&]+(?:['.\-][A-Za-z0-9&]+)*")
SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd", "limited",
    "plc", "llc", "lp", "l.p", "sa", "s.a", "nv", "n.v", "ag", "se", "holdings", "holding",
    "group", "the", "common", "stock", "shares", "ordinary", "class", "a", "b", "c",
}
SKIP_WORDS = ("warrant", "right", "unit", "preferred", "notes", "debenture", "%", "depositary")
# Single-word company names that are also everyday words in hearings.
STOPNAMES = {
    "target", "visa", "ally", "block", "gap", "match", "progressive", "southern", "general",
    "united", "american", "first", "energy", "insight", "unity", "snap", "square", "shift",
    "liberty", "frontier", "national", "republic", "global", "international", "pioneer",
    "public", "service", "trust", "union", "federal", "security", "capital", "community",
    "state", "states", "america", "data", "health", "care", "financial", "realty", "bank",
    "evolution", "progress", "opportunity", "freedom", "patriot", "veterans", "trade",
    "commerce", "chair", "member", "members", "act", "congress", "senate", "house", "china",
    "india", "mexico", "canada", "europe", "texas", "florida", "york", "california", "ohio",
    "u.s", "us", "one", "new", "best", "safety", "vision", "mission", "summit", "pacific",
    "atlantic", "central", "western", "eastern", "northern", "core", "nuclear", "solar",
    "water", "power", "oil", "gas", "steel", "home", "life", "credit", "post", "times",
    "world", "online", "digital", "smart", "clear", "bright", "standard", "focus", "impact",
}
# Multi-word names that are generic phrases in hearing language, plus common first names.
GENERIC_PHRASES = {
    "financial institutions", "community bankers", "national security", "first national",
    "american national", "federal reserve", "capital markets", "public storage",
    "dave", "john", "mike", "bob", "tom", "jim", "bill", "joe", "steve", "mark",
}
ALIASES = {
    "google": "GOOGL", "youtube": "GOOGL", "facebook": "META", "instagram": "META",
    "whatsapp": "META", "jpmorgan": "JPM", "jp morgan": "JPM", "goldman sachs": "GS",
    "goldman": "GS", "morgan stanley": "MS", "citi": "C", "citigroup": "C", "citibank": "C",
    "bank of america": "BAC", "wells fargo": "WFC", "exxon": "XOM", "exxonmobil": "XOM",
    "lockheed": "LMT", "lockheed martin": "LMT", "raytheon": "RTX", "northrop": "NOC",
    "general dynamics": "GD", "boeing": "BA", "nvidia": "NVDA", "microsoft": "MSFT",
    "apple": "AAPL", "amazon": "AMZN", "tesla": "TSLA", "intel": "INTC", "micron": "MU",
    "tsmc": "TSM", "pfizer": "PFE", "moderna": "MRNA", "unitedhealth": "UNH",
    "berkshire": "BRK.B", "chevron": "CVX", "palantir": "PLTR", "coinbase": "COIN",
    "oracle": "ORCL", "ford": "F", "caterpillar": "CAT", "honeywell": "HON",
}


@cache
def common_words() -> frozenset[str]:
    """Single-word company names that are also dictionary words (see common_words.txt)."""
    text = resources.files(__package__).joinpath("common_words.txt").read_text()
    return frozenset(ln for ln in text.splitlines() if ln and not ln.startswith("#"))


TITLES = {"mr", "ms", "mrs", "dr", "chairman", "chairwoman", "chair", "senator", "sen",
          "representative", "rep", "congressman", "congresswoman", "secretary", "general",
          "admiral", "ambassador", "judge", "governor", "member", "leader", "director"}


@dataclass(frozen=True)
class Listing:
    symbol: str
    name: str
    exchange: str


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def clean_name(security_name: str) -> str:
    """'NVIDIA Corporation - Common Stock' -> 'nvidia'."""
    base = security_name.split(" - ")[0]
    toks = [t.lower().strip(".,") for t in _tokens(base)]
    while toks and toks[-1] in SUFFIXES:
        toks.pop()
    return " ".join(toks)


def parse_symbol_file(text: str, symbol_col: str, exchange: str | None) -> list[Listing]:
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
    header = lines[0].split("|")
    out = []
    for ln in lines[1:]:
        row = dict(zip(header, ln.split("|"), strict=False))
        name = row.get("Security Name", "")
        if row.get("ETF") == "Y" or row.get("Test Issue") == "Y":
            continue
        if any(w in name.lower() for w in SKIP_WORDS):
            continue
        out.append(Listing(row[symbol_col], name, exchange or row.get("Exchange", "")))
    return out


def fetch_listings() -> list[Listing]:
    nasdaq = parse_symbol_file(http.get(NASDAQ_URL).decode(), "Symbol", "Q")
    other = parse_symbol_file(http.get(OTHER_URL).decode(), "ACT Symbol", None)
    return nasdaq + other


class MentionIndex:
    """n-gram lookup from cleaned company names to tickers. O(tokens) per statement."""

    MAX_N = 5

    def __init__(self, listings: Iterable[Listing]):
        self.index: dict[tuple[str, ...], str] = {}
        for lst in listings:
            name = clean_name(lst.name)
            key = tuple(name.split())
            if not key or len(name) < 4 or len(key) > self.MAX_N:
                continue
            if len(key) == 1 and (key[0] in STOPNAMES or key[0] in common_words()):
                continue
            if name in GENERIC_PHRASES:
                continue
            # Prefer the first listing seen (e.g. GOOGL before GOOG) and shortest symbols.
            if key not in self.index or len(lst.symbol) < len(self.index[key]):
                self.index[key] = lst.symbol
        self.aliases = {tuple(a.split()) for a in ALIASES}
        for alias, sym in ALIASES.items():
            self.index[tuple(alias.split())] = sym

    def find(self, text: str, exclude: Iterable[str] = ()) -> dict[str, str]:
        """Return {ticker: matched text} for company names mentioned in ``text``.

        ``exclude``: lowercase single words to ignore, e.g. the surnames of the hearing's
        speakers (so "Rogers" the chairman isn't read as Rogers Corp).
        """
        excluded = set(exclude)
        toks = _tokens(text)
        low = [t.lower().strip(".") for t in toks]
        hits: dict[str, str] = {}
        i = 0
        while i < len(toks):
            for n in range(min(self.MAX_N, len(toks) - i), 0, -1):
                key = tuple(low[i : i + n])
                # Single all-caps words are usually acronyms (STEM, CHIPS), not companies.
                single = n == 1 and key not in self.aliases
                acronym = single and toks[i].isupper()
                person = single and (key[0] in excluded or (i > 0 and low[i - 1] in TITLES))
                if key in self.index and toks[i][:1].isupper() and not acronym and not person:
                    hits.setdefault(self.index[key], " ".join(toks[i : i + n]))
                    i += n
                    break
            else:
                i += 1
        return hits


def speaker_surnames(speakers: Iterable[str]) -> set[str]:
    """'Mr. Rogers', 'Chairman Rogers' -> {'rogers'}."""
    return {sp.split()[-1].lower() for sp in speakers if sp}
