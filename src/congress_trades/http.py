"""Tiny polite HTTP helper: custom User-Agent, retries with backoff, per-host throttle."""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "Mozilla/5.0 (compatible; congress-trades/0.1; +https://github.com/rahulr-1006/congress-trades)"
_last_call: dict[str, float] = {}


def get(url: str, *, min_interval: float = 0.25, retries: int = 3, timeout: float = 30) -> bytes:
    host = urllib.parse.urlparse(url).netloc
    for attempt in range(retries + 1):
        wait = _last_call.get(host, 0) + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")
