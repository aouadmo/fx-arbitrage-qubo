"""Public data: ECB euro foreign exchange reference rates.

The ECB publishes one reference rate per currency per business day, quoted as
units of currency per 1 EUR. Source and reuse terms:
https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html

These are mid rates, not tradable quotes. `quotes_from_mids` turns them into
bid/ask quotes with an explicit, configurable spread assumption.
"""

from __future__ import annotations

import csv
import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import CURRENCIES

ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "eurofxref-hist.zip"


@dataclass(frozen=True)
class RateTable:
    """Daily mid rates: `per_eur[t, k]` = units of currency k per 1 EUR on day t."""

    dates: list[str]          # ISO dates, ascending
    currencies: list[str]
    per_eur: np.ndarray       # shape (days, n_currencies), EUR column == 1.0

    def mid_matrix(self, t: int) -> np.ndarray:
        """mid[i, j] = price of 1 unit of currency i expressed in currency j."""
        r = self.per_eur[t]
        return r[None, :] / r[:, None]


def download(cache: Path = DEFAULT_CACHE, url: str = ECB_URL) -> Path:
    """Download the ECB history zip once and keep it in `cache`."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with urllib.request.urlopen(url, timeout=60) as resp:
            cache.write_bytes(resp.read())
    return cache


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path | None = None, currencies: list[str] = CURRENCIES,
         start: str | None = None, end: str | None = None) -> RateTable:
    """Parse the ECB CSV into a RateTable, keeping days where every currency has a rate."""
    path = path or download()
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        date = row["Date"]
        if (start and date < start) or (end and date > end):
            continue
        vals = []
        for c in currencies:
            if c == "EUR":
                vals.append(1.0)
                continue
            v = (row.get(c) or "").strip()
            if v in ("", "N/A"):
                break
            vals.append(float(v))
        else:
            rows.append((date, vals))
    rows.sort()
    if not rows:
        raise ValueError("no complete rows in the requested date range")
    return RateTable([d for d, _ in rows], list(currencies),
                     np.array([v for _, v in rows], dtype=float))


def quotes_from_mids(mid: np.ndarray, spread_bps: float | np.ndarray = 1.0):
    """Turn a mid-rate matrix into (bid, ask) matrices.

    `spread_bps` is the full bid-ask spread in basis points, either one number
    for every pair or an (n, n) symmetric matrix. Bid and ask sit half a spread
    either side of the mid. Diagonal entries are left at 1.0.
    """
    half = np.asarray(spread_bps, dtype=float) / 2e4
    bid = mid * (1.0 - half)
    ask = mid * (1.0 + half)
    np.fill_diagonal(bid, 1.0)
    np.fill_diagonal(ask, 1.0)
    return bid, ask
