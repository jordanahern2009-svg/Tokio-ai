"""Rank a universe of stocks by trailing return -- the "I don't know what to
ask about" entry point, for someone who wants "the best performing stocks"
without already having a ticker or sector in mind.

Universe is a bundled snapshot of the real S&P 500 constituents and their
real GICS sectors (see data/sp500.csv, snapshotted from a public dataset on
SNAPSHOT_DATE below). This exists specifically so tickers and sector
membership are never invented from an LLM's own memory -- that's exactly
the kind of quiet, plausible-looking hallucination this whole project is
built to avoid. Index membership drifts over time (companies get added and
dropped every quarter); this snapshot will go stale eventually and should
be refreshed periodically.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from .prices import DailyBar, fetch_daily_bars

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "sp500.csv"
SNAPSHOT_DATE = "2026-08-02"

_TRADING_DAYS = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252}
VALID_PERIODS = tuple(_TRADING_DAYS) + ("ytd",)
DEFAULT_MAX_WORKERS = 20
MIN_YIELD_WARNING = 0.90  # below this fraction of the universe returning data, flag it


def load_universe(sector: str | None = None) -> list[dict]:
    with DATA_FILE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if sector:
        needle = sector.lower()
        rows = [r for r in rows if needle in r["sector"].lower()]
    return rows


def _return_over_period(bars: list[DailyBar], period: str) -> float | None:
    if not bars:
        return None
    if period == "ytd":
        year_start = f"{date.today().year}-01-01"
        eligible = [b for b in bars if b.date >= year_start]
        if not eligible:
            return None
        start_close = eligible[0].close
    else:
        n = _TRADING_DAYS.get(period)
        if n is None or len(bars) <= n:
            return None
        start_close = bars[-1 - n].close
    if start_close == 0:
        return None
    return bars[-1].close / start_close - 1


def _fetch_one(symbol: str, period: str) -> float | None:
    try:
        # Always pull 2y regardless of the requested period: a "1y" lookback
        # needs a bar from *before* that window too (start_close is at
        # index -1-252), so fetching exactly 1y leaves zero margin and
        # silently returns no data for every symbol.
        bars = fetch_daily_bars(symbol, range_="2y")
    except Exception:
        return None
    return _return_over_period(bars, period)


def top_performers(
    sector: str | None = None,
    period: str = "3mo",
    top_n: int = 10,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict:
    if period not in VALID_PERIODS:
        raise ValueError(f"period must be one of {VALID_PERIODS}, got {period!r}")
    if top_n <= 0:
        # results[:top_n] with a negative top_n silently returns "all but
        # the last N" (Python slice semantics), not an error and not what
        # anyone asking for "the top N" means -- fail loud instead.
        raise ValueError(f"top_n must be positive, got {top_n!r}")

    universe = load_universe(sector)
    if not universe:
        return {
            "error": f"no S&P 500 constituents matched sector {sector!r}",
            "valid_sectors": sorted({r["sector"] for r in load_universe()}),
            "results": [],
        }

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_row = {pool.submit(_fetch_one, row["symbol"], period): row for row in universe}
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            ret = future.result()
            if ret is not None:
                results.append({"symbol": row["symbol"], "name": row["name"], "sector": row["sector"], "return": ret})

    results.sort(key=lambda r: r["return"], reverse=True)
    yield_ratio = len(results) / len(universe)
    out = {
        "universe": "S&P 500" if not sector else f"S&P 500 / {sector}",
        "universe_snapshot_date": SNAPSHOT_DATE,
        "universe_size": len(universe),
        "symbols_with_data": len(results),
        "period": period,
        "top": results[:top_n],
    }
    if yield_ratio < MIN_YIELD_WARNING:
        # _fetch_one swallows all per-symbol failures the same way (network
        # blip, delisting, or the granularity guard in prices.py silently
        # firing across many tickers at once) -- a low yield doesn't say
        # which, but it should be impossible to miss rather than something
        # the caller has to notice by comparing two numbers themselves.
        out["warning"] = (
            f"Only {len(results)}/{len(universe)} symbols returned data "
            f"({yield_ratio:.0%}) -- normally ~99% succeed. Results may be "
            f"skewed; treat this ranking with reduced confidence."
        )
    return out
