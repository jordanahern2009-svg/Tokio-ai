"""Refresh the bundled S&P 500 snapshot (tokio_ai/data/sp500.csv).

Index membership changes every quarter (additions/removals), so the bundled
snapshot will drift stale over time -- run this periodically to update it.

Usage:
    python -m scripts.refresh_sp500
"""

from __future__ import annotations

import csv
import json
import urllib.request
from datetime import date
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
DEST = Path(__file__).resolve().parent.parent / "src" / "tokio_ai" / "data" / "sp500.csv"
SCREENER_MODULE = Path(__file__).resolve().parent.parent / "src" / "tokio_ai" / "tools" / "screener.py"


def fetch_constituents() -> list[dict]:
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def to_yahoo_symbol(symbol: str) -> str:
    # Yahoo uses hyphens for class shares (BRK-B); this dataset uses dots (BRK.B).
    return symbol.replace(".", "-")


def main() -> None:
    rows = fetch_constituents()
    if len(rows) < 400:
        raise RuntimeError(f"got suspiciously few rows ({len(rows)}) -- source format may have changed, aborting")

    with DEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "name", "sector"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "symbol": to_yahoo_symbol(r["Symbol"]),
                "name": r["Security"],
                "sector": r["GICS Sector"],
            })

    today = date.today().isoformat()
    print(f"Wrote {len(rows)} constituents to {DEST}")
    print(f"Now update SNAPSHOT_DATE in {SCREENER_MODULE} to \"{today}\"")


if __name__ == "__main__":
    main()
