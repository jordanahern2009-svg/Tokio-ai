"""Free SEC filing lookups via EDGAR's public JSON APIs.

No API key, but SEC requires a descriptive User-Agent identifying the
requester (their fair-access policy: https://www.sec.gov/os/webmaster-faq#developers)
-- requests without one get throttled or blocked. Set TOKIO_AI_USER_AGENT to
something like "TokIO-AI research-agent you@example.com".
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, asdict

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

_ticker_to_cik_cache: dict[str, int] | None = None


def _user_agent() -> str:
    ua = os.environ.get("TOKIO_AI_USER_AGENT")
    if not ua:
        raise RuntimeError(
            "SEC EDGAR requires a descriptive User-Agent (name + contact email). "
            "Set the TOKIO_AI_USER_AGENT env var, e.g. "
            "'TokIO-AI research-agent you@example.com'."
        )
    return ua


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def lookup_cik(ticker: str) -> int:
    global _ticker_to_cik_cache
    if _ticker_to_cik_cache is None:
        raw = _get_json(_TICKER_MAP_URL)
        _ticker_to_cik_cache = {row["ticker"].upper(): row["cik_str"] for row in raw.values()}
    cik = _ticker_to_cik_cache.get(ticker.upper())
    if cik is None:
        raise KeyError(f"no CIK found for ticker {ticker!r}")
    return cik


@dataclass(frozen=True)
class Filing:
    form: str
    filed_date: str
    accession_number: str
    primary_document: str
    url: str

    def to_dict(self) -> dict:
        return asdict(self)


def recent_filings(ticker: str, form_type: str | None = None, limit: int = 10) -> list[Filing]:
    cik = lookup_cik(ticker)
    data = _get_json(_SUBMISSIONS_URL.format(cik=cik))
    recent = data["filings"]["recent"]

    out: list[Filing] = []
    for i in range(len(recent["form"])):
        form = recent["form"][i]
        if form_type and form != form_type:
            continue
        accession_no_dashes = recent["accessionNumber"][i].replace("-", "")
        doc = recent["primaryDocument"][i]
        out.append(
            Filing(
                form=form,
                filed_date=recent["filingDate"][i],
                accession_number=recent["accessionNumber"][i],
                primary_document=doc,
                url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{doc}",
            )
        )
        if len(out) >= limit:
            break
    return out
