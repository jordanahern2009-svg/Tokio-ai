import os

from tokio_ai.tools import filings


def test_user_agent_missing_raises():
    original = os.environ.pop("TOKIO_AI_USER_AGENT", None)
    try:
        filings._user_agent()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    finally:
        if original is not None:
            os.environ["TOKIO_AI_USER_AGENT"] = original


def test_user_agent_present():
    original = os.environ.get("TOKIO_AI_USER_AGENT")
    os.environ["TOKIO_AI_USER_AGENT"] = "test-agent 1.0"
    try:
        assert filings._user_agent() == "test-agent 1.0"
    finally:
        if original is None:
            os.environ.pop("TOKIO_AI_USER_AGENT", None)
        else:
            os.environ["TOKIO_AI_USER_AGENT"] = original


def test_lookup_cik_finds_ticker_case_insensitively():
    filings._ticker_to_cik_cache = None
    original_get_json = filings._get_json
    filings._get_json = lambda url: {
        "0": {"ticker": "AAPL", "cik_str": 320193},
        "1": {"ticker": "MSFT", "cik_str": 789019},
    }
    try:
        assert filings.lookup_cik("aapl") == 320193
        assert filings.lookup_cik("MSFT") == 789019
    finally:
        filings._get_json = original_get_json
        filings._ticker_to_cik_cache = None


def test_lookup_cik_caches_after_first_call():
    filings._ticker_to_cik_cache = None
    call_count = {"n": 0}

    def fake_get_json(url):
        call_count["n"] += 1
        return {"0": {"ticker": "AAPL", "cik_str": 320193}}

    original_get_json = filings._get_json
    filings._get_json = fake_get_json
    try:
        filings.lookup_cik("AAPL")
        filings.lookup_cik("AAPL")
        assert call_count["n"] == 1
    finally:
        filings._get_json = original_get_json
        filings._ticker_to_cik_cache = None


def test_lookup_cik_unknown_ticker_raises():
    filings._ticker_to_cik_cache = None
    original_get_json = filings._get_json
    filings._get_json = lambda url: {"0": {"ticker": "AAPL", "cik_str": 320193}}
    try:
        try:
            filings.lookup_cik("ZZZZ")
            assert False, "expected KeyError"
        except KeyError:
            pass
    finally:
        filings._get_json = original_get_json
        filings._ticker_to_cik_cache = None


def test_recent_filings_filters_by_form_type_and_builds_correct_url():
    filings._ticker_to_cik_cache = {"AAPL": 320193}
    fake_submissions = {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "10-Q"],
                "filingDate": ["2025-11-01", "2025-11-15", "2025-08-01"],
                "accessionNumber": ["0000320193-25-000079", "0000320193-25-000090", "0000320193-25-000050"],
                "primaryDocument": ["aapl-10k.htm", "aapl-8k.htm", "aapl-10q.htm"],
            }
        }
    }
    original_get_json = filings._get_json
    filings._get_json = lambda url: fake_submissions
    try:
        results = filings.recent_filings("AAPL", form_type="10-K", limit=10)
    finally:
        filings._get_json = original_get_json
        filings._ticker_to_cik_cache = None

    assert len(results) == 1
    f = results[0]
    assert f.form == "10-K"
    assert f.accession_number == "0000320193-25-000079"
    assert f.url == "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-10k.htm"


def test_recent_filings_respects_limit():
    filings._ticker_to_cik_cache = {"AAPL": 320193}
    fake_submissions = {
        "filings": {
            "recent": {
                "form": ["8-K"] * 5,
                "filingDate": [f"2025-01-0{i}" for i in range(1, 6)],
                "accessionNumber": [f"0000320193-25-00000{i}" for i in range(1, 6)],
                "primaryDocument": [f"doc{i}.htm" for i in range(1, 6)],
            }
        }
    }
    original_get_json = filings._get_json
    filings._get_json = lambda url: fake_submissions
    try:
        results = filings.recent_filings("AAPL", limit=2)
    finally:
        filings._get_json = original_get_json
        filings._ticker_to_cik_cache = None
    assert len(results) == 2


def test_recent_filings_limit_zero_or_negative_returns_empty():
    # Real bug found in manual testing: the loop appended a filing BEFORE
    # checking len(out) >= limit, so limit=0 silently returned 1 result
    # instead of 0. Also verifies it short-circuits without even calling
    # _get_json (no network/lookup needed for a request for zero results).
    filings._ticker_to_cik_cache = {"AAPL": 320193}
    original_get_json = filings._get_json
    filings._get_json = lambda url: (_ for _ in ()).throw(AssertionError("should not be called"))
    try:
        assert filings.recent_filings("AAPL", limit=0) == []
        assert filings.recent_filings("AAPL", limit=-3) == []
    finally:
        filings._get_json = original_get_json
        filings._ticker_to_cik_cache = None
