from tokio_ai.tools import screener
from tokio_ai.tools.prices import DailyBar
from tokio_ai.tools.screener import _return_over_period, load_universe, top_performers


def _bar(date: str, close: float) -> DailyBar:
    return DailyBar(date=date, open=close, high=close, low=close, close=close, adj_close=close, volume=1000)


def test_return_over_period_1mo():
    # 25 bars: enough for a 21-trading-day lookback with margin
    bars = [_bar(f"2024-01-{i:02d}", 100 + i) for i in range(1, 26)]
    ret = _return_over_period(bars, "1mo")
    expected = bars[-1].close / bars[-1 - 21].close - 1
    assert abs(ret - expected) < 1e-9


def test_return_over_period_insufficient_bars_returns_none():
    bars = [_bar("2024-01-01", 100), _bar("2024-01-02", 101)]
    assert _return_over_period(bars, "1mo") is None
    assert _return_over_period(bars, "1y") is None


def test_return_over_period_empty_bars_returns_none():
    assert _return_over_period([], "3mo") is None


def test_return_over_period_ytd_uses_first_bar_in_current_year():
    import datetime
    year = datetime.date.today().year
    bars = [
        _bar(f"{year - 1}-12-15", 90),
        _bar(f"{year}-01-02", 100),
        _bar(f"{year}-06-01", 130),
    ]
    ret = _return_over_period(bars, "ytd")
    assert abs(ret - (130 / 100 - 1)) < 1e-9


def test_return_over_period_ytd_no_bars_in_current_year_returns_none():
    bars = [_bar("2000-01-01", 100), _bar("2000-06-01", 110)]
    assert _return_over_period(bars, "ytd") is None


def test_load_universe_no_filter_returns_all_sp500():
    universe = load_universe()
    assert len(universe) > 400  # real S&P 500 snapshot, not a stub
    assert all({"symbol", "name", "sector"} <= row.keys() for row in universe)


def test_load_universe_decodes_special_characters_correctly():
    # Real bug found in manual testing: reading the bundled CSV without an
    # explicit encoding="utf-8" defaults to the platform's locale encoding
    # (cp1252 on many Windows setups), which silently mangles UTF-8 bytes
    # for names like "Brown-Forman" into multi-character mojibake. This
    # only reproduces on a non-UTF-8-default platform -- CI runs on
    # ubuntu-latest (UTF-8 by default) so it would NOT have caught this;
    # asserting the exact expected string here at least locks in the
    # correct value and documents the risk.
    universe = load_universe()
    by_symbol = {row["symbol"]: row["name"] for row in universe}
    assert by_symbol["BF-B"] == "Brown–Forman"
    assert by_symbol["EL"] == "Estée Lauder Companies (The)"


def test_load_universe_sector_filter_is_case_insensitive():
    exact = load_universe("Energy")
    lower = load_universe("energy")
    assert exact == lower
    assert len(exact) > 0
    assert all(row["sector"] == "Energy" for row in exact)


def test_load_universe_unknown_sector_returns_empty():
    assert load_universe("Not A Real Sector") == []


def test_class_share_tickers_use_yahoo_hyphen_format():
    universe = load_universe()
    symbols = {row["symbol"] for row in universe}
    assert "BRK-B" in symbols
    assert "BRK.B" not in symbols


def test_top_performers_rejects_non_positive_top_n():
    # Real bug found in manual testing: results[:top_n] with a negative
    # top_n silently returns "all but the last N" (Python slice semantics)
    # instead of erroring. These all raise before any network call happens
    # (validation runs before fetching), so no mocking needed.
    for bad in (0, -1, -100):
        try:
            top_performers(top_n=bad)
            assert False, f"expected ValueError for top_n={bad}"
        except ValueError:
            pass


def test_top_performers_rejects_invalid_period():
    try:
        top_performers(period="5y")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_top_performers_warns_on_low_data_yield():
    # If most symbols in the universe fail to fetch (network issue, or the
    # granularity guard in prices.py firing broadly), that should be
    # impossible to miss rather than something the caller has to notice by
    # comparing universe_size against symbols_with_data themselves.
    original_fetch_one = screener._fetch_one
    call_count = {"n": 0}

    def mostly_fail(symbol, period):
        call_count["n"] += 1
        return 0.05 if call_count["n"] == 1 else None  # only the first symbol "succeeds"

    screener._fetch_one = mostly_fail
    try:
        result = top_performers(sector="Energy", top_n=5)  # 21 symbols in Energy
    finally:
        screener._fetch_one = original_fetch_one

    assert result["symbols_with_data"] == 1
    assert "warning" in result
    assert "1/21" in result["warning"] or f"1/{result['universe_size']}" in result["warning"]


def test_top_performers_no_warning_when_yield_is_normal():
    original_fetch_one = screener._fetch_one
    screener._fetch_one = lambda symbol, period: 0.05
    try:
        result = top_performers(sector="Energy", top_n=5)
    finally:
        screener._fetch_one = original_fetch_one
    assert "warning" not in result
