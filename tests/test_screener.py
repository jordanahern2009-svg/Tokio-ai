from tokio_ai.tools.prices import DailyBar
from tokio_ai.tools.screener import _return_over_period, load_universe


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
