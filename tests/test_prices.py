from tokio_ai.tools.prices import (
    DailyBar,
    _assert_daily_granularity,
    _epoch_to_date,
    _parse_chart_payload,
)


def _fake_payload(timestamps, opens, highs, lows, closes, volumes, adjcloses=None) -> dict:
    quote = {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    indicators = {"quote": [quote]}
    if adjcloses is not None:
        indicators["adjclose"] = [{"adjclose": adjcloses}]
    return {"chart": {"result": [{"timestamp": timestamps, "indicators": indicators}]}}


def test_epoch_to_date_handles_pre_1970_without_crashing():
    # This is the actual Windows bug found in production: datetime.fromtimestamp()
    # raises OSError on negative epoch values on Windows. -157766400 = 1965-01-01 UTC.
    d = _epoch_to_date(-157766400)
    assert d.isoformat() == "1965-01-01"


def test_epoch_to_date_handles_normal_timestamp():
    # 1704067200 = 2024-01-01 00:00:00 UTC
    assert _epoch_to_date(1704067200).isoformat() == "2024-01-01"


def test_parse_chart_payload_builds_bars():
    payload = _fake_payload(
        timestamps=[1704067200, 1704153600],
        opens=[100.0, 101.0],
        highs=[102.0, 103.0],
        lows=[99.0, 100.0],
        closes=[101.0, 102.0],
        volumes=[1000, 1100],
        adjcloses=[100.5, 101.5],
    )
    bars = _parse_chart_payload(payload, "TEST")
    assert len(bars) == 2
    assert bars[0] == DailyBar(
        date="2024-01-01", open=100.0, high=102.0, low=99.0, close=101.0, adj_close=100.5, volume=1000
    )


def test_parse_chart_payload_skips_null_padded_gaps():
    # Yahoo pads non-trading days with nulls in the OHLC arrays; those rows
    # must be dropped, not turned into a bogus zero-price bar.
    payload = _fake_payload(
        timestamps=[1704067200, 1704153600],
        opens=[100.0, None],
        highs=[102.0, None],
        lows=[99.0, None],
        closes=[101.0, None],
        volumes=[1000, 0],
    )
    bars = _parse_chart_payload(payload, "TEST")
    assert len(bars) == 1
    assert bars[0].date == "2024-01-01"


def test_parse_chart_payload_falls_back_to_close_when_adjclose_missing():
    payload = _fake_payload(
        timestamps=[1704067200],
        opens=[100.0],
        highs=[102.0],
        lows=[99.0],
        closes=[101.0],
        volumes=[1000],
        adjcloses=None,
    )
    bars = _parse_chart_payload(payload, "TEST")
    assert bars[0].adj_close == 101.0


def test_parse_chart_payload_raises_on_empty_result():
    payload = {"chart": {"result": [], "error": "No data found"}}
    try:
        _parse_chart_payload(payload, "BADSYMBOL")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "BADSYMBOL" in str(e)


def _bar(d: str) -> DailyBar:
    return DailyBar(date=d, open=1, high=1, low=1, close=1, adj_close=1, volume=1)


def test_assert_daily_granularity_accepts_consecutive_daily_dates():
    bars = [_bar("2024-01-01"), _bar("2024-01-02"), _bar("2024-01-03"), _bar("2024-01-04")]
    _assert_daily_granularity(bars, "TEST", "10y")  # should not raise


def test_assert_daily_granularity_rejects_silently_downsampled_data():
    # The real bug this guards against: Yahoo returning quarterly bars for
    # interval=1d&range=max on old tickers, with no error of its own.
    bars = [_bar("2024-01-01"), _bar("2024-04-01"), _bar("2024-07-01"), _bar("2024-10-01")]
    try:
        _assert_daily_granularity(bars, "IBM", "max")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "IBM" in str(e)


def test_assert_daily_granularity_skips_check_under_three_bars():
    bars = [_bar("2024-01-01"), _bar("2024-04-01")]
    _assert_daily_granularity(bars, "TEST", "10y")  # too few bars to judge, should not raise
