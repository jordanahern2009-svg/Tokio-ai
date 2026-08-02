from tokio_ai.tools.patterns import bucket_forward_returns, compute_feature
from tokio_ai.tools.prices import DailyBar


def _bar(date: str, open_: float, close: float, volume: int = 1000) -> DailyBar:
    return DailyBar(date=date, open=open_, high=max(open_, close), low=min(open_, close),
                     close=close, adj_close=close, volume=volume)


def test_daily_return_first_bar_is_none():
    bars = [_bar("2024-01-01", 100, 100), _bar("2024-01-02", 100, 110)]
    values = compute_feature(bars, "daily_return")
    assert values[0] is None
    assert abs(values[1] - 0.10) < 1e-9


def test_gap_pct_uses_open_vs_prior_close():
    bars = [_bar("2024-01-01", 100, 105), _bar("2024-01-02", 110, 108)]
    values = compute_feature(bars, "gap_pct")
    assert values[0] is None
    assert abs(values[1] - ((110 - 105) / 105)) < 1e-9


def test_volume_ratio_needs_lookback_window():
    bars = [_bar(f"2024-01-{i:02d}", 100, 100, volume=1000) for i in range(1, 25)]
    values = compute_feature(bars, "volume_ratio")
    assert all(v is None for v in values[:20])
    assert values[20] is not None


def test_unknown_feature_raises():
    bars = [_bar("2024-01-01", 100, 100)]
    try:
        compute_feature(bars, "nonsense")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_bucket_forward_returns_condition_met_goes_to_group_a():
    # Exactly one bar (index 1) has both a defined feature and a valid
    # lookahead: index 0 has no prior bar (feature=None), index 2 has no bar
    # to look ahead to at horizon=1. Its +10% return clears the 5% threshold.
    bars = [_bar("2024-01-01", 100, 100), _bar("2024-01-02", 100, 110), _bar("2024-01-03", 110, 105)]
    group_a, group_b = bucket_forward_returns(bars, "daily_return", ">", 0.05, horizon_days=1)
    assert group_b == []
    assert len(group_a) == 1
    assert abs(group_a[0] - (105 / 110 - 1)) < 1e-9


def test_bucket_forward_returns_condition_not_met_goes_to_group_b():
    # Same shape, but index 1's return (+2%) falls below the 5% threshold.
    bars = [_bar("2024-01-01", 100, 100), _bar("2024-01-02", 100, 102), _bar("2024-01-03", 102, 105)]
    group_a, group_b = bucket_forward_returns(bars, "daily_return", ">", 0.05, horizon_days=1)
    assert group_a == []
    assert len(group_b) == 1
    assert abs(group_b[0] - (105 / 102 - 1)) < 1e-9


def test_bucket_forward_returns_drops_bars_without_enough_lookahead():
    bars = [_bar("2024-01-01", 100, 100), _bar("2024-01-02", 100, 110)]
    group_a, group_b = bucket_forward_returns(bars, "daily_return", ">", 0.05, horizon_days=5)
    assert group_a == []
    assert group_b == []


def test_invalid_op_raises():
    bars = [_bar("2024-01-01", 100, 100), _bar("2024-01-02", 100, 110)]
    try:
        bucket_forward_returns(bars, "daily_return", "==", 0.0, horizon_days=1)
        assert False, "expected ValueError"
    except ValueError:
        pass
