"""Common technical-pattern testing: derive a simple feature from daily bars,
split days into two groups by comparing the feature to a threshold, and run
the split through the rigor engine's permutation test.

This exists because asking an LLM to manually bucket hundreds of raw price
bars inline (as tool_hypothesis alone requires) is unreliable and
token-expensive -- classic case for "let code do the arithmetic, let the
model decide what to ask for."
"""

from __future__ import annotations

from ..rigor.ledger import TestLedger
from ..rigor.stats import permutation_test
from .prices import DailyBar, fetch_daily_bars

FEATURES = ("daily_return", "gap_pct", "volume_ratio")
OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}
VOLUME_LOOKBACK = 20


def compute_feature(bars: list[DailyBar], feature: str) -> list[float | None]:
    """One value per bar, aligned by index. None where undefined (not enough
    history yet, e.g. the first bar has no prior close to diff against)."""
    if feature == "daily_return":
        out: list[float | None] = [None]
        for prev, cur in zip(bars, bars[1:]):
            out.append((cur.close - prev.close) / prev.close if prev.close else None)
        return out
    if feature == "gap_pct":
        out = [None]
        for prev, cur in zip(bars, bars[1:]):
            out.append((cur.open - prev.close) / prev.close if prev.close else None)
        return out
    if feature == "volume_ratio":
        out = [None] * len(bars)
        for i in range(VOLUME_LOOKBACK, len(bars)):
            window = bars[i - VOLUME_LOOKBACK : i]
            avg_vol = sum(b.volume for b in window) / len(window)
            out[i] = (bars[i].volume / avg_vol) if avg_vol else None
        return out
    raise ValueError(f"unknown feature {feature!r}, must be one of {FEATURES}")


def bucket_forward_returns(
    bars: list[DailyBar], feature: str, op: str, threshold: float, horizon_days: int
) -> tuple[list[float], list[float]]:
    """Split days into (condition-met, condition-not-met) groups and return
    each day's forward return from that day's close to horizon_days later."""
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}, must be one of {list(OPS)}")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    values = compute_feature(bars, feature)
    cmp = OPS[op]
    group_a: list[float] = []  # condition met
    group_b: list[float] = []  # condition not met
    for i, v in enumerate(values):
        exit_i = i + horizon_days
        if v is None or exit_i >= len(bars) or bars[i].close == 0:
            continue
        forward_return = bars[exit_i].close / bars[i].close - 1
        (group_a if cmp(v, threshold) else group_b).append(forward_return)
    return group_a, group_b


def test_return_pattern(
    ledger: TestLedger,
    symbol: str,
    feature: str,
    op: str,
    threshold: float,
    horizon_days: int,
    range_: str = "10y",
) -> str:
    bars = fetch_daily_bars(symbol, range_)
    group_a, group_b = bucket_forward_returns(bars, feature, op, threshold, horizon_days)
    result = permutation_test(group_a, group_b)
    name = f"{symbol}_{feature}_{op}{threshold}_{horizon_days}d"
    ledger.record(name, result)
    verdict = ledger.verdict(name)
    # The model has no reliable notion of "today" or the actual data window
    # fetched -- hand it the real dates so it reports facts, not a guess.
    data_window = f"data window: {bars[0].date} to {bars[-1].date}" if bars else "no data"
    return f"{verdict} ({data_window})"
