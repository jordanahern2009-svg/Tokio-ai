"""Does TokIO's rigor engine actually hold its promised error rate?

A test that claims "significant at p<0.05" is making a falsifiable promise:
run it on data where nothing is there, and it should say "significant"
about 5% of the time. That is checkable, so this checks it.

The generator here produces returns with NO forecastable edge whatsoever --
every return is conditionally mean-zero, so no feature computed from the
past can predict the sign of a future one. Any "significant" verdict is
therefore a false positive by construction. What the generator *does*
reproduce is the one thing about real markets that breaks naive tests:
volatility clusters (GARCH(1,1)), and volume is persistent. That makes
condition-days cluster in time, exactly as they do in real price series.

This study is what moved TokIO from a raw-difference shuffled permutation
test to a studentized circular-shift randomization test in 0.3.0. It found
the old one firing at up to 39% against a nominal 5%.

Run it:  python scripts/calibration_study.py [--trials N]
It needs no API key and no network access.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokio_ai.rigor.stats import circular_shift_test  # noqa: E402
from tokio_ai.tools.patterns import paired_forward_returns  # noqa: E402
from tokio_ai.tools.prices import DailyBar  # noqa: E402

ALPHA = 0.05
CONDITIONS = [
    ("daily_return", "<", -0.02),
    ("volume_ratio", ">", 2.0),
]
HORIZONS = (1, 5, 20, 60)


def garch_walk(n: int, seed: int, omega: float = 2.0e-6, alpha: float = 0.09, beta: float = 0.90) -> list[DailyBar]:
    """GARCH(1,1) price path with persistent volume. Unconditional daily vol ~1.4%.

    Conditionally mean-zero returns: there is no edge to find, at any
    horizon, from any feature.
    """
    rng = random.Random(seed)
    var = omega / (1 - alpha - beta)
    price = 100.0
    log_volume_state = 0.0
    bars: list[DailyBar] = []
    for i in range(n):
        r = (var**0.5) * rng.gauss(0, 1)
        var = omega + alpha * r * r + beta * var
        prev, price = price, price * (1 + r)
        open_ = prev * (1 + rng.gauss(0, (var**0.5) / 3))
        log_volume_state = 0.85 * log_volume_state + rng.gauss(0, 0.35) + 6.0 * abs(r)
        bars.append(
            DailyBar(
                date=f"{2000 + i // 252:04d}-01-01",
                open=open_,
                high=max(open_, price),
                low=min(open_, price),
                close=price,
                adj_close=price,
                volume=max(int(2_000_000 * pow(2.718281828, min(log_volume_state, 4.0))), 1000),
            )
        )
    return bars


def _raw_difference_p(group_a: list[float], group_b: list[float], seed: int) -> float:
    """Raw (non-studentized) shuffled permutation p-value: what TokIO did before 0.3.0.

    Kept inline in the study rather than exported from the package, because
    this is the broken version and nothing should be able to import it by
    accident. It is also the slow half of this script -- it Monte-Carlos
    where the current engine enumerates.
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return 1.0
    observed = abs(statistics.fmean(group_a) - statistics.fmean(group_b))
    pool = group_a + group_b
    n_a = len(group_a)
    rng = random.Random(seed)
    hits = 0
    iters = 1000
    for _ in range(iters):
        rng.shuffle(pool)
        if abs(statistics.fmean(pool[:n_a]) - statistics.fmean(pool[n_a:])) >= observed:
            hits += 1
    return (hits + 1) / (iters + 1)


def false_positive_rate(feature: str, op: str, threshold: float, horizon: int, trials: int, n_bars: int):
    old_hits = new_hits = usable = 0
    for t in range(trials):
        bars = garch_walk(n_bars, seed=5000 + t)
        labels, forward = paired_forward_returns(bars, feature, op, threshold, horizon)
        n_a = sum(labels)
        if n_a < 30 or (len(labels) - n_a) < 30:
            continue
        usable += 1
        group_a = [r for flag, r in zip(labels, forward) if flag]
        group_b = [r for flag, r in zip(labels, forward) if not flag]
        old_hits += _raw_difference_p(group_a, group_b, seed=t) < ALPHA
        new_hits += circular_shift_test(labels, forward, iters=5000, seed=t).p_value < ALPHA
    if not usable:
        return None
    return old_hits / usable, new_hits / usable, usable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=120, help="paths per configuration")
    parser.add_argument("--bars", type=int, default=900, help="trading days per path")
    args = parser.parse_args()

    started = time.time()
    print(f"TokIO calibration study -- {args.trials} GARCH paths x {args.bars} bars per configuration", flush=True)
    print(f"No edge exists in this data. A calibrated test fires at {ALPHA:.0%}.\n")
    header = f"{'condition':>24}  {'horizon':>7}  {'pre-0.3.0':>10}  {'0.3.0':>8}  {'trials':>6}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    worst_old = worst_new = 0.0
    for feature, op, threshold in CONDITIONS:
        for horizon in HORIZONS:
            out = false_positive_rate(feature, op, threshold, horizon, args.trials, args.bars)
            if out is None:
                continue
            old, new, usable = out
            worst_old, worst_new = max(worst_old, old), max(worst_new, new)
            print(f"{feature + ' ' + op + str(threshold):>24}  {str(horizon) + 'd':>7}  {old:>9.1%}{'':1}  {new:>7.1%}  {usable:>6}", flush=True)

    print("-" * len(header))
    print(f"{'worst case':>24}  {'':>7}  {worst_old:>9.1%}   {worst_new:>7.1%}")
    print(f"\nElapsed {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
