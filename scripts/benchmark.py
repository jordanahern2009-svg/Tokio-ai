"""Head-to-head: TokIO check() vs the tests a quant would actually reach for.

Question every test answers: does a condition at bar i predict the return
over bars i+1..i+h? Two-sided, alpha = 0.05.

Contenders
  welch_t     Welch t-test on the two groups (scipy). What most people run.
  newey_west  OLS of forward return on a condition dummy with Newey-West
              HAC standard errors, maxlags = h (statsmodels). The textbook
              fix for overlapping horizons.
  stat_boot   Stationary block bootstrap of the (condition, outcome) pairs,
              block length from arch's `optimal_block_length`, studentized
              and centered (arch). What a careful quant does.
  tokio_hod   tokio_ai.check() default engine: Hodrick (1992) standard
              errors, overlap handled exactly, plus a short HAC.
  tokio_rot   tokio_ai.check(method="rotation"): studentized circular-shift
              randomization.

Two numbers per test, and both matter:
  size   How often it fires on data with NO edge. Should be ~5%. Above that,
         it calls noise a signal. Below it, the test is conservative and
         pays for that in power.
  power  How often it fires when a real edge is planted. Higher is better,
         but only among tests whose size is honest; a test that fires 50%
         of the time on noise gets high "power" for free.

The planted edge: when the condition holds at bar i, every return in
i+1..i+h is shifted by the same drift. Its size is set relative to how
much the test can possibly see -- `effect` standard errors of the gap,
where the standard error treats overlapping windows as sharing
information (n/h effective observations) -- so power lands mid-range
instead of saturating at 100% for frequent conditions and 0% for rare
ones. The condition labels are computed before planting and passed in
unchanged, so the test sees exactly the relationship that was planted and
nothing else.

Run:  python scripts/benchmark.py [--trials N] [--workers N]
Needs:  pip install -e ".[calibration]"   (scipy, statsmodels, arch)
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_check import HORIZONS, NULL_GRID, condition, generate  # noqa: E402

from tokio_ai.check import check, forward_returns  # noqa: E402

ALPHA = 0.05
TESTS = ("welch_t", "newey_west", "stat_boot", "tokio_hod", "tokio_rot")
BOOT_REPS = 499


def _pairs(labels, fwd):
    xs, ys = [], []
    for c, v in zip(labels, fwd):
        if c is None or v is None:
            continue
        xs.append(1.0 if c else 0.0)
        ys.append(v)
    return xs, ys


def p_welch(x, y):
    from scipy.stats import ttest_ind

    a = [v for c, v in zip(x, y) if c]
    b = [v for c, v in zip(x, y) if not c]
    return float(ttest_ind(a, b, equal_var=False).pvalue)


def p_newey_west(x, y, h):
    import numpy as np
    import statsmodels.api as sm

    X = sm.add_constant(np.asarray(x))
    fit = sm.OLS(np.asarray(y), X).fit(cov_type="HAC", cov_kwds={"maxlags": h})
    return float(fit.pvalues[1])


def _welch_t_np(x, y):
    import numpy as np

    m = x > 0.5
    a, b = y[m], y[~m]
    if len(a) < 2 or len(b) < 2:
        return math.nan
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else math.nan


def p_stationary_bootstrap(x, y, seed):
    import numpy as np
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    x = np.asarray(x)
    y = np.asarray(y)
    t_obs = _welch_t_np(x, y)
    if not math.isfinite(t_obs):
        return 1.0
    # Block length has to cover the dependence in BOTH series: the outcome's
    # overlap and the condition's persistence. Take the larger of the two.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl = optimal_block_length(np.column_stack([x, y]))["stationary"].max()
    bs = StationaryBootstrap(max(float(bl), 1.0), x, y, seed=np.random.default_rng(seed))
    hits = 0
    total = 0
    for data, _ in bs.bootstrap(BOOT_REPS):
        tb = _welch_t_np(data[0], data[1])
        if not math.isfinite(tb):
            continue
        total += 1
        # Centered at the observed statistic: the bootstrap distribution
        # approximates the sampling distribution around the truth, so the
        # null distribution is t_b - t_obs.
        hits += abs(tb - t_obs) >= abs(t_obs)
    return (hits + 1) / (total + 1)


def run_config(job):
    gen, cond_kind, h, trials, n_bars, effect = job
    counts = {("size", t): 0 for t in TESTS} | {("power", t): 0 for t in TESTS}
    usable = 0
    for trial in range(trials):
        r = generate(gen, n_bars, seed=20_000 + trial)
        labels = condition(cond_kind, r)
        n_true = sum(1 for c in labels if c is True)
        n_false = sum(1 for c in labels if c is False)
        if n_true < 40 or n_false < 40:
            continue
        usable += 1
        sigma = (sum(v * v for v in r) / len(r)) ** 0.5
        se_gap = sigma * math.sqrt(h) * math.sqrt(h * (1 / n_true + 1 / n_false))
        drift = effect * se_gap / h
        planted = list(r)
        for i, c in enumerate(labels):
            if c:
                for j in range(i + 1, min(i + 1 + h, n_bars)):
                    planted[j] += drift
        for kind, series in (("size", r), ("power", planted)):
            fwd = forward_returns(series, h)
            x, y = _pairs(labels, fwd)
            res = check(series, labels, horizon=h, iters=2000, seed=trial)
            ps = {
                "welch_t": p_welch(x, y),
                "newey_west": p_newey_west(x, y, h),
                "stat_boot": p_stationary_bootstrap(x, y, seed=trial),
                "tokio_hod": res.p_hodrick,
                "tokio_rot": res.p_rotation,
            }
            for t in TESTS:
                counts[(kind, t)] += ps[t] <= ALPHA
    return gen, cond_kind, h, usable, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--bars", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--effect", type=float, default=1.5,
                        help="planted forward gap, in standard errors (see module docstring)")
    args = parser.parse_args()

    jobs = [(g, c, h, args.trials, args.bars, args.effect) for g, c in NULL_GRID for h in HORIZONS]
    started = time.time()
    print(f"benchmark -- {args.trials} paths x {args.bars} bars per row, planted effect "
          f"{args.effect} SE. size should be ~5%; power is only comparable among tests "
          f"whose size is honest.\n", flush=True)
    cols = " | ".join(f"{t} size | {t} power" for t in TESTS)
    print(f"| generator | condition | h | {cols} | paths |")
    print("|---|---|---:|" + "---:|" * (2 * len(TESTS)) + "---:|", flush=True)
    totals = {t: [0, 0, 0] for t in TESTS}  # size hits, power hits, n
    worst = {t: 0.0 for t in TESTS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for gen, cond_kind, h, usable, c in pool.map(run_config, jobs):
            if not usable:
                continue
            cells = []
            for t in TESTS:
                s, p = c[("size", t)] / usable, c[("power", t)] / usable
                worst[t] = max(worst[t], s)
                totals[t][0] += c[("size", t)]
                totals[t][1] += c[("power", t)]
                totals[t][2] += usable
                cells.append(f"{s:.1%} | {p:.1%}")
            print(f"| {gen} | {cond_kind} | {h} | " + " | ".join(cells) + f" | {usable} |", flush=True)
    print("\n| test | mean size | worst size | mean power |\n|---|---:|---:|---:|")
    for t in TESTS:
        s, p, n = totals[t]
        print(f"| {t} | {s / n:.1%} | {worst[t]:.1%} | {p / n:.1%} |")
    print(f"\nelapsed {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
