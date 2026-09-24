"""False-positive rate of `tokio_ai.check()` on harsher nulls than the original study.

`calibration_study.py` established that the engine holds on GARCH(1,1) data
with two conditions. `check()` puts the engine in front of arbitrary user
conditions, so it has to hold on more than that. This study adds:

  * fat tails        -- GARCH with Student-t (df=4) shocks
  * regime switches  -- volatility jumping between a calm and a crisis state
                        that each persist for months
  * a persistent condition -- "20-bar momentum is positive", which stays
                        true or false for weeks at a time. That is the
                        hardest case for a rotation test, because there are
                        few genuinely independent label runs to rotate.
  * autocorrelated returns -- AR(1) at -0.25 (bid-ask bounce) and +0.15
                        (trend), tested with a condition independent of the
                        returns. This is the case the Hodrick engine's
                        short HAC exists for.

In every row the condition carries no information about future returns, so
every SIGNIFICANT verdict is a false positive. A calibrated test fires about
5% of the time. Both of check()'s engines are reported.

Alongside it runs what most people actually use -- a Welch t-test on the
same two groups (scipy's `ttest_ind(equal_var=False)`) -- so the table shows
what the calibration is worth, not just that it exists.

Run it:  python scripts/calibration_check.py [--trials N] [--workers N]
Needs scipy for the t-test column; no API key, no network.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokio_ai.check import check, forward_returns  # noqa: E402

ALPHA = 0.05
HORIZONS = (1, 5, 20)
GENERATORS = ("garch", "garch_t4", "regime")
CONDITIONS = ("drop_2pct", "up_day", "momentum_20", "vol_shock")
# Autocorrelated returns: bid-ask bounce (negative) and trend (positive).
# Conditions computed from past returns genuinely predict AR returns, so
# they are not nulls here; only a condition independent of the returns is.
AR_GENERATORS = ("ar_neg", "ar_pos")
EXOG = "exog_persistent"
NULL_GRID = [(g, c) for g in GENERATORS for c in CONDITIONS + (EXOG,)] + [
    (g, EXOG) for g in AR_GENERATORS
]


def generate(kind: str, n: int, seed: int) -> list[float]:
    if kind in AR_GENERATORS:
        phi = -0.25 if kind == "ar_neg" else 0.15
        out, prev = [], 0.0
        for e in generate("garch", n, seed):
            prev = phi * prev + e
            out.append(prev)
        return out
    rng = random.Random(seed)
    out: list[float] = []
    if kind in ("garch", "garch_t4"):
        omega, a, b = 2.0e-6, 0.09, 0.90
        var = omega / (1 - a - b)
        for _ in range(n):
            if kind == "garch":
                z = rng.gauss(0, 1)
            else:
                # Student-t(4) scaled to unit variance: var of t(df) is df/(df-2).
                df = 4
                chi2 = sum(rng.gauss(0, 1) ** 2 for _ in range(df))
                z = rng.gauss(0, 1) / math.sqrt(chi2 / df) / math.sqrt(df / (df - 2))
            r = math.sqrt(var) * z
            var = omega + a * r * r + b * var
            out.append(r)
        return out
    if kind == "regime":
        # Calm 0.8% daily vol, crisis 3%; each state persists ~100 bars.
        calm = True
        for _ in range(n):
            if rng.random() < 0.01:
                calm = not calm
            out.append(rng.gauss(0, 0.008 if calm else 0.03))
        return out
    raise ValueError(kind)


def condition(kind: str, r: list[float]) -> list[bool | None]:
    """Each value uses only information up to and including its own bar."""
    n = len(r)
    if kind == "drop_2pct":
        return [x < -0.02 for x in r]
    if kind == "up_day":
        return [x > 0 for x in r]
    if kind == "momentum_20":
        out: list[bool | None] = [None] * n
        growth = [1.0]
        for x in r:
            growth.append(growth[-1] * (1 + x))
        for i in range(19, n):
            out[i] = growth[i + 1] / growth[i - 19] > 1
        return out
    if kind == EXOG:
        # Independent of the returns, flipping about every 30 bars. The
        # seed is derived from r[0] only so each path gets its own label
        # series reproducibly: Random() hashes the seed, so the draws carry
        # no information about r[0]'s sign or size -- and r[0] is in no
        # tested forward window anyway (windows start at bar 1).
        rng = random.Random(int(abs(r[0]) * 1e15) + 17)
        state, out = rng.random() < 0.5, []
        for _ in r:
            if rng.random() < 1 / 30:
                state = not state
            out.append(state)
        return out
    if kind == "vol_shock":
        out = [None] * n
        for i in range(20, n):
            window = r[i - 20 : i]
            m = sum(window) / 20
            sd = math.sqrt(sum((x - m) ** 2 for x in window) / 19)
            out[i] = abs(r[i]) > 2 * sd
        return out
    raise ValueError(kind)


def welch_p(labels: list[bool | None], fwd: list[float | None]) -> float | None:
    from scipy.stats import ttest_ind

    a = [v for c, v in zip(labels, fwd) if c is True and v is not None]
    b = [v for c, v in zip(labels, fwd) if c is False and v is not None]
    if len(a) < 30 or len(b) < 30:
        return None
    return float(ttest_ind(a, b, equal_var=False).pvalue)


def run_config(args):
    gen, cond_kind, horizon, trials, n_bars = args
    usable = hod_hits = rot_hits = t_hits = 0
    for t in range(trials):
        r = generate(gen, n_bars, seed=10_000 + t)
        labels = condition(cond_kind, r)
        res = check(r, labels, horizon=horizon, iters=2000, seed=t)
        if res.verdict == "NOT REPORTABLE":
            continue
        usable += 1
        hod_hits += res.p_hodrick <= ALPHA
        rot_hits += res.p_rotation <= ALPHA
        p = welch_p(labels, forward_returns(r, horizon))
        t_hits += p is not None and p <= ALPHA
    return gen, cond_kind, horizon, usable, hod_hits, rot_hits, t_hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--bars", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = [(g, c, h, args.trials, args.bars) for g, c in NULL_GRID for h in HORIZONS]
    started = time.time()
    print(f"check() calibration -- {args.trials} null paths x {args.bars} bars per row. "
          f"No edge exists; a calibrated test fires at {ALPHA:.0%}.\n", flush=True)
    print("| generator | condition | horizon | t-test | TokIO hodrick (default) | TokIO rotation | usable paths |")
    print("|---|---|---:|---:|---:|---:|---:|", flush=True)
    worst = {"t": 0.0, "hod": 0.0, "rot": 0.0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for gen, cond_kind, h, usable, hod, rot, tt in pool.map(run_config, jobs):
            if not usable:
                print(f"| {gen} | {cond_kind} | {h} | - | - | - | 0 |", flush=True)
                continue
            rates = {"t": tt / usable, "hod": hod / usable, "rot": rot / usable}
            for k, v in rates.items():
                worst[k] = max(worst[k], v)
            print(f"| {gen} | {cond_kind} | {h} | {rates['t']:.1%} | {rates['hod']:.1%} | "
                  f"{rates['rot']:.1%} | {usable} |", flush=True)
    print(f"\nworst case: t-test {worst['t']:.1%}, TokIO hodrick {worst['hod']:.1%}, "
          f"TokIO rotation {worst['rot']:.1%}")
    print(f"elapsed {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
