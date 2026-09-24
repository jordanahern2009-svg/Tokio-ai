# Does the rigor engine actually work?

TokIO's whole pitch is that it won't call noise a signal. That is a
falsifiable claim, so this is the file where it gets falsified.

A test that reports "significant at p < 0.05" is promising something
specific: run it on data where nothing is there, and it should say
"significant" about 5% of the time. Not 20%. Not 45%. You can check this,
and until v0.3.0 nobody had.

**When we checked, TokIO failed.** Badly, and on exactly the questions it was
built to answer. This document is the study, the diagnosis, the fix, and the
same study re-run afterwards.

Reproduce all of it:

```bash
python scripts/calibration_study.py --trials 200
```

That is the exact invocation behind every number below. No API key and no
network; about 9 minutes on a laptop, nearly all of it spent running the
*old* method for comparison. The default (`--trials 120`) is quicker.

## How you test a test

Generate price data with **no predictable structure at all**, then ask TokIO
to find some. Every "significant" answer it returns is, by construction, a
false positive. Count them.

The generator is a GARCH(1,1) process. Returns are conditionally mean-zero,
so nothing computed from the past can forecast the sign of anything in the
future — there is no edge to find, at any horizon, with any feature. But it
reproduces the one property of real markets that matters here: **volatility
clusters.** Quiet stretches and violent stretches arrive in runs, exactly as
they do in real price series. Volume is modeled as persistent for the same
reason.

That detail turns out to be the whole story. An earlier version of this
study used plain Gaussian random walks and TokIO passed cleanly — 4.2%,
3.3%, 8.3%, 5.8% across horizons, all within noise of the promised 5%. The
bugs below are invisible on data that is too well-behaved, which is a decent
argument for why they survived a full test suite and a prior bug hunt.

## What we found

Two independent defects, which compound.

### 1. Selecting on volatility breaks a raw-difference permutation test

A permutation test shuffles group labels and asks how often chance produces
a gap this large. That is valid when the two groups are *exchangeable* under
the null — informally, when the only thing that could differ between them is
the mean.

Every condition worth asking TokIO about violates this. "Days that fell more
than 2%." "Days with unusual volume." These select high-volatility days **by
construction**, and their forward returns are genuinely more spread out than
baseline days are. Measured on the study data, the condition group had
**1.75x the variance** of the baseline group — and was 15x smaller (n≈54 vs
n≈844).

Pooling a small, noisy group with a large, quiet one and shuffling makes the
small group's mean look far more stable than it is. The null distribution
comes out too narrow, and ordinary noise clears the bar. This is the
Behrens–Fisher problem, and it shows up even at a one-day horizon where
nothing overlaps.

### 2. Overlapping forward windows are not independent observations

Ask about a 20-day forward return and consecutive observations share 19 of
their 20 days. They are nearly the same number. Shuffling labels treats them
as 800 independent draws when the sample holds nothing like 800 independent
pieces of information — and because volatility clusters, the *labels* arrive
in runs too, so the real overlap is worse than random.

Here is that defect in one line, on synthetic data with no relationship
whatsoever between the condition and the outcome:

| method | p-value |
|---|---|
| shuffled raw difference | **0.00033** |
| studentized circular shift | 0.49 |

Same data. One of these is wrong by a factor of about 1,500. It is a
regression test now
(`test_circular_shift_is_not_fooled_by_clustered_labels_on_autocorrelated_values`).

## The fix

**Studentize the statistic.** Instead of permuting the raw mean difference,
permute a Welch-style statistic — the difference divided by its own standard
error, with each group's variance recomputed on every relabeling. Permuting
a studentized statistic stays asymptotically valid when the groups have
different variances; permuting a raw difference does not. See Chung & Romano
(2013), *Exact and asymptotically robust permutation tests*.

**Rotate instead of shuffling.** For time-ordered data, don't scramble the
labels — slide the whole label series along the value series and wrap it
around. Every rotation preserves the autocorrelation of both series exactly,
so overlapping windows and clustered conditions are built into the null
rather than assumed away. The hypothesis being tested becomes the right one:
*are these two series related, beyond what each one's own internal structure
already explains?*

A rotation is also cheap. Only the condition-met days move, and they are
typically a small slice of all days, so TokIO enumerates **every distinct
rotation** rather than sampling — the p-value is exact rather than Monte
Carlo, and the whole thing got roughly 7x faster.

## Results

False-positive rate on data containing no edge. Nominal is 5%.

| condition | horizon | before (v0.2.0) | after (v0.3.0) |
|---|---|---|---|
| `daily_return < -2%` | 1d | 15.8% | **7.0%** |
| `daily_return < -2%` | 5d | 21.6% | **5.8%** |
| `daily_return < -2%` | 20d | 30.2% | **4.1%** |
| `daily_return < -2%` | 60d | 45.1% | **6.8%** |
| `volume_ratio > 2` | 1d | 4.0% | **4.5%** |
| `volume_ratio > 2` | 5d | 21.5% | **4.0%** |
| `volume_ratio > 2` | 20d | 26.0% | **3.5%** |
| `volume_ratio > 2` | 60d | 21.5% | **5.5%** |

Both columns come from the same run on the same simulated data, so the
comparison is paired rather than two studies stitched together. 200 paths per
configuration; a few are skipped where the condition group falls below the
`MIN_SAMPLE` floor, so the effective count per row is 162-200 and the
standard error on each figure is roughly 1.6 points.

Worst case went from **45.1% to 7.0%**. Every corrected figure lands
between 3.5% and 7.0%, all within sampling noise of the promised 5%.

Calibration did not come for free in the other direction: a planted, real
effect is still detected (`test_circular_shift_finds_a_real_planted_effect`).
A test that never fires would be perfectly "calibrated" and perfectly
useless.

## The example in our own README was a false positive

The uncomfortable part. Until v0.3.0 the README's showcase result was AAPL
gapping up more than 2% at the open, tested against the next 5 days, over 10
years of real data. It reported **p = 0.0042** and a confident story about
gap-fade.

Re-run on the same real data with the corrected test:

| method | p-value | verdict |
|---|---|---|
| shuffled raw difference (v0.2.0) | 0.0042 | "significant gap-fade" |
| studentized circular shift (v0.3.0) | **0.089** | not significant |

The variance ratio between the two groups is **2.68x** — those 76 gap-up
days really are far more volatile than the other 2,430, precisely the
condition that broke the old test. The pattern may well be real; the honest
statement is that 10 years of AAPL history is not enough to establish it.

A tool built to stop people fooling themselves with statistics had, in its
own front-page example, fooled itself with statistics. Finding that is the
point of writing the check.

## `check()` on harsher nulls

`check()` runs the engine on any condition a user writes, so the engine has
to hold on more than the two conditions and the one generator above. This
second study adds:

- **fat tails**: GARCH with Student-t(4) shocks (`garch_t4`)
- **volatility regimes**: a calm (0.8%/day) and a crisis (3%/day) state,
  each lasting about 100 bars (`regime`)
- **a persistent condition**: "20-bar momentum is positive" stays true or
  false for weeks at a time
- **autocorrelated returns**: AR(1) at −0.25, like intraday bid-ask
  bounce (`ar_neg`), and at +0.15, like trend (`ar_pos`). These are tested
  with a persistent condition that is *independent* of the returns. A
  condition built from past returns genuinely predicts AR returns, so it
  wouldn't be a null.

In every row, the condition carries no information about future returns.
Both of `check()`'s engines are reported (see the head-to-head below for
why there are two), next to a Welch t-test
(`scipy.stats.ttest_ind(equal_var=False)`).
300 null paths × 1,000 bars per row.

```bash
python scripts/calibration_check.py --trials 300   # ~1 min on 4 cores
```

| generator | condition | horizon | t-test | TokIO `hodrick` (default) | TokIO `rotation` |
|---|---|---:|---:|---:|---:|
| garch | drop 2% | 1 | 4.6% | 4.3% | 4.3% |
| garch | drop 2% | 5 | 7.5% | 3.6% | 3.2% |
| garch | drop 2% | 20 | **21.2%** | 3.2% | 2.5% |
| garch | up day | 1 | 3.3% | 3.0% | 3.3% |
| garch | up day | 5 | 5.7% | 5.7% | 5.3% |
| garch | up day | 20 | 6.3% | 6.3% | 5.0% |
| garch | momentum 20 | 1 | 6.7% | 7.0% | 4.3% |
| garch | momentum 20 | 5 | **35.3%** | 6.0% | 5.3% |
| garch | momentum 20 | 20 | **56.7%** | 4.7% | 2.3% |
| garch | vol shock | 1 | 5.7% | 4.0% | 5.7% |
| garch | vol shock | 5 | 3.0% | 1.3% | 2.7% |
| garch | vol shock | 20 | 3.7% | 3.7% | 4.0% |
| garch | independent, persistent | 1 | 3.0% | 2.7% | 3.7% |
| garch | independent, persistent | 5 | **34.3%** | 5.7% | 6.0% |
| garch | independent, persistent | 20 | **61.0%** | 3.7% | 5.0% |
| garch_t4 | drop 2% | 1 | 5.3% | 4.1% | 4.7% |
| garch_t4 | drop 2% | 5 | 6.5% | 3.6% | 4.2% |
| garch_t4 | drop 2% | 20 | **25.6%** | 3.7% | 3.7% |
| garch_t4 | up day | 1 | 4.7% | 3.7% | 4.0% |
| garch_t4 | up day | 5 | 6.3% | 7.3% | 6.0% |
| garch_t4 | up day | 20 | 6.7% | 7.3% | 5.3% |
| garch_t4 | momentum 20 | 1 | 5.0% | 5.0% | 4.7% |
| garch_t4 | momentum 20 | 5 | **33.0%** | 5.0% | 4.3% |
| garch_t4 | momentum 20 | 20 | **51.7%** | 4.3% | 3.3% |
| garch_t4 | vol shock | 1 | 4.3% | 3.7% | 3.7% |
| garch_t4 | vol shock | 5 | 4.3% | 3.3% | 3.0% |
| garch_t4 | vol shock | 20 | 4.0% | 3.3% | 3.7% |
| garch_t4 | independent, persistent | 1 | 4.7% | 5.0% | 4.3% |
| garch_t4 | independent, persistent | 5 | **36.3%** | 5.7% | 7.0% |
| garch_t4 | independent, persistent | 20 | **57.7%** | 4.0% | 4.3% |
| regime | drop 2% | 1 | 4.0% | 3.7% | 4.0% |
| regime | drop 2% | 5 | 10.7% | 7.4% | 5.0% |
| regime | drop 2% | 20 | **24.1%** | 5.4% | 3.3% |
| regime | up day | 1 | 6.3% | 6.3% | 6.7% |
| regime | up day | 5 | 3.7% | 4.7% | 4.7% |
| regime | up day | 20 | 5.3% | 5.0% | 4.3% |
| regime | momentum 20 | 1 | 7.7% | 7.0% | 3.7% |
| regime | momentum 20 | 5 | **32.3%** | 6.3% | 2.7% |
| regime | momentum 20 | 20 | **54.7%** | 2.7% | 3.3% |
| regime | vol shock | 1 | 4.0% | 4.0% | 3.7% |
| regime | vol shock | 5 | 7.3% | 5.0% | 5.3% |
| regime | vol shock | 20 | 9.0% | 4.7% | 6.3% |
| regime | independent, persistent | 1 | 5.0% | 4.7% | 3.7% |
| regime | independent, persistent | 5 | **36.7%** | 5.3% | 4.3% |
| regime | independent, persistent | 20 | **60.7%** | 3.0% | 3.3% |
| ar_neg | independent, persistent | 1 | 0.3% | 2.3% | 4.7% |
| ar_neg | independent, persistent | 5 | **33.3%** | 3.7% | 5.7% |
| ar_neg | independent, persistent | 20 | **60.0%** | 3.3% | 5.0% |
| ar_pos | independent, persistent | 1 | 8.0% | 2.7% | 3.7% |
| ar_pos | independent, persistent | 5 | **36.3%** | 6.0% | 6.0% |
| ar_pos | independent, persistent | 20 | **60.7%** | 4.3% | 5.0% |

**Worst case: t-test 61.0%, TokIO `hodrick` 7.4%, TokIO `rotation` 7.0%.**
With 300 paths per row, one standard error is about 1.3 points, so both
engines are within noise of 5%. The t-test's failures sit where the time
structure is strongest: persistent conditions at long horizons, where it
"finds" an edge in 55–61% of markets that have none.

This is not hypothetical on real data. On 10 years of SPY, "20-day
momentum up → next 20 days" gets p = 0.0001 from the t-test, p = 0.33 from
`check()`'s default engine, and p = 0.31 from the rotation engine.

## Head-to-head: TokIO vs Newey-West vs the stationary bootstrap

Beating a naive t-test is a low bar. The real question a quant would ask is
how this compares to the tools they already trust:

- **Newey-West**: OLS of the forward return on a condition dummy, HAC
  standard errors with `maxlags = h` (statsmodels). This is the textbook fix
  for overlapping horizons.
- **Stationary block bootstrap** of the (condition, outcome) pairs, with
  block length from `arch`'s `optimal_block_length` and a studentized,
  centered statistic. This is what a careful quant does.

Every test answers the same question on the same paths, and we measure two
things:

- **Size:** how often it fires with no edge in the data (should be ~5%).
- **Power:** how often it fires with a real edge planted, sized to about
  1.5 standard errors so that power lands mid-range.

200 paths x 1,000 bars per row, on the same 51 null configurations as
above.

```bash
pip install -e ".[calibration]"
python scripts/benchmark.py --trials 200    # ~7 min on 4 cores
```

| test | mean size | **worst size** | mean power |
|---|---:|---:|---:|
| Welch t-test | 19.3% | **64.0%** | 74.0% |
| Newey-West (HAC, lags = h) | 7.0% | **15.0%** | 74.2% |
| stationary bootstrap (arch) | 7.8% | **15.0%** | 74.3% |
| **TokIO `hodrick` (default)** | **4.7%** | **7.7%** | 73.7% |
| TokIO `rotation` | 4.6% | 8.3% | 72.7% |

Where they differ (size / power):

| setup | Newey-West | bootstrap | TokIO `hodrick` | TokIO `rotation` |
|---|---:|---:|---:|---:|
| GARCH, momentum 20 → next 20 | 12.5% / 100% | 13.5% / 100% | 5.5% / 100% | 3.0% / 100% |
| regime, momentum 20 → next 20 | 12.0% / 100% | 13.0% / 100% | 6.5% / 100% | 3.5% / 100% |
| AR(+0.15) returns, independent condition → next 20 | 10.0% / 100% | 11.0% / 100% | 7.0% / 100% | 7.5% / 100% |
| GARCH-t4, drop 2% → next 20 | 15.0% / 100% | 15.0% / 98.3% | 6.7% / 100% | 8.3% / 85.0% |
| **GARCH, momentum 20 → next 1** | 6.0% / 26.0% | 7.5% / 31.0% | 6.0% / **25.5%** | 4.5% / **12.5%** |
| **regime, momentum 20 → next 1** | 6.5% / 26.0% | 8.5% / 30.5% | 6.0% / **25.5%** | 6.0% / **16.0%** |

(The GARCH-t4 rows have 60 usable paths, since 2% drops are rarer under
those tails. One standard error there is about 2.8 points.)

### How the default engine was chosen

The first version of this benchmark had only the rotation test. It held
its size everywhere, but lost to Newey-West in one clear place: a
persistent condition at a one-bar horizon, where it caught half as many
real edges. The mechanism: when a persistent condition carries a real edge,
the edge makes the return series persistent too. A rotation test keeps
that persistence in its null, so rotations a few bars off the truth look
nearly as extreme as the truth.

What we tried, in order:

1. **A minimum-shift guard band** (exclude near-identity rotations, as
   circular-shift tests in spatial statistics do). Power went from 12.5% to
   ~20%, but size rose to 6.5-8%. Rejected.
2. **Hodrick (1992) "1B" standard errors.** Regroup the statistic by bar
   instead of by observation, `sum_i a_i y_i = sum_t r_t B_t` with `B_t`
   the rolling sum of conditions whose window contains bar t, so the
   overlap is handled exactly instead of estimated. This matched
   Newey-West's power with far better size, but leaked to 10% when the
   one-bar returns are themselves autocorrelated (AR +0.15). That's
   expected: 1B assumes they aren't.
3. **Cauchy combination** of the rotation and 1B p-values (Liu & Xie 2020).
   Worst case 10%, worse than either alone. Rejected.
4. **1B plus a short Bartlett HAC** over the one-bar terms
   (`rigor/overlap.py`). The long-range overlap stays exact inside `B_t`;
   the kernel only has to cover the returns' own short memory, so it uses
   the Newey-West rule-of-thumb bandwidth, 4(n/100)^(2/9), about 6 at a
   thousand bars, not h. **Shipped as the default.**

The rotation test still runs on every `check()`, as a second opinion that
assumes nothing about return autocorrelation. When the two land on
opposite sides of alpha, the result says so, and which assumption the
verdict depends on.

**Neither engine is uniformly best, and we say so.** The Hodrick engine
relies on the one-bar returns having *short-range* autocorrelation, which is
true of daily bars and of bid-ask bounce, but not of returns with slow mean
drift. The rotation engine relies on stationarity but assumes nothing
about autocorrelation, and pays for that in power on persistent conditions
at short horizons. `method="rotation"` makes the rotation test the verdict.

## What is still not handled

- **Multiple testing across sessions.** The `TestLedger` corrects for every
  hypothesis in one conversation. Start a new chat and the counter resets,
  so testing 20 ideas across 20 chats dodges the correction entirely.
- **Universe and selection bias.** Testing one condition on a ticker you
  chose *because* you already noticed something there is not corrected by
  anything here, and cannot be.
- **The rotation null assumes stationarity.** A structural break mid-sample
  (a regime change, a company that behaves like two different companies
  before and after) violates it.
- **Artificial junctions.** A circular shift joins the end of the series
  to its start. `check()` also drops bars with missing data, which joins the
  bars on either side of every gap. A quick test with one 50-bar gap
  (GARCH, momentum condition, 200 paths) measured 6% vs 7% without the gap,
  so the effect looks negligible. But many gaps haven't been tested.
- **Simulation is not reality.** The studies cover volatility clustering,
  fat tails and volatility regimes. They do not simulate jumps, intraday
  structure, or drifting means.
