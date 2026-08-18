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
- **One artificial junction.** A circular shift joins the end of the series
  to its start. Negligible at n in the thousands, not free.
- **GARCH is not reality.** It captures volatility clustering, which is what
  broke the old test. Real returns also have jumps, fat tails beyond GARCH,
  and intraday structure this study does not simulate.
