"""Honest statistical primitives for testing trading/market hypotheses.

Every function here exists because eyeballing a mean return and calling it
an "edge" is how retail (and plenty of professional) research goes wrong.
Nothing in this module tells you a pattern is real without also telling you
how likely it is to be noise, and MIN_SAMPLE below is a hard floor, not a
suggestion.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

MIN_SAMPLE = 30


@dataclass(frozen=True)
class PermutationResult:
    observed_gap: float
    p_value: float
    n_a: int
    n_b: int
    iters: int = 5000
    seed: int | None = 0
    # Which test statistic the p-value came from. Kept on the result (and
    # persisted with it) because the answer changed in 0.3.0 -- a stored
    # verdict has to say which one produced it, or old and new results
    # silently look comparable when they aren't.
    statistic: str = "studentized_mean_diff"
    # var(group A) / var(group B). Surfaced because it is the single number
    # that says whether the studentization below was load-bearing for this
    # particular test: at ~1.0 studentized and raw agree, and the further
    # from 1.0, the more the raw-difference test would have overstated
    # significance. See docs/calibration.md.
    variance_ratio: float | None = None

    @property
    def meets_min_sample(self) -> bool:
        return self.n_a >= MIN_SAMPLE and self.n_b >= MIN_SAMPLE


def _welch_t(sum_a: float, sumsq_a: float, n_a: int, sum_b: float, sumsq_b: float, n_b: int) -> float | None:
    """Welch t-statistic from running sums. None when it is undefined.

    Uses the computational form (sumsq - sum^2/n); the caller centers the
    pool first, which keeps that form numerically well-behaved.
    """
    if n_a < 2 or n_b < 2:
        return None
    var_a = (sumsq_a - sum_a * sum_a / n_a) / (n_a - 1)
    var_b = (sumsq_b - sum_b * sum_b / n_b) / (n_b - 1)
    # Tiny negative values are possible from floating-point cancellation.
    var_a = max(var_a, 0.0)
    var_b = max(var_b, 0.0)
    diff = sum_a / n_a - sum_b / n_b
    denom = (var_a / n_a + var_b / n_b) ** 0.5
    if denom == 0:
        # Both groups constant. Not "undefined" -- either they sit at the
        # same value (no difference at all, statistic 0) or they are
        # perfectly separated with zero noise, which is as extreme as a
        # difference can possibly be. Returning None here would have scored
        # a perfect separation as p=1.0, the exact inversion of the truth.
        if diff == 0:
            return 0.0
        return math.inf if diff > 0 else -math.inf
    return diff / denom


def permutation_test(a: list[float], b: list[float], iters: int = 5000, seed: int | None = 0) -> PermutationResult:
    """Two-sided permutation test for a difference in means between `a` and `b`.

    Pools both samples, reshuffles the labels `iters` times, and counts how
    often a random relabeling produces a statistic at least as extreme as
    the one actually observed. Makes no distributional assumption (unlike a
    t-test), which matters for the fat-tailed, skewed returns markets
    actually produce.

    **The statistic is studentized** (Welch-style: the mean difference
    divided by its own standard error), not the raw mean difference. This is
    not a cosmetic choice -- it is the difference between a calibrated test
    and a broken one for this project's main use case.

    A plain permutation test on a raw mean difference is only valid when the
    two groups are exchangeable under the null, which requires equal
    variances. Every interesting condition here selects on volatility: "days
    that fell more than 2%", "days with unusual volume". Those days have
    genuinely higher variance than baseline days, and there are far fewer of
    them. Pooling then understates how noisy the small group's mean really
    is, and the test reports significance that isn't there. Measured on
    GARCH(1,1) data with no forecastable edge whatsoever, the raw-difference
    version fired at 14-31% against a nominal 5%. Studentizing restores it.
    See `docs/calibration.md` for the full study, and Chung & Romano (2013),
    "Exact and asymptotically robust permutation tests", for why permuting a
    studentized statistic is asymptotically valid under unequal variances
    while permuting a raw difference is not.

    `iters` and `seed` are carried on the result so every verdict is
    independently reproducible from its own report, not just from reading
    the source code's current defaults.

    The p-value uses the standard plus-one correction, `(hits+1)/(iters+1)`
    -- never a bare `hits/iters`. With finite Monte Carlo resampling, a raw
    ratio can land on exactly 0.0, which reads as "impossible under the
    null" when the honest claim is "no more extreme than roughly
    1/(iters+1)". Reporting an exact zero would be the project's own rigor
    layer overclaiming the one thing it exists to prevent. See Phipson &
    Smyth (2010), "Permutation P-values Should Never Be Zero".
    """
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters!r}")
    n_a, n_b = len(a), len(b)
    if not a or not b:
        return PermutationResult(0.0, 1.0, n_a, n_b, iters, seed)

    observed_gap = statistics.fmean(a) - statistics.fmean(b)
    var_ratio: float | None = None
    if n_a >= 2 and n_b >= 2:
        var_b_raw = statistics.variance(b)
        var_ratio = (statistics.variance(a) / var_b_raw) if var_b_raw else None

    # Center the pool once. Shifting every value by a constant leaves both
    # the mean difference and both variances unchanged, so the statistic is
    # identical -- but it keeps the sumsq form below from cancelling
    # catastrophically when values are far from zero.
    pool = list(a) + list(b)
    grand_mean = statistics.fmean(pool)
    pool = [x - grand_mean for x in pool]

    total_sum = sum(pool)
    total_sumsq = sum(x * x for x in pool)

    observed_t = _welch_t(
        sum(pool[:n_a]), sum(x * x for x in pool[:n_a]), n_a,
        sum(pool[n_a:]), sum(x * x for x in pool[n_a:]), n_b,
    )
    if observed_t is None:
        # Not enough spread (or samples) to studentize -- no honest claim of
        # significance is available, so make none.
        return PermutationResult(observed_gap, 1.0, n_a, n_b, iters, seed, variance_ratio=var_ratio)

    rng = random.Random(seed)
    target = abs(observed_t)
    hits = 0
    for _ in range(iters):
        rng.shuffle(pool)
        sum_a = 0.0
        sumsq_a = 0.0
        for x in pool[:n_a]:  # one pass; group B follows by subtraction
            sum_a += x
            sumsq_a += x * x
        t = _welch_t(sum_a, sumsq_a, n_a, total_sum - sum_a, total_sumsq - sumsq_a, n_b)
        if t is not None and abs(t) >= target:
            hits += 1

    return PermutationResult(
        observed_gap, (hits + 1) / (iters + 1), n_a, n_b, iters, seed, variance_ratio=var_ratio
    )


def bonferroni_correct(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Family-wise correction: strict, conservative, easy to explain."""
    if not p_values:
        return []
    threshold = alpha / len(p_values)
    return [p <= threshold for p in p_values]


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """False-discovery-rate correction: less conservative than Bonferroni,
    the standard choice when screening many hypotheses (e.g. one agent
    session testing several signals) and tolerating a controlled fraction of
    false positives among the ones called significant.

    Standard BH step-up procedure: sort p-values ascending, find the largest
    rank k where p_(k) <= (k/m)*alpha, and reject (call significant) every
    hypothesis at or below that rank.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    significant = [False] * m
    largest_k = 0
    for rank, idx in enumerate(order, start=1):
        threshold = (rank / m) * alpha
        if p_values[idx] <= threshold:
            largest_k = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= largest_k:
            significant[idx] = True
    return significant


# Relative slack when comparing a rotation's statistic to the observed one.
# The observed rotation must always count as "at least as extreme" as
# itself, and the FFT path computes it through a different floating-point
# route than the direct sum -- without slack, rounding can drop the identity
# from the count and report a p-value below the smallest honest one.
_TIE_RTOL = 1e-9


def _at_least_as_extreme(t: float, observed: float) -> bool:
    return abs(t) >= abs(observed) * (1 - _TIE_RTOL)


def _numpy():
    """numpy if installed, else None. One seam, so tests can force the
    pure-Python reference path and compare the two."""
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _fft_rotation_hits(np, a, centered, n_a, n_b, total_sum, total_sumsq, observed_t) -> int:
    """Count rotations at least as extreme as the observed one -- all n of
    them, in O(n log n).

    Group A's sum under shift s is sum_k a[k] * v[k - s] for the 0/1
    condition indicator a: a circular cross-correlation, which the FFT
    computes for every s at once (likewise the sum of squares, with v^2 in
    place of v). The pure-Python loop costs O(n * n_a) for the same answer,
    which at a million bars means hours.

    The FFT runs at a padded power-of-two length and the result is folded
    back to length n. A length-n FFT would give the circular correlation
    directly, but n is whatever the user's data happens to be, and a length
    with a large prime factor (999,995 = 5 x 199,999) sends the FFT down a
    path several times slower. Padding to L >= 2n makes the correlation
    linear, and circular[s] = linear[s] + linear[s - n] recovers it exactly.
    """
    n = len(centered)
    size = 1 << (2 * n - 1).bit_length()
    fa = np.fft.rfft(a, size)

    def circular_corr(x):
        full = np.fft.irfft(fa * np.conj(np.fft.rfft(x, size)), size)
        return full[:n] + full[size - n :]

    sum_a = circular_corr(centered)
    sumsq_a = circular_corr(centered * centered)
    sum_b = total_sum - sum_a
    sumsq_b = total_sumsq - sumsq_a

    var_a = np.maximum((sumsq_a - sum_a * sum_a / n_a) / (n_a - 1), 0.0)
    var_b = np.maximum((sumsq_b - sum_b * sum_b / n_b) / (n_b - 1), 0.0)
    diff = sum_a / n_a - sum_b / n_b
    denom = np.sqrt(var_a / n_a + var_b / n_b)
    # Zero spread in both groups: as in _welch_t, a perfect separation is
    # infinitely extreme, not "no evidence". After FFT rounding, "zero" has
    # to be a tolerance scaled to the data rather than an exact comparison.
    scale = max(1.0, float(np.sqrt(total_sumsq / n)))
    flat = denom <= 1e-12 * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(flat, 0.0, diff / np.where(flat, 1.0, denom))
    t = np.where(flat & (np.abs(diff) > 1e-12 * scale), np.inf, np.abs(t))
    return int(np.count_nonzero(t >= abs(observed_t) * (1 - _TIE_RTOL)))


def _circular_shift_numpy(np, labels, values, seed) -> PermutationResult:
    a = np.asarray(labels, dtype=bool)
    v = np.asarray(values, dtype=float)
    n = len(v)
    n_a = int(a.sum())
    n_b = n - n_a
    if n_a == 0 or n_b == 0:
        return PermutationResult(0.0, 1.0, n_a, n_b, n, seed, statistic="circular_shift")

    group_a, group_b = v[a], v[~a]
    observed_gap = float(group_a.mean() - group_b.mean())
    var_ratio: float | None = None
    if n_a >= 2 and n_b >= 2:
        var_b_raw = float(group_b.var(ddof=1))
        var_ratio = float(group_a.var(ddof=1)) / var_b_raw if var_b_raw else None

    centered = v - v.mean()
    total_sum = float(centered.sum())
    total_sumsq = float((centered * centered).sum())
    ca = centered[a]
    sum_a0 = float(ca.sum())
    sumsq_a0 = float((ca * ca).sum())
    observed_t = _welch_t(sum_a0, sumsq_a0, n_a, total_sum - sum_a0, total_sumsq - sumsq_a0, n_b)
    if observed_t is None:
        return PermutationResult(
            observed_gap, 1.0, n_a, n_b, n, seed,
            statistic="circular_shift", variance_ratio=var_ratio,
        )
    hits = _fft_rotation_hits(np, a.astype(float), centered, n_a, n_b, total_sum, total_sumsq, observed_t)
    # The identity rotation is in the count, so hits >= 1 and the p-value
    # can never be zero.
    return PermutationResult(
        observed_gap, hits / n, n_a, n_b, n, seed,
        statistic="circular_shift", variance_ratio=var_ratio,
    )


def circular_shift_test(
    labels: list[bool], values: list[float], iters: int = 5000, seed: int | None = 0
) -> PermutationResult:
    """Randomization test for time-ordered data: rotate, don't shuffle.

    `labels[i]` is whether the condition held on day i; `values[i]` is that
    day's forward return. Both series must be in time order and the same
    length.

    Why this exists instead of just calling `permutation_test`: shuffling
    individual labels destroys the time structure of *both* series, and that
    structure is real. Forward returns over a horizon longer than one day
    come from overlapping windows, so neighbouring values share most of
    their content and are heavily autocorrelated. Conditions cluster in time
    too (volatile days arrive in bursts; volume is persistent). A shuffled
    null quietly assumes neither is true, so it produces a null distribution
    that is far too narrow and calls noise significant. Measured on
    GARCH(1,1) data with no forecastable edge at all, the shuffled version
    fired at 22-39% for multi-day horizons against a nominal 5%.

    Rotating the label series against the value series instead preserves the
    autocorrelation of each one exactly, and tests the hypothesis that
    actually matters: are the two series related, beyond what their own
    internal structure explains? Every rotation is an equally likely
    relabeling under that null.

    With numpy installed, all `n` distinct rotations are evaluated at once
    through an FFT (see `_fft_rotation_hits`), so the p-value is *exact* at
    any n and `iters` is unused. Without numpy, a pure-Python loop
    enumerates all rotations when `n <= iters` (exact) and samples `iters`
    of them otherwise (Monte Carlo). The loop is the reference that the FFT
    path is tested against. The identity rotation is included in the count
    either way, which is where the usual plus-one comes from here.

    The statistic is the same studentized Welch difference used by
    `permutation_test` -- see there for why it is not the raw mean gap.
    """
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters!r}")
    if len(labels) != len(values):
        raise ValueError(f"labels and values must be the same length, got {len(labels)} and {len(values)}")

    np = _numpy()
    if np is not None:
        # Exact over every rotation at any n. `iters` is then unused; the
        # result's `iters` field reports the n rotations actually counted.
        return _circular_shift_numpy(np, labels, values, seed)

    n = len(values)
    true_idx = [i for i, flag in enumerate(labels) if flag]
    n_a = len(true_idx)
    n_b = n - n_a
    if n_a == 0 or n_b == 0:
        return PermutationResult(0.0, 1.0, n_a, n_b, iters, seed, statistic="circular_shift")

    group_a = [values[i] for i in true_idx]
    group_b = [values[i] for i, flag in enumerate(labels) if not flag]
    observed_gap = statistics.fmean(group_a) - statistics.fmean(group_b)
    var_ratio: float | None = None
    if n_a >= 2 and n_b >= 2:
        var_b_raw = statistics.variance(group_b)
        var_ratio = (statistics.variance(group_a) / var_b_raw) if var_b_raw else None

    # Centering leaves the statistic unchanged (see permutation_test) but
    # keeps the sum-of-squares form stable.
    grand_mean = statistics.fmean(values)
    centered = [v - grand_mean for v in values]
    total_sum = sum(centered)
    total_sumsq = sum(v * v for v in centered)

    def t_at(shift: int) -> float | None:
        sum_a = 0.0
        sumsq_a = 0.0
        for j in true_idx:
            v = centered[j - shift]  # negative indices wrap, which is the rotation
            sum_a += v
            sumsq_a += v * v
        return _welch_t(sum_a, sumsq_a, n_a, total_sum - sum_a, total_sumsq - sumsq_a, n_b)

    observed_t = t_at(0)
    if observed_t is None:
        return PermutationResult(
            observed_gap, 1.0, n_a, n_b, iters, seed,
            statistic="circular_shift", variance_ratio=var_ratio,
        )

    if n <= iters:
        shifts = range(n)  # every distinct rotation -> exact p-value
        exact = True
    else:
        rng = random.Random(seed)
        shifts = [0] + [rng.randrange(1, n) for _ in range(iters - 1)]
        exact = False

    hits = 0
    for s in shifts:
        t = t_at(s)
        if t is not None and _at_least_as_extreme(t, observed_t):
            hits += 1

    total = n if exact else iters
    # hits already includes the identity rotation, so this is the usual
    # never-report-exactly-zero guarantee without a separate +1.
    return PermutationResult(
        observed_gap, hits / total, n_a, n_b, total, seed,
        statistic="circular_shift", variance_ratio=var_ratio,
    )
