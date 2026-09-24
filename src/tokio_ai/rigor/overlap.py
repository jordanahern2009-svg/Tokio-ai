"""Hodrick (1992) standard errors for overlapping horizons, with a short HAC.

The problem every long-horizon test has to solve: forward returns over h
bars overlap, so consecutive observations share h-1 of their bars. Newey-West
estimates the resulting autocorrelation with a kernel and a bandwidth of
about h -- and at long horizons with a persistent condition, that estimate is
poor. Measured here, Newey-West fires 12-15% of the time against a nominal 5%.

Hodrick's insight is that nothing about the overlap needs estimating. The
statistic

    S = sum_i a_i * y_i,   y_i = r_{i+1} + ... + r_{i+h}

(a_i the centered condition, r the one-bar log returns) can be regrouped by
bar instead of by observation:

    S = sum_t r_t * B_t,   B_t = a_{t-h} + ... + a_{t-1}

-- each bar's return times the rolling sum of the conditions whose windows
contain it. The overlap is now inside B_t, known exactly. What is left is a
sum of one-bar terms u_t = (r_t - rbar) * B_t, whose only dependence comes
from the one-bar returns themselves. Under the null of no predictability,
those are a martingale difference sequence and the variance is simply
sum u_t^2 (Hodrick's "1B" estimator; Ang & Bekaert 2007 found it holds its
size where Newey-West and Hansen-Hodrick do not).

Real bar returns are not exactly a martingale difference: bid-ask bounce
makes intraday returns negatively autocorrelated, and trend makes them
positively autocorrelated. Plain 1B then leaks (measured: 10% at h=20 with
AR(1) = +0.15 returns). So the variance adds a Bartlett-kernel HAC over u_t
with a SHORT bandwidth -- the Newey-West rule of thumb 4(n/100)^(2/9), about
5 at a thousand bars. Short is enough, because the long-range structure is
already exact in B_t; the kernel only has to cover the return series' own
short memory. That combination held a 7.7% worst case across 51 null
configurations, including autocorrelated returns, where Newey-West reached
15%. See docs/calibration.md.

Pure Python, with an optional numpy path for large n. Both are tested
against each other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OverlapResult:
    z: float
    p_value: float
    bandwidth: int
    n_bars: int


def default_bandwidth(n: int) -> int:
    """Newey & West (1994) rule of thumb, floor(4 * (n/100)^(2/9))."""
    return int(math.floor(4 * (max(n, 1) / 100) ** (2 / 9)))


def _two_sided_normal_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def hodrick_test(
    bar_returns: list[float | None],
    labels: list[bool | None],
    horizon: int,
    bandwidth: int | None = None,
) -> OverlapResult:
    """Does labels[i] predict the h-bar return starting at bar i+1?

    `bar_returns[t]` is bar t's simple return (None/NaN = missing);
    `labels[i]` the condition at the close of bar i (None = unknown). Same
    alignment as `tokio_ai.check`. A label is used only when its whole
    forward window is present, exactly as `check` pairs them, so both
    engines test the same observations.

    Returns are converted to log returns so that windows add exactly. A
    bar at or below -100% has no logarithm; the whole series then falls
    back to simple returns, where the additive regrouping is an
    approximation.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon!r}")
    if len(bar_returns) != len(labels):
        raise ValueError("bar_returns and labels must be the same length")
    try:
        import numpy as np
    except ImportError:
        return _hodrick_python(bar_returns, labels, horizon, bandwidth)
    return _hodrick_numpy(np, bar_returns, labels, horizon, bandwidth)


def _missing(x) -> bool:
    if x is None:
        return True
    try:
        return math.isnan(x)
    except TypeError:
        return False


def _hodrick_python(bar_returns, labels, horizon, bandwidth) -> OverlapResult:
    # Positional access below; a pandas Series would index by label.
    bar_returns = list(bar_returns)
    labels = list(labels)
    n = len(bar_returns)
    present = [not _missing(x) for x in bar_returns]
    raw = [float(x) if ok else 0.0 for x, ok in zip(bar_returns, present)]
    use_log = all(x > -1.0 for x, ok in zip(raw, present) if ok)
    r = [math.log1p(x) if (ok and use_log) else x for x, ok in zip(raw, present)]

    # Missing-bar prefix counts, so "is the window i+1..i+h complete" is O(1).
    gaps = [0]
    for ok in present:
        gaps.append(gaps[-1] + (0 if ok else 1))
    valid = [False] * n
    for i in range(n - horizon):
        lab = labels[i]
        if lab is None or _missing(lab):
            continue
        valid[i] = gaps[i + 1 + horizon] - gaps[i + 1] == 0
    used = [float(bool(labels[i])) for i in range(n) if valid[i]]
    if not used:
        return OverlapResult(0.0, 1.0, 0, n)
    abar = sum(used) / len(used)
    a = [(float(bool(labels[i])) - abar) if valid[i] else 0.0 for i in range(n)]

    cum = [0.0]
    for x in a:
        cum.append(cum[-1] + x)
    B = [cum[t] - cum[max(t - horizon, 0)] for t in range(n)]

    n_present = sum(present)
    rbar = sum(r) / n_present if n_present else 0.0
    u = [((r[t] - rbar) * B[t]) if present[t] else 0.0 for t in range(n)]
    S = sum(u)

    L = default_bandwidth(n_present) if bandwidth is None else bandwidth
    V = sum(x * x for x in u)
    for k in range(1, L + 1):
        w = 1 - k / (L + 1)
        V += 2 * w * sum(u[t] * u[t + k] for t in range(n - k))
    if V <= 0:
        return OverlapResult(0.0, 1.0, L, n)
    z = S / math.sqrt(V)
    return OverlapResult(z, _two_sided_normal_p(z), L, n)


def _hodrick_numpy(np, bar_returns, labels, horizon, bandwidth) -> OverlapResult:
    try:
        raw = np.asarray(bar_returns, dtype=float)
        lab = np.asarray(labels, dtype=float)
    except (TypeError, ValueError):
        return _hodrick_python(bar_returns, labels, horizon, bandwidth)
    n = len(raw)
    present = ~np.isnan(raw)
    raw = np.where(present, raw, 0.0)
    use_log = bool(np.all(raw[present] > -1.0))
    r = np.log1p(raw) if use_log else raw  # missing bars are 0 either way

    gaps = np.concatenate([[0], np.cumsum(~present)])
    valid = np.zeros(n, dtype=bool)
    if n > horizon:
        i = np.arange(n - horizon)
        valid[i] = ~np.isnan(lab[i]) & (gaps[i + 1 + horizon] - gaps[i + 1] == 0)
    if not valid.any():
        return OverlapResult(0.0, 1.0, 0, n)
    flag = np.where(np.isnan(lab), 0.0, (lab != 0).astype(float))
    abar = flag[valid].mean()
    a = np.where(valid, flag - abar, 0.0)

    cum = np.concatenate([[0.0], np.cumsum(a)])
    t = np.arange(n)
    B = cum[t] - cum[np.maximum(t - horizon, 0)]

    n_present = int(present.sum())
    rbar = r[present].sum() / n_present if n_present else 0.0
    u = np.where(present, (r - rbar) * B, 0.0)
    S = float(u.sum())

    L = default_bandwidth(n_present) if bandwidth is None else bandwidth
    V = float(u @ u)
    for k in range(1, L + 1):
        V += 2 * (1 - k / (L + 1)) * float(u[k:] @ u[:-k])
    if V <= 0:
        return OverlapResult(0.0, 1.0, L, n)
    z = S / math.sqrt(V)
    return OverlapResult(z, _two_sided_normal_p(z), L, n)
