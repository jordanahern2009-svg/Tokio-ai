"""Test a condition against YOUR data, with a test whose error rate is measured.

This is the library entry point: no agent, no API key, no network. Hand it a
return series you already have and a condition you think predicts it, and it
tells you whether the relationship is distinguishable from noise -- using the
same calibrated engine the agent uses (`rigor.stats.circular_shift_test`,
whose false-positive rate is measured in docs/calibration.md).

    import tokio_ai
    r = prices.pct_change()
    print(tokio_ai.check(r, r < -0.02, horizon=5))

Works with plain lists, numpy arrays and pandas Series. Nothing here imports
numpy or pandas; they are read through the ordinary iteration protocol.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .rigor.ledger import TestLedger
from .rigor.provenance import stamp
from .rigor.stats import MIN_SAMPLE, circular_shift_test

# Mirrors the ledger's threshold for flagging unequal variances, so the
# library and the agent say the same thing about the same data.
_VARIANCE_FLAG = 1.5


@dataclass(frozen=True)
class CheckResult:
    verdict: str  # "SIGNIFICANT", "NOT SIGNIFICANT" or "NOT REPORTABLE"
    p_value: float
    horizon: int
    n_condition: int
    n_other: int
    mean_condition: float
    mean_other: float
    variance_ratio: float | None
    alpha: float
    # Number of tests the verdict was corrected for: 1 unless a ledger was
    # passed. Part of the result because "significant" means different
    # things at 1 test and at 20.
    tests_corrected_for: int
    method: str
    iters: int
    seed: int | None
    provenance: str

    @property
    def significant(self) -> bool:
        return self.verdict == "SIGNIFICANT"

    @property
    def gap(self) -> float:
        return self.mean_condition - self.mean_other

    def __str__(self) -> str:
        h = f"{self.horizon} bar" + ("s" if self.horizon != 1 else "")
        if self.verdict == "NOT REPORTABLE":
            head = (
                f"NOT REPORTABLE: the condition held on {self.n_condition} bars and "
                f"failed on {self.n_other}; both need at least {MIN_SAMPLE} before any "
                f"verdict means anything."
            )
        else:
            correction = (
                f" after correcting for {self.tests_corrected_for} tests"
                if self.tests_corrected_for > 1 else ""
            )
            head = (
                f"{self.verdict}{correction} (p={self.p_value:.4f}, alpha={self.alpha}). "
                f"Over the next {h}, the {self.n_condition} condition bars averaged "
                f"{self.mean_condition:+.3%} vs {self.mean_other:+.3%} on the other "
                f"{self.n_other} (gap {self.gap:+.3%})."
            )
        lines = [head]
        vr = self.variance_ratio
        if vr is not None and (vr > _VARIANCE_FLAG or vr < 1 / _VARIANCE_FLAG):
            if vr > 1:
                which = f"the condition bars are {vr:.2f}x as variable as the rest (it selects volatile periods)"
            else:
                which = f"the other bars are {1 / vr:.2f}x as variable as the condition bars (it selects calm periods)"
            # Pooled-variance tests are anti-conservative when the SMALLER
            # group is the noisier one, conservative the other way round.
            smaller_is_noisier = (self.n_condition < self.n_other) == (vr > 1)
            bias = "overstates" if smaller_is_noisier else "understates"
            lines.append(
                f"Unequal variances: {which}. Here, a test that pools the variances "
                f"{bias} significance; this test accounts for it."
            )
        lines.append(
            f"[{self.method}, iters={self.iters}, seed={self.seed} | {self.provenance}]"
        )
        return "\n".join(lines)


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return math.isnan(x)
    except TypeError:
        return False


def _aligned(returns: Any, condition: Any) -> None:
    """Refuse to silently pair two pandas objects by position when their
    indexes disagree -- the classic way a result ends up testing Monday's
    signal against Thursday's return without anyone noticing."""
    r_idx = getattr(returns, "index", None)
    c_idx = getattr(condition, "index", None)
    if r_idx is None or c_idx is None or not hasattr(r_idx, "equals"):
        return
    if not r_idx.equals(c_idx):
        raise ValueError(
            "returns and condition have different indexes; align them first "
            "(e.g. condition = condition.reindex(returns.index))"
        )


def forward_returns(returns: list[float | None], horizon: int) -> list[float | None]:
    """Compounded return over bars i+1 .. i+horizon, for each bar i.

    Starts at i+1, not i, deliberately: the condition at bar i is assumed
    knowable at the END of bar i, so the earliest return it can act on is the
    next one. Including bar i's own return would let a condition like
    "today fell 2%" predict itself.
    """
    n = len(returns)
    out: list[float | None] = [None] * n
    for i in range(n - horizon):
        growth = 1.0
        for r in returns[i + 1 : i + 1 + horizon]:
            if r is None:
                growth = None
                break
            growth *= 1.0 + r
        if growth is not None:
            out[i] = growth - 1.0
    return out


def check(
    returns: Iterable[Any],
    condition: Iterable[Any],
    horizon: int = 1,
    *,
    alpha: float = 0.05,
    iters: int = 5000,
    seed: int | None = 0,
    ledger: TestLedger | None = None,
    name: str | None = None,
) -> CheckResult:
    """Does `condition` at bar i predict the return over the next `horizon` bars?

    `returns` is per-bar simple returns in time order (e.g.
    `prices.pct_change()`). `condition` is a same-length boolean series,
    where `condition[i]` must be computable from information available at
    the close of bar i. Missing values (None/NaN) in either are skipped --
    but note pandas turns `NaN > x` into False, not NaN, so build conditions
    from shifted series with `.where(shifted.notna())` to keep "unknown"
    distinct from "no".

    The test is a studentized circular-shift randomization: it rotates the
    condition series against the outcome series, which keeps the time
    structure of both (volatility clustering, overlapping multi-bar
    windows) intact. Plain t-tests and shuffle tests destroy that structure
    and were measured calling noise significant up to 45% of the time on
    realistic data; see docs/calibration.md.

    Pass a `TestLedger` as `ledger` when checking several conditions on the
    same data, and every verdict is Benjamini-Hochberg corrected against all
    of them. Testing ten ideas and reporting the one that cleared p<0.05 is
    the single most common way to fool yourself, and this makes it hard.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon!r}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}")
    _aligned(returns, condition)

    r = [None if _is_missing(x) else float(x) for x in returns]
    c = [None if _is_missing(x) else bool(x) for x in condition]
    if len(r) != len(c):
        raise ValueError(f"returns and condition must be the same length, got {len(r)} and {len(c)}")

    fwd = forward_returns(r, horizon)
    labels: list[bool] = []
    values: list[float] = []
    for flag, v in zip(c, fwd):
        if flag is None or v is None:
            continue
        labels.append(flag)
        values.append(v)

    result = circular_shift_test(labels, values, iters=iters, seed=seed)
    group_a = [v for flag, v in zip(labels, values) if flag]
    group_b = [v for flag, v in zip(labels, values) if not flag]
    mean_a = statistics.fmean(group_a) if group_a else math.nan
    mean_b = statistics.fmean(group_b) if group_b else math.nan

    if ledger is not None:
        ledger.record(name or f"check_{len(ledger.tests) + 1}_h{horizon}", result)
        significant = ledger.significant_at(len(ledger.tests) - 1, alpha=alpha)
        corrected_for = len(ledger.tests)
    else:
        # <=, matching benjamini_hochberg, so a lone check and a ledger of
        # one give the same answer.
        significant = result.p_value <= alpha
        corrected_for = 1

    if not result.meets_min_sample:
        verdict = "NOT REPORTABLE"
    else:
        verdict = "SIGNIFICANT" if significant else "NOT SIGNIFICANT"

    return CheckResult(
        verdict=verdict,
        p_value=result.p_value,
        horizon=horizon,
        n_condition=result.n_a,
        n_other=result.n_b,
        mean_condition=mean_a,
        mean_other=mean_b,
        variance_ratio=result.variance_ratio,
        alpha=alpha,
        tests_corrected_for=corrected_for,
        method=result.statistic,
        iters=result.iters,
        seed=result.seed,
        provenance=stamp(),
    )
