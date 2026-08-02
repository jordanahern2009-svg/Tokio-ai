"""Honest statistical primitives for testing trading/market hypotheses.

Every function here exists because eyeballing a mean return and calling it
an "edge" is how retail (and plenty of professional) research goes wrong.
Nothing in this module tells you a pattern is real without also telling you
how likely it is to be noise, and MIN_SAMPLE below is a hard floor, not a
suggestion.
"""

from __future__ import annotations

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

    @property
    def meets_min_sample(self) -> bool:
        return self.n_a >= MIN_SAMPLE and self.n_b >= MIN_SAMPLE


def permutation_test(a: list[float], b: list[float], iters: int = 5000, seed: int | None = 0) -> PermutationResult:
    """Two-sided permutation test for a difference in means between `a` and `b`.

    Pools both samples, reshuffles the labels `iters` times, and counts how
    often a random relabeling produces a gap at least as extreme as the one
    actually observed. Makes no distributional assumption (unlike a t-test),
    which matters for the fat-tailed, skewed returns markets actually produce.

    `iters` and `seed` are carried on the result so every verdict is
    independently reproducible from its own report, not just from reading
    the source code's current defaults.
    """
    if not a or not b:
        return PermutationResult(0.0, 1.0, len(a), len(b), iters, seed)
    observed = statistics.fmean(a) - statistics.fmean(b)
    pool = list(a) + list(b)
    n = len(a)
    rng = random.Random(seed)
    hits = 0
    for _ in range(iters):
        rng.shuffle(pool)
        gap = statistics.fmean(pool[:n]) - statistics.fmean(pool[n:])
        if abs(gap) >= abs(observed):
            hits += 1
    return PermutationResult(observed, hits / iters, len(a), len(b), iters, seed)


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
