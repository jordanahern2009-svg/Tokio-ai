"""Agent-facing wrapper around the rigor engine: the tool that decides
whether a claimed pattern gets to call itself a signal."""

from __future__ import annotations

from ..rigor.ledger import TestLedger
from ..rigor.stats import permutation_test


def test_hypothesis(
    ledger: TestLedger,
    name: str,
    group_a: list[float],
    group_b: list[float],
    iters: int = 5000,
    seed: int = 0,
) -> str:
    result = permutation_test(group_a, group_b, iters=iters, seed=seed)
    ledger.record(name, result)
    return ledger.verdict(name)
