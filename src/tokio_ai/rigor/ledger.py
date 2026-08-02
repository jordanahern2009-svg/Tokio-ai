"""Session-level bookkeeping so an agent conversation can't p-hack itself.

Every hypothesis test run in a session gets recorded here. Verdicts are
issued relative to everything tested so far in the SAME session -- the 5th
signal tested in one sitting needs a stronger raw p-value than the 1st to
count as significant, which is exactly the discipline a naive
"just check p<0.05 every time" agent would skip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .provenance import stamp
from .stats import PermutationResult, benjamini_hochberg


@dataclass
class RecordedTest:
    name: str
    result: PermutationResult


@dataclass
class TestLedger:
    __test__ = False  # not a pytest test class despite the name

    tests: list[RecordedTest] = field(default_factory=list)

    def record(self, name: str, result: PermutationResult) -> None:
        self.tests.append(RecordedTest(name, result))

    def verdict(self, name: str, alpha: float = 0.05) -> str:
        p_values = [t.result.p_value for t in self.tests]
        sig = benjamini_hochberg(p_values, alpha=alpha)
        # Walk backwards so a reused name reports its MOST RECENT test, not
        # a stale first result -- every entry (including earlier duplicates)
        # still counts toward the correction denominator either way.
        for i in range(len(self.tests) - 1, -1, -1):
            t = self.tests[i]
            if t.name != name:
                continue
            if not t.result.meets_min_sample:
                return (
                    f"NOT REPORTABLE: n={t.result.n_a}/{t.result.n_b} is below the "
                    f"MIN_SAMPLE floor -- sample too small to say anything."
                )
            corrected = "SIGNIFICANT" if sig[i] else "NOT SIGNIFICANT"
            return (
                f"{corrected} after correcting for {len(self.tests)} hypothesis "
                f"test(s) run this session (raw p={t.result.p_value:.4f}, "
                f"gap={t.result.observed_gap:+.4%}, "
                f"reproducible with seed={t.result.seed} iters={t.result.iters})."
            )
        raise KeyError(f"no test recorded under name {name!r}")

    def summary(self) -> str:
        if not self.tests:
            return "No hypotheses tested yet."
        p_values = [t.result.p_value for t in self.tests]
        sig = benjamini_hochberg(p_values, alpha=0.05)
        lines = [f"{len(self.tests)} hypothesis test(s) this session (BH-corrected, alpha=0.05):"]
        for t, s in zip(self.tests, sig):
            flag = "SIGNIFICANT" if s else "not significant"
            lines.append(f"  {t.name}: p={t.result.p_value:.4f} gap={t.result.observed_gap:+.4%} -> {flag}")
        lines.append(f"[{stamp()}]")
        return "\n".join(lines)
