from tokio_ai.rigor.ledger import TestLedger
from tokio_ai.rigor.stats import PermutationResult, benjamini_hochberg, permutation_test


def test_near_identical_groups_are_not_significant():
    ledger = TestLedger()
    a = list(range(30))
    b = [x + 0.001 for x in a]
    result = permutation_test(a, b, iters=500, seed=0)
    ledger.record("noop", result)
    assert ledger.verdict("noop").startswith("NOT SIGNIFICANT")


def test_small_sample_is_not_reportable():
    ledger = TestLedger()
    result = permutation_test([1.0] * 5, [0.0] * 5, iters=100, seed=0)
    ledger.record("tiny", result)
    assert ledger.verdict("tiny").startswith("NOT REPORTABLE")


def test_verdicts_match_manual_bh_correction():
    ledger = TestLedger()
    ps = [0.001, 0.02, 0.03, 0.04, 0.2]
    for i, p in enumerate(ps):
        ledger.record(f"h{i}", PermutationResult(observed_gap=0.01, p_value=p, n_a=40, n_b=40))
    expected = benjamini_hochberg(ps, alpha=0.05)
    for i, is_sig in enumerate(expected):
        verdict = ledger.verdict(f"h{i}")
        assert verdict.startswith("SIGNIFICANT" if is_sig else "NOT SIGNIFICANT")


def test_unknown_name_raises():
    ledger = TestLedger()
    try:
        ledger.verdict("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_summary_reports_count():
    ledger = TestLedger()
    ledger.record("h0", PermutationResult(observed_gap=0.01, p_value=0.5, n_a=40, n_b=40))
    ledger.record("h1", PermutationResult(observed_gap=0.02, p_value=0.5, n_a=40, n_b=40))
    assert "2 hypothesis test(s)" in ledger.summary()
