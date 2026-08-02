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


def test_reusing_a_name_reports_the_latest_test_not_the_first():
    # Real bug found in manual testing: verdict() used to return on the
    # FIRST match by name, so re-recording under the same name (a retry, or
    # the model reusing a label) silently kept reporting stale results
    # forever, even after a materially different second test.
    ledger = TestLedger()
    ledger.record("AAPL_test", PermutationResult(observed_gap=0.01, p_value=0.5, n_a=40, n_b=40))
    assert ledger.verdict("AAPL_test").startswith("NOT SIGNIFICANT")

    ledger.record("AAPL_test", PermutationResult(observed_gap=0.05, p_value=0.001, n_a=40, n_b=40))
    verdict = ledger.verdict("AAPL_test")
    assert verdict.startswith("SIGNIFICANT")
    assert "p=0.0010" in verdict
    assert "2 hypothesis" in verdict  # both entries still count toward correction


def test_summary_reports_count():
    ledger = TestLedger()
    ledger.record("h0", PermutationResult(observed_gap=0.01, p_value=0.5, n_a=40, n_b=40))
    ledger.record("h1", PermutationResult(observed_gap=0.02, p_value=0.5, n_a=40, n_b=40))
    assert "2 hypothesis test(s)" in ledger.summary()
