from tokio_ai.rigor.stats import (
    MIN_SAMPLE,
    PermutationResult,
    benjamini_hochberg,
    bonferroni_correct,
    permutation_test,
)


def test_identical_groups_gives_p_value_one():
    a = [1.0] * 40
    b = [1.0] * 40
    result = permutation_test(a, b, iters=200, seed=1)
    assert result.observed_gap == 0.0
    assert result.p_value == 1.0


def test_clearly_separated_groups_gives_low_p_value():
    a = [1.0] * 30
    b = [0.0] * 30
    result = permutation_test(a, b, iters=2000, seed=1)
    assert result.p_value < 0.01


def test_p_value_is_never_exactly_zero():
    # Real bug found by an independent review pass: a raw hits/iters ratio
    # can land on exactly 0.0 with finite Monte Carlo resampling, which
    # overclaims "impossible under the null" -- the honest floor is
    # 1/(iters+1) via the standard plus-one correction (Phipson & Smyth
    # 2010). This is a maximally-separated case designed to hit zero raw
    # hits if the correction weren't applied.
    a = [1.0] * 30
    b = [0.0] * 30
    result = permutation_test(a, b, iters=1000, seed=1)
    assert result.p_value > 0.0
    assert result.p_value >= 1 / 1001


def test_iters_must_be_positive():
    for bad in (0, -1, -100):
        try:
            permutation_test([1.0, 2.0], [3.0, 4.0], iters=bad)
            assert False, f"expected ValueError for iters={bad}"
        except ValueError:
            pass


def test_meets_min_sample():
    result = PermutationResult(observed_gap=0.1, p_value=0.03, n_a=MIN_SAMPLE, n_b=MIN_SAMPLE)
    assert result.meets_min_sample
    small = PermutationResult(observed_gap=0.1, p_value=0.03, n_a=MIN_SAMPLE - 1, n_b=MIN_SAMPLE)
    assert not small.meets_min_sample


def test_bonferroni_matches_hand_calculation():
    p_values = [0.001, 0.02, 0.03, 0.04, 0.2]
    # alpha/5 = 0.01 -> only 0.001 survives
    assert bonferroni_correct(p_values, alpha=0.05) == [True, False, False, False, False]


def test_benjamini_hochberg_matches_textbook_example():
    p_values = [0.001, 0.02, 0.03, 0.04, 0.2]
    # classic BH worked example: first four survive, last does not
    assert benjamini_hochberg(p_values, alpha=0.05) == [True, True, True, True, False]


def test_empty_inputs_do_not_crash():
    assert bonferroni_correct([]) == []
    assert benjamini_hochberg([]) == []
    result = permutation_test([], [1.0])
    assert result.p_value == 1.0
