import random

import pytest

from tokio_ai.rigor.stats import (
    MIN_SAMPLE,
    PermutationResult,
    benjamini_hochberg,
    bonferroni_correct,
    circular_shift_test,
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


# --- studentization (0.3.0) -------------------------------------------------


def test_unequal_variance_does_not_manufacture_significance():
    """The 0.3.0 headline bug, as a regression test.

    Group A is small and noisy, group B is large and quiet, and both are
    centered on zero -- there is no real difference in means. A permutation
    test on the RAW mean difference pools the two and concludes A's mean is
    far more stable than it is, producing spurious significance. The
    studentized statistic has to survive this.
    """
    rng = random.Random(7)
    noisy_small = [rng.gauss(0, 5.0) for _ in range(40)]
    quiet_large = [rng.gauss(0, 1.0) for _ in range(600)]
    result = permutation_test(noisy_small, quiet_large, iters=2000, seed=1)
    assert result.p_value > 0.05
    assert result.statistic == "studentized_mean_diff"
    assert result.variance_ratio is not None and result.variance_ratio > 4


def test_variance_ratio_is_reported():
    a = [1.0, -1.0] * 30       # variance 1
    b = [2.0, -2.0] * 30       # variance 4
    result = permutation_test(a, b, iters=200, seed=0)
    assert result.variance_ratio == pytest.approx(0.25, rel=0.01)


def test_zero_variance_groups_that_differ_are_maximally_extreme():
    """Both groups constant but separated: perfect separation, not 'undefined'.

    An earlier draft of the studentized statistic returned None when the
    denominator was zero, which scored this case at p=1.0 -- the exact
    inverse of the truth.
    """
    result = permutation_test([1.0] * 30, [0.0] * 30, iters=2000, seed=1)
    assert result.p_value < 0.01


def test_zero_variance_identical_groups_are_not_significant():
    result = permutation_test([1.0] * 30, [1.0] * 30, iters=500, seed=1)
    assert result.p_value > 0.5


# --- circular-shift randomization (0.3.0) -----------------------------------


def test_circular_shift_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        circular_shift_test([True, False], [1.0], iters=100)


def test_circular_shift_rejects_nonpositive_iters():
    with pytest.raises(ValueError, match="iters must be positive"):
        circular_shift_test([True, False], [1.0, 2.0], iters=0)


def test_circular_shift_handles_single_class_labels():
    for labels in ([True] * 50, [False] * 50):
        result = circular_shift_test(labels, [float(i) for i in range(50)], iters=100)
        assert result.p_value == 1.0


def test_circular_shift_p_value_is_exact_when_all_rotations_enumerated():
    """With n <= iters every distinct rotation is evaluated, so the p-value is
    an exact fraction over n -- and can never be zero, since the identity
    rotation always counts itself."""
    n = 200
    rng = random.Random(3)
    values = [rng.gauss(0, 1) for _ in range(n)]
    labels = [i % 5 == 0 for i in range(n)]
    result = circular_shift_test(labels, values, iters=5000, seed=0)
    assert result.iters == n
    assert result.p_value >= 1 / n
    assert (result.p_value * n) == pytest.approx(round(result.p_value * n))


def test_circular_shift_finds_a_real_planted_effect():
    """Calibration must not come at the cost of never detecting anything."""
    rng = random.Random(11)
    labels, values = [], []
    for i in range(400):
        flag = i % 7 == 0
        labels.append(flag)
        values.append(rng.gauss(0.9 if flag else 0.0, 0.5))
    assert circular_shift_test(labels, values, iters=5000, seed=0).p_value < 0.01


def test_circular_shift_is_not_fooled_by_clustered_labels_on_autocorrelated_values():
    """The overlap defect, as a regression test.

    `values` is a slow random walk (strongly autocorrelated, like overlapping
    forward returns) and `labels` mark one contiguous block (clustered in
    time, like a volatility burst). There is no relationship between the two
    beyond each one's own internal structure, but a shuffled label test
    reliably calls this significant. Rotation must not.
    """
    rng = random.Random(5)
    values, level = [], 0.0
    for _ in range(600):
        level += rng.gauss(0, 1)
        values.append(level)
    labels = [140 <= i < 200 for i in range(600)]
    assert circular_shift_test(labels, values, iters=5000, seed=0).p_value > 0.05


# --- FFT all-rotation fast path (0.4.0) -------------------------------------

def _brute_force(labels, values, monkeypatch):
    import tokio_ai.rigor.stats as stats_mod

    monkeypatch.setattr(stats_mod, "_numpy", lambda: None)
    return circular_shift_test(labels, values, iters=len(values) + 1)


@pytest.mark.parametrize("seed", range(12))
def test_fft_path_matches_exhaustive_loop_exactly(seed, monkeypatch):
    pytest.importorskip("numpy")
    rng = random.Random(seed)
    n = rng.randrange(80, 400)
    labels = [rng.random() < rng.choice([0.05, 0.3, 0.6]) for _ in range(n)]
    values = [rng.gauss(0, 1) * (3 if rng.random() < 0.1 else 1) for _ in range(n)]
    fast = circular_shift_test(labels, values, iters=len(values) + 1)
    slow = _brute_force(labels, values, monkeypatch)
    assert fast.p_value == slow.p_value
    assert fast.iters == slow.iters == n


def test_fft_path_handles_heavy_ties(monkeypatch):
    # Discrete values make many rotations tie the observed statistic
    # exactly -- the case where floating-point route differences bite.
    pytest.importorskip("numpy")
    rng = random.Random(5)
    labels = [rng.random() < 0.3 for _ in range(300)]
    values = [float(rng.choice([-1, 0, 1])) for _ in range(300)]
    fast = circular_shift_test(labels, values, iters=301)
    slow = _brute_force(labels, values, monkeypatch)
    assert fast.p_value == slow.p_value


def test_fft_path_survives_a_large_offset(monkeypatch):
    # Prices-scale values with tiny differences: cancellation territory.
    pytest.importorskip("numpy")
    rng = random.Random(9)
    labels = [rng.random() < 0.2 for _ in range(250)]
    values = [1e4 + rng.gauss(0, 1e-3) for _ in range(250)]
    fast = circular_shift_test(labels, values, iters=251)
    slow = _brute_force(labels, values, monkeypatch)
    assert fast.p_value == pytest.approx(slow.p_value, abs=2 / 250)


def test_fft_path_perfect_separation_is_maximally_extreme(monkeypatch):
    pytest.importorskip("numpy")
    labels = [i % 10 == 0 for i in range(200)]
    values = [1.0 if flag else 0.0 for flag in labels]
    # Zero spread in both groups, so the statistic is infinite. Every
    # rotation by a multiple of the period (200 / 10 = 20 of them) lines the
    # 1s up again and is exactly as extreme; nothing else comes close.
    result = circular_shift_test(labels, values)
    assert result.p_value == pytest.approx(20 / 200)
    assert result.p_value == _brute_force(labels, values, monkeypatch).p_value


def test_fft_path_is_exact_beyond_iters():
    # With numpy, n > iters no longer falls back to Monte Carlo sampling.
    pytest.importorskip("numpy")
    rng = random.Random(2)
    labels = [rng.random() < 0.2 for _ in range(3000)]
    values = [rng.gauss(0, 1) for _ in range(3000)]
    result = circular_shift_test(labels, values, iters=500)
    assert result.iters == 3000
