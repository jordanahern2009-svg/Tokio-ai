import math
import random

import pytest

import tokio_ai
from tokio_ai import CheckResult, TestLedger, check
from tokio_ai.check import forward_returns
from tokio_ai.rigor.stats import MIN_SAMPLE


def _noise(n, seed=0, sd=0.01):
    rng = random.Random(seed)
    return [rng.gauss(0, sd) for _ in range(n)]


def test_package_exports_check_without_pulling_in_the_agent():
    # Fresh interpreter: this test process has other tests' imports loaded.
    # The library path must not cost users an LLM client or a TUI import.
    import subprocess
    import sys

    assert tokio_ai.check is check
    code = (
        "import sys, tokio_ai; "
        "print(any(m.split('.')[0] in ('openai', 'textual') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_forward_returns_start_on_the_next_bar():
    # Bar i's own return must never be part of its outcome, or a condition
    # like "today fell 2%" would predict itself.
    r = [0.10, 0.20, -0.50, 0.0]
    fwd = forward_returns(r, 1)
    assert fwd[0] == pytest.approx(0.20)
    assert fwd[1] == pytest.approx(-0.50)
    assert fwd[2] == pytest.approx(0.0)
    assert fwd[3] is None


def test_forward_returns_compound_rather_than_sum():
    fwd = forward_returns([0.0, 0.10, 0.10, 0.0], 2)
    assert fwd[0] == pytest.approx(1.1 * 1.1 - 1)


def test_forward_returns_missing_value_poisons_only_windows_that_touch_it():
    fwd = forward_returns([0.0, 0.01, None, 0.02, 0.03], 1)
    assert fwd[0] == pytest.approx(0.01)
    assert fwd[1] is None
    assert fwd[2] == pytest.approx(0.02)


def test_planted_edge_is_found():
    # Separate seeds for returns and condition: sharing one would correlate
    # them through the generator and plant an edge the test didn't intend.
    rng = random.Random(9003)
    n = 1500
    r = _noise(n, seed=3)
    cond = [rng.random() < 0.1 for _ in range(n)]
    for i in range(n - 1):
        if cond[i]:
            r[i + 1] += 0.01  # one full standard deviation, the next bar
    result = check(r, cond, horizon=1)
    assert result.verdict == "SIGNIFICANT"
    assert result.significant
    assert result.gap > 0.005


def test_edge_that_only_exists_same_bar_is_not_found():
    # The condition IS the bar's own return. With correct alignment there is
    # nothing to find; with the classic off-by-one there would be a huge
    # "edge". This is the lookahead bug check() is built to make impossible.
    r = _noise(2000, seed=11)
    cond = [x < -0.01 for x in r]
    result = check(r, cond, horizon=1)
    assert result.p_value > 0.01
    assert abs(result.gap) < 0.003


def test_small_condition_group_is_not_reportable():
    r = _noise(500, seed=1)
    cond = [i < MIN_SAMPLE - 5 for i in range(500)]
    result = check(r, cond)
    assert result.verdict == "NOT REPORTABLE"
    assert not result.significant
    assert "NOT REPORTABLE" in str(result)


def test_missing_values_are_skipped_not_counted():
    r = [math.nan] + _noise(400, seed=2)
    cond = [None] + [i % 5 == 0 for i in range(400)]
    result = check(r, cond)
    assert result.n_condition + result.n_other == 399  # last bar has no forward return


def test_nan_condition_is_skipped_not_treated_as_true():
    # bool(float("nan")) is True -- the naive conversion would silently put
    # every missing signal into the condition group.
    r = _noise(300, seed=4)
    cond = [math.nan] * 100 + [False] * 200
    result = check(r, cond)
    assert result.n_condition == 0


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        check([0.0] * 10, [True] * 9)


@pytest.mark.parametrize("horizon", [0, -1])
def test_bad_horizon_raises(horizon):
    with pytest.raises(ValueError, match="horizon"):
        check([0.0] * 10, [True] * 10, horizon=horizon)


def test_bad_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        check([0.0] * 10, [True] * 10, alpha=5)


def test_ledger_corrects_across_checks():
    rng = random.Random(9008)
    r = _noise(1200, seed=8)
    ledger = TestLedger()
    results = [
        check(r, [rng.random() < 0.2 for _ in range(1200)], ledger=ledger, iters=500)
        for _ in range(5)
    ]
    assert [x.tests_corrected_for for x in results] == [1, 2, 3, 4, 5]
    assert len(ledger.tests) == 5
    assert "after correcting for 5 tests" in str(results[-1])
    assert "correcting" not in str(results[0])


def test_ledger_can_turn_a_lone_significant_result_insignificant():
    # A p just under 0.05 passes alone, and must fail once it is one of many.
    rng = random.Random(9021)
    n = 1500
    r = _noise(n, seed=21)
    cond = [rng.random() < 0.1 for _ in range(n)]
    for i in range(n - 1):
        if cond[i]:
            r[i + 1] += 0.0025
    alone = check(r, cond)
    assert 0.005 < alone.p_value <= 0.05, "fixture drifted out of the borderline band"
    ledger = TestLedger()
    for s in range(9):
        noise_rng = random.Random(100 + s)
        check(r, [noise_rng.random() < 0.1 for _ in range(n)], ledger=ledger, iters=500)
    corrected = check(r, cond, ledger=ledger)
    assert alone.significant
    assert not corrected.significant


def test_str_reports_the_numbers_and_the_method():
    result = check(_noise(600, seed=5), [i % 4 == 0 for i in range(600)], horizon=5)
    text = str(result)
    assert result.verdict in text
    assert "next 5 bars" in text
    assert "hodrick_hac" in text and "circular_shift" in text
    assert "tokio-ai" in text


def test_result_is_a_frozen_record():
    result = check(_noise(300, seed=6), [i % 3 == 0 for i in range(300)])
    assert isinstance(result, CheckResult)
    with pytest.raises(Exception):
        result.p_value = 0.0


def test_numpy_input_matches_list_input():
    np = pytest.importorskip("numpy")
    r = _noise(500, seed=9)
    cond = [i % 6 == 0 for i in range(500)]
    a = check(r, cond, iters=500)
    b = check(np.array(r), np.array(cond), iters=500)
    assert a.p_value == b.p_value
    assert a.gap == pytest.approx(b.gap)


def test_pandas_series_from_pct_change():
    pd = pytest.importorskip("pandas")
    prices = pd.Series(
        [100 * (1.001 ** i) + (i % 7) for i in range(400)],
        index=pd.date_range("2020-01-01", periods=400, freq="B"),
    )
    r = prices.pct_change()  # leading NaN
    result = check(r, r < 0, horizon=3, iters=500)
    # 400 - 3, not 400 - 1 - 3: pandas evaluates `NaN < 0` as False, not NaN,
    # so bar 0 arrives as a real False, and its forward window (bars 1-3) is
    # complete. Only the last `horizon` bars lack an outcome.
    assert result.n_condition + result.n_other == 400 - 3


def test_pandas_misaligned_indexes_raise_instead_of_pairing_by_position():
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    r = pd.Series(_noise(100), index=idx)
    cond = (r < 0).shift(1, freq="B")  # same length, dates moved
    with pytest.raises(ValueError, match="different indexes"):
        check(r, cond)


def _result(n_condition, n_other, variance_ratio):
    return CheckResult(
        verdict="NOT SIGNIFICANT", p_value=0.5, horizon=1, n_condition=n_condition,
        n_other=n_other, mean_condition=0.0, mean_other=0.0, variance_ratio=variance_ratio,
        alpha=0.05, tests_corrected_for=1, method="circular_shift", iters=10, seed=0,
        provenance="test",
    )


def test_variance_note_small_noisy_group_overstates():
    text = str(_result(50, 900, 2.5))
    assert "selects volatile periods" in text
    assert "overstates" in text


def test_variance_note_large_calm_group_reports_the_true_direction():
    # SPY 20-day momentum is this shape: the condition holds on most days and
    # those days are the calm ones. Calling them "volatile" was a real bug.
    text = str(_result(1742, 751, 0.5))
    assert "selects calm periods" in text
    assert "2.00x" in text
    assert "overstates" in text


def test_variance_note_small_calm_group_understates():
    text = str(_result(50, 900, 0.4))
    assert "selects calm periods" in text
    assert "understates" in text


def test_no_variance_note_when_variances_are_close():
    assert "Unequal variances" not in str(_result(100, 900, 1.2))


# --- numpy fast path must agree with the pure-Python reference -------------

@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_numpy_prepare_matches_python_prepare(seed, horizon):
    pytest.importorskip("numpy")
    from tokio_ai.check import _prepare_numpy, _prepare_python

    rng = random.Random(seed)
    r = [rng.gauss(0, 0.02) for _ in range(600)]
    cond = [rng.random() < 0.3 for _ in range(600)]
    for i in rng.sample(range(600), 25):  # scattered gaps in both series
        r[i] = math.nan if i % 2 else None
    for i in rng.sample(range(600), 25):
        cond[i] = math.nan if i % 2 else None
    fast = _prepare_numpy(r, cond, horizon)
    slow = _prepare_python(r, cond, horizon)
    assert list(fast[0]) == slow[0]
    assert list(fast[1]) == pytest.approx(slow[1], rel=1e-9, abs=1e-12)
    assert fast[2] == pytest.approx(slow[2]) and fast[3] == pytest.approx(slow[3])


def test_total_loss_return_falls_back_to_the_exact_path():
    pytest.importorskip("numpy")
    from tokio_ai.check import _prepare_numpy

    r = _noise(200, seed=3)
    r[50] = -1.0
    assert _prepare_numpy(r, [i % 3 == 0 for i in range(200)], 5) is None
    result = check(r, [i % 3 == 0 for i in range(200)], horizon=5)
    assert result.n_condition > 0


def test_check_agrees_with_and_without_numpy(monkeypatch):
    pytest.importorskip("numpy")
    import sys

    import tokio_ai.rigor.stats as stats_mod

    # `tokio_ai.check` is the function (re-exported by the package), which
    # shadows the submodule of the same name for attribute-style imports.
    check_mod = sys.modules["tokio_ai.check"]

    rng = random.Random(44)
    r = _noise(700, seed=44)
    cond = [rng.random() < 0.25 for _ in range(700)]
    fast = check(r, cond, horizon=5, iters=701)
    monkeypatch.setattr(check_mod, "_prepare_numpy", lambda *a: None)
    monkeypatch.setattr(stats_mod, "_numpy", lambda: None)
    slow = check(r, cond, horizon=5, iters=701)
    assert fast.p_value == slow.p_value
    assert fast.gap == pytest.approx(slow.gap)
    assert fast.variance_ratio == pytest.approx(slow.variance_ratio)


def test_pandas_where_object_condition_takes_the_fast_path():
    pd = pytest.importorskip("pandas")
    from tokio_ai.check import _prepare_numpy

    prices = pd.Series([100 + i * 0.1 + (i % 5) for i in range(300)])
    past = prices.shift(20)
    cond = (prices > past).where(past.notna())  # object dtype: True/False/NaN
    out = _prepare_numpy(prices.pct_change(), cond, 5)
    assert out is not None
    assert len(out[0]) == 300 - 20 - 5


# --- two engines: Hodrick (default) and rotation ----------------------------

def test_default_method_is_hodrick_and_rotation_is_reported_too():
    rng = random.Random(9101)
    r = _noise(800, seed=101)
    cond = [rng.random() < 0.3 for _ in range(800)]
    res = check(r, cond, horizon=5)
    assert res.method == "hodrick_hac"
    assert res.p_value == res.p_hodrick
    assert 0 < res.p_rotation <= 1


def test_rotation_method_uses_the_rotation_p_value():
    rng = random.Random(9102)
    r = _noise(800, seed=102)
    cond = [rng.random() < 0.3 for _ in range(800)]
    res = check(r, cond, horizon=5, method="rotation")
    assert res.method == "circular_shift"
    assert res.p_value == res.p_rotation


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        check([0.0] * 50, [True] * 50, method="t-test")


def test_hodrick_finds_a_persistent_short_horizon_edge_the_rotation_misses():
    # The case the Hodrick engine was adopted for: a condition that persists
    # for weeks, carrying a next-bar edge. Rotations a few bars off the truth
    # still line up with the edge, which drains the rotation test's power.
    rng = random.Random(9103)
    n = 3000
    r = _noise(n, seed=103)
    state, cond = False, []
    for _ in range(n):
        if rng.random() < 1 / 40:
            state = not state
        cond.append(state)
    for i in range(n - 1):
        if cond[i]:
            r[i + 1] += 0.0012
    res = check(r, cond, horizon=1)
    assert res.p_hodrick < 0.01
    assert res.p_rotation > res.p_hodrick


def test_second_opinion_line_only_when_the_engines_split():
    base = dict(
        verdict="SIGNIFICANT", p_value=0.01, horizon=1, n_condition=500, n_other=500,
        mean_condition=0.0, mean_other=0.0, variance_ratio=1.0, alpha=0.05,
        tests_corrected_for=1, method="hodrick_hac", iters=1000, seed=0, provenance="test",
    )
    split = CheckResult(**base, p_hodrick=0.01, p_rotation=0.30)
    agree = CheckResult(**base, p_hodrick=0.01, p_rotation=0.02)
    assert "Second opinion disagrees" in str(split)
    assert "p=0.3000" in str(split)
    assert "Second opinion" not in str(agree)


def test_ledger_records_the_primary_engine():
    rng = random.Random(9104)
    r = _noise(600, seed=104)
    ledger = TestLedger()
    res = check(r, [rng.random() < 0.3 for _ in range(600)], ledger=ledger)
    assert ledger.tests[-1].result.p_value == res.p_hodrick
    assert ledger.tests[-1].result.statistic == "hodrick_hac"
