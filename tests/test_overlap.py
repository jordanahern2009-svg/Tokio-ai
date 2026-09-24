import math
import random

import pytest

from tokio_ai.rigor.overlap import _hodrick_python, default_bandwidth, hodrick_test


def _series(n, seed):
    rng = random.Random(seed)
    r = [rng.gauss(0, 0.01) for _ in range(n)]
    lab = [rng.random() < 0.3 for _ in range(n)]
    return r, lab


def test_regrouping_identity_holds_exactly():
    # sum_i a_i * y_i (by observation) must equal sum_t r_t * B_t (by bar):
    # the whole method rests on this. Checked with bandwidth 0, where
    # z = S / sqrt(sum u^2), by rebuilding S the direct way.
    r, lab = _series(300, 1)
    h = 7
    logs = [math.log1p(x) for x in r]
    valid = list(range(300 - h))
    abar = sum(lab[i] for i in valid) / len(valid)
    rbar = sum(logs) / len(logs)
    direct = sum((lab[i] - abar) * sum(logs[i + 1 : i + 1 + h]) for i in valid)
    a = [(lab[i] - abar) if i in set(valid) else 0.0 for i in range(300)]
    B = [sum(a[max(t - h, 0) : t]) for t in range(300)]
    V = sum(((logs[t] - rbar) * B[t]) ** 2 for t in range(300))
    res = hodrick_test(r, lab, h, bandwidth=0)
    assert res.z == pytest.approx(direct / math.sqrt(V), rel=1e-9)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("h", [1, 5, 20])
def test_python_and_numpy_paths_agree(seed, h):
    np = pytest.importorskip("numpy")
    from tokio_ai.rigor.overlap import _hodrick_numpy

    r, lab = _series(700, seed)
    rng = random.Random(seed + 50)
    for i in rng.sample(range(700), 15):
        r[i] = None
    for i in rng.sample(range(700), 15):
        lab[i] = None
    a = _hodrick_python(r, lab, h, None)
    b = _hodrick_numpy(np, r, lab, h, None)
    assert a.z == pytest.approx(b.z, rel=1e-9)
    assert a.bandwidth == b.bandwidth


def test_windows_touching_a_missing_bar_are_excluded():
    # A missing bar drops the labels whose windows contain it; the result
    # must change, and stay finite rather than propagating NaN.
    r, lab = _series(400, 3)
    r2 = list(r)
    r2[200] = None
    res_full = hodrick_test(r, lab, 5, bandwidth=0)
    res_gap = hodrick_test(r2, lab, 5, bandwidth=0)
    assert res_full.z != res_gap.z
    assert math.isfinite(res_gap.z)


def test_default_bandwidth_is_the_newey_west_rule():
    assert default_bandwidth(100) == 4
    assert default_bandwidth(1000) == 6
    assert default_bandwidth(1_000_000) == 30


def test_total_loss_bar_falls_back_to_simple_returns():
    r, lab = _series(300, 4)
    r[100] = -1.0
    res = hodrick_test(r, lab, 3)
    assert math.isfinite(res.z)


def test_no_usable_labels_gives_no_evidence():
    res = hodrick_test([0.01] * 50, [None] * 50, 5)
    assert res.p_value == 1.0


def test_null_rejection_rate_is_near_nominal():
    # Small in-suite calibration: iid returns, independent labels.
    hits = 0
    for seed in range(200):
        r, lab = _series(400, 1000 + seed)
        hits += hodrick_test(r, lab, 5).p_value <= 0.05
    assert 2 <= hits <= 20  # 1%-10% of 200
