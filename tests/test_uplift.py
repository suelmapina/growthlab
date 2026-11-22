import numpy as np
import pytest

from growthlab.uplift import metrics, segments


@pytest.fixture
def experiment():
    """Synthetic RCT where only customers with x > 0 respond to treatment."""
    rng = np.random.default_rng(0)
    n = 40_000
    x = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    p = 0.05 + 0.10 * t * (x > 0)
    y = (rng.random(n) < p).astype(int)
    return x, y, t


def test_ate_recovers_effect(experiment):
    x, y, t = experiment
    est, se = metrics.ate(y, t)
    assert est == pytest.approx(0.05, abs=3 * se)


def test_qini_ranks_true_uplift_above_random(experiment):
    x, y, t = experiment
    good = metrics.qini_coefficient(y, t, x)
    bad = metrics.qini_coefficient(y, t, -x)
    rnd = metrics.qini_coefficient(y, t, np.random.default_rng(1).random(len(x)))
    assert good > rnd > bad
    assert abs(rnd) < 0.1


def test_uplift_at_top_half(experiment):
    x, y, t = experiment
    assert metrics.uplift_at(y, t, x, 0.5) == pytest.approx(0.10, abs=0.015)


def test_qini_curve_starts_at_zero(experiment):
    x, y, t = experiment
    frac, qini = metrics.qini_curve(y, t, x)
    assert frac[0] == 0 and qini[0] == 0 and frac[-1] == 1


def test_paired_bootstrap_detects_better_model(experiment):
    x, y, t = experiment
    rnd = np.random.default_rng(2).random(len(x))
    res = metrics.paired_bootstrap(y, t, x, rnd, metrics.qini_coefficient, reps=30)
    assert res["ci_low"] > 0


def test_segments_cover_everyone():
    rng = np.random.default_rng(0)
    uplift, baseline = rng.normal(size=1000), rng.random(1000)
    seg = segments.assign_segments(uplift, baseline)
    assert set(seg) == set(segments.order)
    assert (seg[uplift < 0] == "Sleeping Dogs").all()
