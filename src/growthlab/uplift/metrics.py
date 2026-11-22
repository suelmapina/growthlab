"""Uplift evaluation on randomized data: Qini / uplift curves and top-k uplift.

All metrics compare treated vs control outcomes among the customers a model ranks highest.
They are only valid because treatment was randomly assigned.
"""

from __future__ import annotations

import numpy as np


def ranked(y: np.ndarray, t: np.ndarray, score: np.ndarray):
    order = np.argsort(-np.asarray(score), kind="stable")
    return np.asarray(y, float)[order], np.asarray(t, int)[order]


def qini_curve(y, t, score, points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Incremental outcomes among the top-k, scaled to the treated count (Radcliffe's Qini).

    Returns (fraction targeted, qini value), starting at (0, 0).
    """
    ys, ts = ranked(y, t, score)
    cumt = np.cumsum(ts)
    cumc = np.cumsum(1 - ts)
    yt = np.cumsum(ys * ts)
    yc = np.cumsum(ys * (1 - ts))
    qini = yt - yc * cumt / np.maximum(cumc, 1)

    n = len(ys)
    idx = np.unique(np.linspace(0, n - 1, points).astype(int))
    return np.r_[0.0, (idx + 1) / n], np.r_[0.0, qini[idx]]


def qini_coefficient(y, t, score, points: int = 100) -> float:
    """Area between the model's Qini curve and random targeting, normalised by the total gain.

    0 = no better than random; higher is better.
    """
    frac, qini = qini_curve(y, t, score, points)
    random = frac * qini[-1]
    return float(np.trapezoid(qini - random, frac) / qini[-1])


def uplift_at(y, t, score, top: float) -> float:
    """Observed treatment effect (treated rate − control rate) among the top `top` fraction."""
    ys, ts = ranked(y, t, score)
    k = int(np.ceil(len(ys) * top))
    ys, ts = ys[:k], ts[:k]
    return float(ys[ts == 1].mean() - ys[ts == 0].mean())


def paired_bootstrap(y, t, scorea, scoreb, metric, reps: int = 200, seed: int = 0) -> dict:
    """Bootstrap the difference metric(a) − metric(b) on the same resampled customers."""
    y, t = np.asarray(y), np.asarray(t)
    scorea, scoreb = np.asarray(scorea), np.asarray(scoreb)
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = metric(y[idx], t[idx], scorea[idx]) - metric(y[idx], t[idx], scoreb[idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "diff": float(np.mean(diffs)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_not_better": float(np.mean(diffs <= 0)),
    }


def ate(y, t) -> tuple[float, float]:
    """Average treatment effect (difference in means) and its standard error."""
    y, t = np.asarray(y, float), np.asarray(t, int)
    yt, yc = y[t == 1], y[t == 0]
    se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    return float(yt.mean() - yc.mean()), float(se)
