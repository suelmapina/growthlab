"""Holdout evaluation metrics for CLV models."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return {
        "mae": float(np.mean(np.abs(predicted - actual))),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "total_error_pct": float((predicted.sum() - actual.sum()) / actual.sum()),
        "spearman": float(spearmanr(actual, predicted).statistic),
    }


def top_share_captured(actual: np.ndarray, score: np.ndarray, top: float = 0.2) -> float:
    """Share of total `actual` held by the top `top` fraction of customers ranked by `score`."""
    actual, score = np.asarray(actual, float), np.asarray(score, float)
    n = math.ceil(len(score) * top)
    idx = np.argsort(-score, kind="stable")[:n]
    return float(actual[idx].sum() / actual.sum())


def calibration_table(
    calibfreq: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray | None = None,
    maxbucket: int = 7,
) -> pd.DataFrame:
    """Mean actual vs predicted holdout purchases, grouped by calibration-period frequency."""
    freq = np.minimum(np.asarray(calibfreq), maxbucket).astype(int)
    df = pd.DataFrame({"bucket": freq, "actual": actual, "predicted": predicted})
    if baseline is not None:
        df["baseline"] = baseline
    table = df.groupby("bucket").agg(
        customers=("actual", "size"), **{c: (c, "mean") for c in df.columns[1:]}
    )
    table.index = [str(b) if b < maxbucket else f"{maxbucket}+" for b in table.index]
    table.index.name = "repeat purchases in calibration"
    return table
