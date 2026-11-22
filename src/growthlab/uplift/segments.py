"""Assign customers to the four uplift segments and check them against observed outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd

order = ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]


def assign_segments(uplift: np.ndarray, baseline: np.ndarray, persuadabletop: float = 0.25):
    """Rule on model predictions (individual potential outcomes are never observed):

    - Sleeping Dogs: predicted uplift < 0 (contacting them lowers the outcome)
    - Persuadables: the top `persuadabletop` share of predicted uplift among the rest
    - Sure Things / Lost Causes: remaining customers with high / low baseline (untreated) outcome
    """
    uplift, baseline = np.asarray(uplift), np.asarray(baseline)
    seg = np.empty(len(uplift), dtype=object)

    negative = uplift < 0
    seg[negative] = "Sleeping Dogs"

    rest = ~negative
    cutoff = np.quantile(uplift[rest], 1 - persuadabletop)
    persuadable = rest & (uplift >= cutoff)
    seg[persuadable] = "Persuadables"

    other = rest & ~persuadable
    median = np.median(baseline[other])
    seg[other & (baseline >= median)] = "Sure Things"
    seg[other & (baseline < median)] = "Lost Causes"
    return seg


def segment_report(seg, y, t) -> pd.DataFrame:
    """Observed treated vs control outcome rates per segment (valid because of randomisation)."""
    df = pd.DataFrame({"segment": seg, "y": np.asarray(y, float), "t": np.asarray(t, int)})
    rows = []
    for name in order:
        part = df[df["segment"] == name]
        if part.empty:
            continue
        treated, control = part.loc[part.t == 1, "y"], part.loc[part.t == 0, "y"]
        se = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
        rows.append(
            {
                "segment": name,
                "share": len(part) / len(df),
                "treated_rate": treated.mean(),
                "control_rate": control.mean(),
                "observed_uplift": treated.mean() - control.mean(),
                "uplift_se": se,
            }
        )
    return pd.DataFrame(rows).set_index("segment")
