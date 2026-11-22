"""The decision rule: contact a customer only if the incremental value covers the cost.

    expected incremental profit = margin × CLV × uplift − cost

- CLV: expected 12-month revenue if the customer stays (Module 1)
- uplift: increase in response probability caused by the incentive (Module 2)
- margin: gross margin on revenue
- cost: cost per contacted customer (incentive + contact)
"""

from __future__ import annotations

import numpy as np

defaultmargin = 0.30
defaultcost = 5.0


def expected_profit(clv, uplift, margin: float = defaultmargin, cost: float = defaultcost):
    return margin * np.asarray(clv) * np.asarray(uplift) - cost


def decide(clv, uplift, margin: float = defaultmargin, cost: float = defaultcost):
    """Boolean mask: True = send the incentive."""
    return expected_profit(clv, uplift, margin, cost) > 0


def segment(uplift: float, baseline: float, thresholds: dict) -> str:
    """Single-customer version of the Module 2 segment rule, using stored thresholds."""
    if uplift < 0:
        return "Sleeping Dogs"
    if uplift >= thresholds["persuadable_cutoff"]:
        return "Persuadables"
    return "Sure Things" if baseline >= thresholds["baseline_median"] else "Lost Causes"
