"""Uplift (CATE) estimators.

Each `fit_*` returns a fitted, picklable model; `predict_uplift(name, model, x)` scores it.

- response: classic targeting model, P(outcome | treated). Not causal; the baseline to beat.
- slearner: one model with the treatment as a feature.
- tlearner: separate models for treated and control.
- xlearner / drlearner: econml meta-learners that correct for the 85/15 group imbalance.
"""

from __future__ import annotations

import warnings

import numpy as np
from econml.dr import DRLearner
from econml.metalearners import XLearner
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression

from growthlab.config import seed

features = [f"f{i}" for i in range(12)]  # `exposure` is post-treatment and is never used

treeparams = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": seed,
    "verbose": -1,
}


def classifier() -> LGBMClassifier:
    return LGBMClassifier(**treeparams)


def regressor(**overrides) -> LGBMRegressor:
    return LGBMRegressor(**(treeparams | overrides))


def prob(model, x) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def fit_response(x, y, t) -> LGBMClassifier:
    return classifier().fit(x[t == 1], y[t == 1])


def fit_slearner(x, y, t) -> LGBMClassifier:
    return classifier().fit(np.column_stack([x, t]), y)


def fit_tlearner(x, y, t) -> tuple[LGBMClassifier, LGBMClassifier]:
    """(treated model, control model). The control model also gives the baseline outcome."""
    return classifier().fit(x[t == 1], y[t == 1]), classifier().fit(x[t == 0], y[t == 0])


def fit_xlearner(x, y, t) -> XLearner:
    model = XLearner(
        models=regressor(),
        cate_models=regressor(min_child_samples=2000),
        propensity_model=LogisticRegression(max_iter=1000),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(y, t, X=x)
    return model


def fit_drlearner(x, y, t) -> DRLearner:
    model = DRLearner(
        model_propensity=LogisticRegression(max_iter=1000),
        model_regression=regressor(),
        model_final=regressor(min_child_samples=5000),  # DR pseudo-outcomes are very noisy
        cv=3,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(y, t, X=x)
    return model


fitters = {
    "response": fit_response,
    "slearner": fit_slearner,
    "tlearner": fit_tlearner,
    "xlearner": fit_xlearner,
    "drlearner": fit_drlearner,
}


def predict_uplift(name: str, model, x) -> np.ndarray:
    """Predicted uplift (for `response`: predicted response probability, used as a ranking)."""
    x = np.asarray(x, np.float32)
    if name == "response":
        return prob(model, x)
    if name == "slearner":
        ones, zeros = np.ones(len(x)), np.zeros(len(x))
        return prob(model, np.column_stack([x, ones])) - prob(model, np.column_stack([x, zeros]))
    if name == "tlearner":
        treated, control = model
        return prob(treated, x) - prob(control, x)
    return np.asarray(model.effect(x)).ravel()


def predict_baseline(tlearner, x) -> np.ndarray:
    """Probability of the outcome without treatment, from the T-learner's control model."""
    return prob(tlearner[1], np.asarray(x, np.float32))
