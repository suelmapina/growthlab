"""BG/NBD (purchase frequency) + Gamma-Gamma (spend) models on top of pymc-marketing."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pymc_marketing import clv
from pymc_marketing.clv.utils import rfm_summary, rfm_train_test_split

from growthlab.config import seed

# Purchases are counted per calendar day; recency and T are expressed in weeks.
timeunit = "D"
timescaler = 7

# column names required by pymc-marketing
bgnbdcols = ["customer_id", "frequency", "recency", "T"]
ggcols = ["customer_id", "frequency", "monetary_value"]


def rfm_split(invoices: pd.DataFrame, trainend: str, testend: str) -> pd.DataFrame:
    """Calibration-period RFM summary plus holdout-period purchase counts."""
    return rfm_train_test_split(
        invoices,
        customer_id_col="customer_id",
        datetime_col="date",
        monetary_value_col="revenue",
        train_period_end=trainend,
        test_period_end=testend,
        time_unit=timeunit,
        time_scaler=timescaler,
    )


def rfm_full(invoices: pd.DataFrame, observationend: str | None = None) -> pd.DataFrame:
    """RFM summary over the full history, used to score customers today."""
    return rfm_summary(
        invoices,
        customer_id_col="customer_id",
        datetime_col="date",
        monetary_value_col="revenue",
        observation_period_end=observationend,
        time_unit=timeunit,
        time_scaler=timescaler,
    )


def fit_kwargs(method: str) -> dict:
    if method == "mcmc":
        return {
            "chains": 4,
            "draws": 1000,
            "tune": 1000,
            "random_seed": seed,
            "progressbar": False,
        }
    return {}


def fit_bgnbd(rfm: pd.DataFrame, method: str = "map") -> clv.BetaGeoModel:
    model = clv.BetaGeoModel(data=rfm[bgnbdcols])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(method=method, **fit_kwargs(method))
    return model


def fit_gamma_gamma(rfm: pd.DataFrame, method: str = "map") -> clv.GammaGammaModel:
    """Gamma-Gamma is fitted on repeat customers only (spend is the mean of repeat purchases)."""
    repeat = rfm.loc[rfm["frequency"] > 0, ggcols]
    model = clv.GammaGammaModel(data=repeat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(method=method, **fit_kwargs(method))
    return model


def point(da) -> np.ndarray:
    """Posterior mean of a (chain, draw, customer_id) DataArray, in input order."""
    return da.mean(("chain", "draw")).to_numpy()


def interval(da, q: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Central credible interval per customer (only meaningful for MCMC fits)."""
    lo, hi = (1 - q) / 2, 1 - (1 - q) / 2
    qs = da.quantile([lo, hi], dim=("chain", "draw"))
    return qs.sel(quantile=lo).to_numpy(), qs.sel(quantile=hi).to_numpy()


def expected_purchases(model: clv.BetaGeoModel, rfm: pd.DataFrame, weeks: float):
    return model.expected_purchases(data=rfm[bgnbdcols], future_t=weeks)


def expected_spend(model: clv.GammaGammaModel, rfm: pd.DataFrame):
    """Expected spend per purchase. One-time buyers get the population mean."""
    return model.expected_customer_spend(data=rfm[ggcols])


def expected_clv(
    gg: clv.GammaGammaModel,
    bgnbd: clv.BetaGeoModel,
    rfm: pd.DataFrame,
    months: int = 12,
    monthlydiscount: float = 0.008,  # ≈ 10 % per year
):
    data = rfm[["customer_id", "frequency", "recency", "T", "monetary_value"]]
    return gg.expected_customer_lifetime_value(
        transaction_model=bgnbd,
        data=data,
        future_t=months,
        discount_rate=monthlydiscount,
        time_unit="W",
    )
