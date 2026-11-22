"""GrowthLab decision API.

    uv run uvicorn growthlab.api.main:app --reload      # docs at http://localhost:8000/docs

- POST /decide-action      score a customer from their purchase history + campaign features
- GET  /customers/{id}     precomputed decision for an Online Retail II customer
- GET  /health
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from pymc_marketing import clv as pmclv

from growthlab import config, version
from growthlab.clv import model as clvmodel
from growthlab.decision import engine, simulate
from growthlab.uplift import learners

app = FastAPI(
    title="GrowthLab",
    version=version,
    description="Decide who should receive a retention incentive: margin × CLV × uplift − cost.",
)


class purchasehistory(BaseModel):
    frequency: int = Field(ge=0, description="number of repeat purchase days", examples=[6])
    recency: float = Field(ge=0, description="weeks between first and last purchase", examples=[60])
    age: float = Field(gt=0, description="weeks since first purchase (T)", examples=[80])
    spend: float = Field(ge=0, description="average value of repeat purchases (€)", examples=[350])

    @model_validator(mode="after")
    def consistent(self):
        if self.recency > self.age:
            raise ValueError("recency cannot exceed age")
        if self.frequency == 0 and (self.recency > 0 or self.spend > 0):
            raise ValueError("a customer with no repeat purchases has recency = 0 and spend = 0")
        return self


class decisionrequest(BaseModel):
    history: purchasehistory
    features: list[float] = Field(
        min_length=12, max_length=12, description="campaign features f0…f11 (Criteo schema)"
    )
    margin: float = Field(engine.defaultmargin, gt=0, le=1)
    cost: float = Field(engine.defaultcost, ge=0, description="cost per contacted customer (€)")


class decision(BaseModel):
    action: str
    expectedprofit: float
    clv12m: float
    palive: float
    uplift: float
    baseline: float
    segment: str


@lru_cache
def artefacts() -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bgnbd = pmclv.BetaGeoModel.load(str(config.modelsdir / "bgnbd.nc"))
        gg = pmclv.GammaGammaModel.load(str(config.modelsdir / "gamma_gamma.nc"))
        # load() restores the posterior; the PyMC graph is needed for predictions
        for m in (bgnbd, gg):
            if not hasattr(m, "model"):
                m.build_model()
    return {
        "bgnbd": bgnbd,
        "gg": gg,
        "uplift": joblib.load(config.modelsdir / "uplift.joblib"),
        "thresholds": json.loads(simulate.thresholdspath.read_text()),
        "decisions": pd.read_parquet(simulate.decisionspath).set_index("customer_id"),
    }


def score(h: purchasehistory, features: list[float], margin: float, cost: float) -> decision:
    a = artefacts()
    rfm = pd.DataFrame(
        {
            "customer_id": [0],
            "frequency": [h.frequency],
            "recency": [h.recency],
            "T": [h.age],
            "monetary_value": [h.spend],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clv = float(clvmodel.point(clvmodel.expected_clv(a["gg"], a["bgnbd"], rfm))[0])
        palive = float(
            clvmodel.point(a["bgnbd"].expected_probability_alive(data=rfm[clvmodel.bgnbdcols]))[0]
        )
    clv = min(clv, a["thresholds"]["clv_cap"])

    x = np.asarray([features], np.float32)
    bundle = a["uplift"]
    uplift = float(learners.predict_uplift(bundle["name"], bundle["model"], x)[0])
    baseline = float(learners.predict_baseline(bundle["tlearner"], x)[0])
    profit = float(engine.expected_profit(clv, uplift, margin, cost))

    return decision(
        action="send incentive" if profit > 0 else "do not contact",
        expectedprofit=round(profit, 2),
        clv12m=round(clv, 2),
        palive=round(palive, 4),
        uplift=round(uplift, 5),
        baseline=round(baseline, 5),
        segment=engine.segment(uplift, baseline, a["thresholds"]),
    )


@app.get("/health")
def health() -> dict:
    a = artefacts()
    return {"status": "ok", "uplift_model": a["uplift"]["name"], "customers": len(a["decisions"])}


@app.post("/decide-action", response_model=decision)
def decide_action(req: decisionrequest) -> decision:
    return score(req.history, req.features, req.margin, req.cost)


@app.get("/customers/{customerid}")
def customer(customerid: int) -> dict:
    """Decision for a real Online Retail II customer (uplift from the simulated pairing)."""
    table = artefacts()["decisions"]
    if customerid not in table.index:
        raise HTTPException(status_code=404, detail="customer not found")
    row = table.loc[customerid]
    return {"customer_id": customerid} | {
        k: (round(float(v), 4) if isinstance(v, int | float | np.floating) else v)
        for k, v in row.items()
    }
