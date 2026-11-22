import numpy as np
import pandas as pd
import pytest

from growthlab import config
from growthlab.decision import engine, simulate


def test_expected_profit_formula():
    assert engine.expected_profit(1000, 0.02, margin=0.3, cost=5) == pytest.approx(1.0)


def test_decide_contacts_only_profitable_customers():
    clv = np.array([1000, 1000, 100])
    uplift = np.array([0.05, -0.01, 0.05])
    assert engine.decide(clv, uplift, margin=0.3, cost=5).tolist() == [True, False, False]


def test_segment_rule():
    th = {"persuadable_cutoff": 0.02, "baseline_median": 0.01}
    assert engine.segment(-0.001, 0.5, th) == "Sleeping Dogs"
    assert engine.segment(0.03, 0.0, th) == "Persuadables"
    assert engine.segment(0.01, 0.02, th) == "Sure Things"
    assert engine.segment(0.01, 0.001, th) == "Lost Causes"


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    customers = pd.DataFrame(
        {
            "customer_id": range(200),
            "clv": rng.gamma(2, 500, 200),
            "activity": rng.random(200),
            "p_alive": 1.0,
        }
    )
    pool = pd.DataFrame(
        {
            "uplift": rng.normal(0.01, 0.01, 2000),
            "response": rng.random(2000),
            "baseline": rng.random(2000),
            "y": rng.integers(0, 2, 2000),
            "t": rng.integers(0, 2, 2000),
        }
    )
    return customers, pool


def test_aligned_pairing_matches_ranks(toy):
    customers, pool = toy
    df = simulate.pair(customers, pool, "aligned", np.random.default_rng(1))
    assert df["activity"].rank().corr(df["baseline"].rank()) == pytest.approx(1.0)


def test_simulation_reports_every_policy(toy):
    customers, pool = toy
    sim = simulate.simulate(customers, pool, "independent", 0.3, 5, reps=3, seed=0)
    table = simulate.summarise(sim)
    assert list(table.index) == list(simulate.policies.values())
    assert table.loc["Contact nobody", "profit_mean"] == 0


needsmodels = pytest.mark.skipif(
    not (config.modelsdir / "uplift.joblib").exists() or not simulate.decisionspath.exists(),
    reason="run `make clv uplift decision` first",
)


@needsmodels
def test_api_end_to_end():
    from fastapi.testclient import TestClient

    from growthlab.api.main import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"

    body = {
        "history": {"frequency": 6, "recency": 60, "age": 80, "spend": 350},
        "features": [12.6, 10.1, 8.2, 4.7, 10.3, 4.1, -4.5, 4.8, 3.9, 13.2, 5.3, -0.2],
    }
    res = client.post("/decide-action", json=body)
    assert res.status_code == 200
    out = res.json()
    assert out["action"] in {"send incentive", "do not contact"}
    assert out["clv12m"] > 0

    bad = body | {"history": {"frequency": 2, "recency": 90, "age": 80, "spend": 100}}
    assert client.post("/decide-action", json=bad).status_code == 422

    assert client.get("/customers/-1").status_code == 404
