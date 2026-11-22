"""Decision engine: compare targeting policies on a simulated population.

No public dataset has both purchase histories and a randomised retention experiment, so each
Online Retail customer (with their real CLV) is paired with a Criteo test-set user (with a
predicted uplift *and* a real randomised outcome). Two pairings bracket the unknown truth:

- independent: random pairing — value and persuadability are unrelated
- aligned: the most active buyers are paired with the users most likely to respond anyway

Policies are scored with the *transformed outcome* z = y · (t/p − (1−t)/(1−p)), whose
expectation is the true individual uplift in a randomised experiment. A policy's value is
Σ_targeted (margin · CLV · z) − cost · #targeted, averaged over many random pairings.
Using observed outcomes (not the model's own predictions) keeps the evaluation honest.

    uv run python -m growthlab.decision.simulate --margin 0.3 --cost 5
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from growthlab import config, figures
from growthlab.decision import engine

reportdir = config.root / "reports" / "decision"
decisionspath = config.processeddir / "decisions.parquet"
thresholdspath = config.modelsdir / "decision_thresholds.json"

clvcapquantile = 0.995  # a handful of wholesale accounts would otherwise dominate every sum

policies = {
    "nobody": "Contact nobody",
    "everyone": "Contact everyone",
    "top20_clv": "Top 20% by CLV",
    "top20_response": "Top 20% by response model",
    "top20_uplift": "Top 20% by predicted uplift",
    "top20_profit": "Top 20% by expected profit",
    "growthlab": "GrowthLab: profit > 0",
}


def load(model: str = "xlearner"):
    clv = pd.read_parquet(config.processeddir / "clv_scores.parquet")
    up = pd.read_parquet(config.processeddir / "uplift_scores.parquet")
    outcome = "visit" if "visit" in up.columns else "conversion"
    cap = float(clv["clv_12m"].quantile(clvcapquantile))
    customers = pd.DataFrame(
        {
            "customer_id": clv["customer_id"],
            "clv": clv["clv_12m"].clip(upper=cap),
            "activity": clv["expected_purchases_52w"],
            "p_alive": clv["p_alive"],
        }
    )
    pool = pd.DataFrame(
        {
            "uplift": up[f"score_{model}"],
            "response": up["score_response"],
            "baseline": up["baseline"],
            "y": up[outcome],
            "t": up["treatment"],
        }
    )
    return customers, pool, cap


def pair(customers: pd.DataFrame, pool: pd.DataFrame, mode: str, rng) -> pd.DataFrame:
    """Draw one Criteo user per customer, randomly or rank-aligned on activity ↔ baseline."""
    draw = pool.iloc[rng.choice(len(pool), len(customers), replace=False)].reset_index(drop=True)
    if mode == "aligned":
        byactivity = np.argsort(customers["activity"].to_numpy(), kind="stable")
        bybaseline = np.argsort(draw["baseline"].to_numpy(), kind="stable")
        aligned = np.empty(len(customers), dtype=int)
        aligned[byactivity] = bybaseline
        draw = draw.iloc[aligned].reset_index(drop=True)
    return pd.concat([customers.reset_index(drop=True), draw], axis=1)


def top(score: np.ndarray, share: float = 0.2) -> np.ndarray:
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-score, kind="stable")[: math.ceil(len(score) * share)]] = True
    return mask


def targets(clv, uplift, response, margin: float, cost: float) -> dict[str, np.ndarray]:
    """Boolean contact mask for every policy."""
    profit = engine.expected_profit(clv, uplift, margin, cost)
    n = len(clv)
    return {
        "nobody": np.zeros(n, dtype=bool),
        "everyone": np.ones(n, dtype=bool),
        "top20_clv": top(clv),
        "top20_response": top(response),
        "top20_uplift": top(uplift),
        "top20_profit": top(profit),
        "growthlab": profit > 0,
    }


def simulate(customers, pool, mode: str, margin: float, cost: float, reps: int, seed: int):
    """Score every policy on `reps` random pairings (NumPy version of `pair` + `targets`)."""
    rng = np.random.default_rng(seed)
    p = float(pool["t"].mean())
    y, t = pool["y"].to_numpy(float), pool["t"].to_numpy(float)
    z = y * (t / p - (1 - t) / (1 - p))  # transformed outcome: E[z | x] = true uplift
    uplift, response = pool["uplift"].to_numpy(), pool["response"].to_numpy()
    baseline = pool["baseline"].to_numpy()
    clv = customers["clv"].to_numpy()
    n = len(customers)
    byactivity = np.argsort(customers["activity"].to_numpy(), kind="stable")

    rows = []
    for rep in range(reps):
        idx = rng.choice(len(pool), n, replace=False)
        if mode == "aligned":
            aligned = np.empty(n, dtype=int)
            aligned[byactivity] = np.argsort(baseline[idx], kind="stable")
            idx = idx[aligned]
        value = margin * clv * z[idx]
        for name, mask in targets(clv, uplift[idx], response[idx], margin, cost).items():
            rows.append(
                {
                    "rep": rep,
                    "policy": name,
                    "contacted": int(mask.sum()),
                    "profit": float(value[mask].sum() - cost * mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def summarise(sim: pd.DataFrame) -> pd.DataFrame:
    """Mean profit with its standard error, and the paired difference vs GrowthLab.

    The spread across pairings is mostly noise in the transformed outcome, so what we report is
    the precision of the *expected* profit (standard error), not a range of campaign outcomes.
    Differences are paired: every policy is scored on the same simulated customers.
    """
    profit = sim.pivot(index="rep", columns="policy", values="profit")[list(policies)]
    contacted = sim.pivot(index="rep", columns="policy", values="contacted")[list(policies)]
    reps = len(profit)
    diff = profit.sub(profit["growthlab"], axis=0)
    table = pd.DataFrame(
        {
            "contacted": contacted.mean().round(0),
            "profit_mean": profit.mean(),
            "profit_se": profit.std(ddof=1) / np.sqrt(reps),
            "vs_growthlab": diff.mean(),
            "vs_growthlab_se": diff.std(ddof=1) / np.sqrt(reps),
        }
    )
    table.index = [policies[n] for n in table.index]
    return table


def save_decisions(customers, pool, margin, cost, seed) -> dict:
    """One row per real customer for the API/demo (uplift comes from one independent pairing)."""
    rng = np.random.default_rng(seed)
    df = pair(customers, pool, "independent", rng)
    df["expected_profit"] = engine.expected_profit(df["clv"], df["uplift"], margin, cost)
    df["action"] = np.where(df["expected_profit"] > 0, "send incentive", "do not contact")

    nonneg = pool.loc[pool["uplift"] >= 0, "uplift"]
    thresholds = {
        "persuadable_cutoff": float(nonneg.quantile(0.75)),
        "baseline_median": float(pool.loc[pool["uplift"] >= 0, "baseline"].median()),
    }
    df["segment"] = [
        engine.segment(u, b, thresholds) for u, b in zip(df["uplift"], df["baseline"], strict=True)
    ]
    cols = [
        "customer_id",
        "clv",
        "p_alive",
        "uplift",
        "baseline",
        "expected_profit",
        "action",
        "segment",
    ]
    df[cols].to_parquet(decisionspath, index=False)
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate targeting policies.")
    parser.add_argument("--margin", type=float, default=engine.defaultmargin)
    parser.add_argument("--cost", type=float, default=engine.defaultcost)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()

    t0 = time.time()
    reportdir.mkdir(parents=True, exist_ok=True)
    customers, pool, cap = load()
    print(
        f"{len(customers):,} customers · {len(pool):,} Criteo test users · CLV capped at "
        f"€{cap:,.0f} (p{clvcapquantile * 100:g})"
    )
    print(f"margin {args.margin:.0%} · cost €{args.cost:g} · {args.reps} pairings\n")

    summary = {"margin": args.margin, "cost": args.cost, "reps": args.reps, "clv_cap": cap}
    for mode in ("independent", "aligned"):
        sim = simulate(customers, pool, mode, args.margin, args.cost, args.reps, config.seed)
        table = summarise(sim)
        table.round(2).to_csv(reportdir / f"policies_{mode}.csv")
        summary[mode] = table.round(2).to_dict(orient="index")
        print(f"— {mode} pairing —")
        print(table.round(0).to_string(), "\n")

    costs = [0.5, 1, 2, 5, 10, 20, 50]
    sens = []
    for cost in costs:
        sim = simulate(customers, pool, "independent", args.margin, cost, 50, config.seed)
        for name, part in sim.groupby("policy"):
            sens.append({"cost": cost, "policy": name, "profit_mean": part["profit"].mean()})
    sens = pd.DataFrame(sens)
    sens.round(2).to_csv(reportdir / "sensitivity.csv", index=False)
    figures.sensitivity(sens, reportdir / "sensitivity.png")

    thresholds = save_decisions(customers, pool, args.margin, args.cost, config.seed)
    thresholdspath.write_text(json.dumps(thresholds | {"clv_cap": cap}, indent=2))
    (reportdir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"✓ decisions → {decisionspath.relative_to(config.root)}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
