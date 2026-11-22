"""CLV pipeline: clean → calibration/holdout evaluation → refit on full history → score.

uv run python -m growthlab.clv.train              # fast MAP fit
uv run python -m growthlab.clv.train --method mcmc  # full posterior + credible intervals
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from growthlab import config, figures
from growthlab.clv import evaluate, model, prepare

reportdir = config.root / "reports" / "clv"
scorespath = config.processeddir / "clv_scores.parquet"


def load_invoices() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(config.retailparquet)
    lines, log = prepare.clean_transactions(raw)
    invoices = prepare.to_invoices(lines)
    return invoices, log


def evaluate_holdout(invoices: pd.DataFrame, trainend: str, testend: str, method: str) -> dict:
    rfm = model.rfm_split(invoices, trainend, testend)
    weeks = (pd.Timestamp(testend) - pd.Timestamp(trainend)).days / 7

    # actual holdout outcomes
    inholdout = (invoices["date"] > trainend) & (invoices["date"] <= testend)
    holdoutrev = invoices[inholdout].groupby("customer_id")["revenue"].sum()
    actualrev = rfm["customer_id"].map(holdoutrev).fillna(0).to_numpy()
    actualn = rfm["test_frequency"].to_numpy()

    incal = invoices["date"] <= trainend
    calrev = invoices[incal].groupby("customer_id")["revenue"].agg(["sum", "mean"])
    caltotal = rfm["customer_id"].map(calrev["sum"]).to_numpy()
    calmean = rfm["customer_id"].map(calrev["mean"]).to_numpy()

    # models
    bgnbd = model.fit_bgnbd(rfm, method)
    gg = model.fit_gamma_gamma(rfm, method)
    predn = model.point(model.expected_purchases(bgnbd, rfm, weeks))
    predspend = model.point(model.expected_spend(gg, rfm))
    predrev = predn * predspend

    # naive baselines: historical purchase rate continues with no churn; past spend ranks future
    basen = rfm["frequency"].to_numpy() / rfm["T"].to_numpy() * weeks
    baserev = basen * calmean

    repeat = rfm["frequency"] > 0
    ggcorr = float(
        np.corrcoef(rfm.loc[repeat, "frequency"], rfm.loc[repeat, "monetary_value"])[0, 1]
    )

    calib = evaluate.calibration_table(rfm["frequency"], actualn, predn, basen)

    return {
        "rfm": rfm,
        "calibration": calib,
        "metrics": {
            "customers": int(len(rfm)),
            "repeat_customers": int(repeat.sum()),
            "holdout_weeks": round(weeks, 1),
            "frequency_spend_corr": round(ggcorr, 3),
            "purchases": {
                "bgnbd": evaluate.error_metrics(actualn, predn),
                "naive_rate": evaluate.error_metrics(actualn, basen),
            },
            "revenue": {
                "bgnbd_gg": evaluate.error_metrics(actualrev, predrev),
                "naive_rate": evaluate.error_metrics(actualrev, baserev),
            },
            "top20_revenue_captured": {
                "bgnbd_gg": evaluate.top_share_captured(actualrev, predrev),
                "past_spend": evaluate.top_share_captured(actualrev, caltotal),
                "oracle": evaluate.top_share_captured(actualrev, actualrev),
            },
        },
    }


def score_today(invoices: pd.DataFrame, method: str) -> pd.DataFrame:
    """Refit on the full history and score every customer's next-12-month value."""
    rfm = model.rfm_full(invoices)
    bgnbd = model.fit_bgnbd(rfm, method)
    gg = model.fit_gamma_gamma(rfm, method)

    scores = rfm[["customer_id", "frequency", "recency", "T", "monetary_value"]].copy()
    scores["p_alive"] = model.point(bgnbd.expected_probability_alive(data=rfm[model.bgnbdcols]))
    scores["expected_purchases_52w"] = model.point(model.expected_purchases(bgnbd, rfm, 52))
    scores["expected_spend"] = model.point(model.expected_spend(gg, rfm))
    clvda = model.expected_clv(gg, bgnbd, rfm, months=12)
    scores["clv_12m"] = model.point(clvda)
    if method == "mcmc":
        scores["clv_12m_p05"], scores["clv_12m_p95"] = model.interval(clvda, 0.9)

    config.modelsdir.mkdir(exist_ok=True)
    bgnbd.save(str(config.modelsdir / "bgnbd.nc"))
    gg.save(str(config.modelsdir / "gamma_gamma.nc"))
    return scores


def log_mlflow(args, metrics: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{config.root / 'mlflow.db'}")
    mlflow.set_experiment("growthlab-clv")
    with mlflow.start_run(run_name=f"bgnbd-gg-{args.method}"):
        mlflow.log_params(
            {"method": args.method, "train_end": args.trainend, "test_end": args.testend}
        )
        flat = {
            f"{group}.{name}.{k}": v
            for group in ("purchases", "revenue")
            for name, d in metrics[group].items()
            for k, v in d.items()
        } | {f"top20.{k}": v for k, v in metrics["top20_revenue_captured"].items()}
        mlflow.log_metrics(flat)
        mlflow.log_artifacts(str(reportdir), artifact_path="report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the CLV models.")
    parser.add_argument("--method", choices=["map", "mcmc"], default="map")
    parser.add_argument("--train-end", dest="trainend", default="2011-06-09")
    parser.add_argument("--test-end", dest="testend", default="2011-12-09")
    parser.add_argument("--no-mlflow", dest="nomlflow", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    reportdir.mkdir(parents=True, exist_ok=True)

    invoices, log = load_invoices()
    log.to_csv(reportdir / "cleaning_log.csv", index=False)
    print(log.to_string(index=False))
    print(
        f"\n{len(invoices):,} invoices, {invoices['customer_id'].nunique():,} customers, "
        f"{invoices['date'].min():%Y-%m-%d} → {invoices['date'].max():%Y-%m-%d}\n"
    )

    result = evaluate_holdout(invoices, args.trainend, args.testend, args.method)
    metrics = result["metrics"]
    result["calibration"].round(3).to_csv(reportdir / "calibration.csv")
    figures.calibration(result["calibration"], reportdir / "calibration.png")
    (reportdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(result["calibration"].round(2).to_string(), "\n")
    print(json.dumps(metrics, indent=2))

    scores = score_today(invoices, args.method)
    scores.to_parquet(scorespath, index=False)
    print(f"\n✓ scored {len(scores):,} customers → {scorespath.relative_to(config.root)}")
    print(
        scores[["p_alive", "expected_purchases_52w", "expected_spend", "clv_12m"]]
        .describe()
        .round(2)
        .to_string()
    )

    if not args.nomlflow:
        log_mlflow(args, metrics)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
