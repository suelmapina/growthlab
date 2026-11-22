"""Uplift pipeline: fit uplift learners on Criteo, compare them on a held-out test set.

uv run python -m growthlab.uplift.train                     # outcome = visit
uv run python -m growthlab.uplift.train --outcome conversion
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from growthlab import config, figures
from growthlab.uplift import learners, metrics, segments

reportdir = config.root / "reports" / "uplift"
scorespath = config.processeddir / "uplift_scores.parquet"


def load(outcome: str):
    df = pd.read_parquet(config.criteoparquet)
    x = df[learners.features].to_numpy(np.float32)
    y = df[outcome].to_numpy(np.int8)
    t = df["treatment"].to_numpy(np.int8)
    # stratify on treatment × outcome so both test arms keep enough positives
    strata = t * 2 + y
    return train_test_split(x, y, t, test_size=0.3, stratify=strata, random_state=config.seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare uplift models.")
    parser.add_argument("--outcome", choices=["visit", "conversion"], default="visit")
    parser.add_argument("--no-mlflow", dest="nomlflow", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    reportdir.mkdir(parents=True, exist_ok=True)
    xtrain, xtest, ytrain, ytest, ttrain, ttest = load(args.outcome)
    ate, atese = metrics.ate(ytest, ttest)
    print(f"train {len(ytrain):,} · test {len(ytest):,} · treated {ttest.mean():.1%}")
    print(f"test ATE on {args.outcome}: {ate * 100:+.3f} pp (± {1.96 * atese * 100:.3f})\n")

    rng = np.random.default_rng(config.seed)
    scores = {"random": rng.random(len(ytest))}
    timings = {}

    fitted = {}
    for name, fit in learners.fitters.items():
        start = time.time()
        fitted[name] = fit(xtrain, ytrain, ttrain)
        scores[name] = learners.predict_uplift(name, fitted[name], xtest)
        timings[name] = time.time() - start
        print(f"  fitted {name:<10} {timings[name]:5.0f}s")
    baseline = learners.predict_baseline(fitted["tlearner"], xtest)

    results = {}
    for name in figures.upliftlabels:
        s = scores[name]
        results[name] = {
            "qini_coefficient": metrics.qini_coefficient(ytest, ttest, s),
            "uplift_top10": metrics.uplift_at(ytest, ttest, s, 0.10),
            "uplift_top30": metrics.uplift_at(ytest, ttest, s, 0.30),
            "fit_seconds": round(timings.get(name, 0.0), 1),
        }
    table = pd.DataFrame(results).T
    print("\n" + table.round(4).to_string())

    best = max(
        (n for n in results if n not in ("random", "response")),
        key=lambda n: results[n]["qini_coefficient"],
    )
    print(f"\nbest causal model: {best}")

    # is the causal model really better than plain response targeting?
    top10 = lambda y, t, s: metrics.uplift_at(y, t, s, 0.10)  # noqa: E731
    versus = {}
    for metricname, fn in (("qini_coefficient", metrics.qini_coefficient), ("uplift_top10", top10)):
        versus[metricname] = metrics.paired_bootstrap(
            ytest, ttest, scores[best], scores["response"], fn, seed=config.seed
        )
        v = versus[metricname]
        print(
            f"  {best} − response, {metricname}: {v['diff']:+.4f} "
            f"[95% CI {v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"
        )

    seg = segments.assign_segments(scores[best], baseline)
    segtable = segments.segment_report(seg, ytest, ttest)
    print("\n" + segtable.round(4).to_string())

    # artefacts
    curves = {n: metrics.qini_curve(ytest, ttest, scores[n]) for n in figures.upliftlabels}
    figures.qini(curves, reportdir / "qini.png")
    table.round(5).to_csv(reportdir / "models.csv")
    segtable.round(5).to_csv(reportdir / "segments.csv")
    summary = {
        "outcome": args.outcome,
        "train_rows": int(len(ytrain)),
        "test_rows": int(len(ytest)),
        "ate": ate,
        "ate_se": atese,
        "best_model": best,
        "models": results,
        "best_vs_response_bootstrap": versus,
    }
    (reportdir / "metrics.json").write_text(json.dumps(summary, indent=2))

    out = pd.DataFrame({"treatment": ttest, args.outcome: ytest, "segment": seg})
    for name in figures.upliftlabels:
        if name != "random":
            out[f"score_{name}"] = scores[name]
    out["baseline"] = baseline
    out.to_parquet(scorespath, index=False)

    # the serving bundle: best causal model + control model for the baseline outcome
    config.modelsdir.mkdir(exist_ok=True)
    joblib.dump(
        {
            "name": best,
            "model": fitted[best],
            "tlearner": fitted["tlearner"],
            "features": learners.features,
            "outcome": args.outcome,
            "ate": ate,
        },
        config.modelsdir / "uplift.joblib",
    )

    if not args.nomlflow:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{config.root / 'mlflow.db'}")
        mlflow.set_experiment("growthlab-uplift")
        with mlflow.start_run(run_name=f"uplift-{args.outcome}"):
            mlflow.log_params({"outcome": args.outcome, "best_model": best} | learners.treeparams)
            mlflow.log_metrics(
                {f"{n}.{k}": v for n, r in results.items() for k, v in r.items()} | {"ate": ate}
            )
            mlflow.log_artifacts(str(reportdir), artifact_path="report")

    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
