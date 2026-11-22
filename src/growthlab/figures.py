"""All README figures, in the portfolio style. Re-render without retraining:

uv run python -m growthlab.figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from growthlab import config, theme  # noqa: E402

reports = config.root / "reports"

upliftlabels = {
    "random": "Random",
    "response": "Response model (non-causal)",
    "slearner": "S-learner",
    "tlearner": "T-learner",
    "xlearner": "X-learner",
    "drlearner": "DR-learner",
}
upliftstyle = {  # (colour, linewidth, linestyle): one highlight, everything else quieter
    "random": (theme.mutedforeground, 1.0, (0, (4, 3))),
    "response": (theme.light, 1.6, "-"),
    "slearner": (theme.context, 1.4, "-"),
    "tlearner": (theme.context, 1.4, (0, (2, 1.5))),
    "xlearner": (theme.primary, 2.6, "-"),
    "drlearner": (theme.secondary, 1.6, "-"),
}

policylabels = {
    "everyone": "Contact everyone",
    "top20_clv": "Top 20% by CLV",
    "top20_response": "Top 20% by response model",
    "top20_uplift": "Top 20% by predicted uplift",
    "growthlab": "Profit rule (GrowthLab)",
}
policystyle = {
    "everyone": (theme.mutedforeground, 1.2, (0, (4, 3))),
    "top20_clv": (theme.light, 1.6, "-"),
    "top20_response": (theme.context, 1.6, "-"),
    "top20_uplift": (theme.secondary, 1.6, "-"),
    "growthlab": (theme.primary, 2.8, "-"),
}


def money(v: float) -> str:
    """€12k / −€3k (sign before the currency symbol)."""
    if abs(v) < 500:
        return "€0"
    return f"{'−' if v < 0 else ''}€{abs(v) / 1000:,.0f}k"


euros = mticker.FuncFormatter(lambda v, _: money(v))


def figure(title: str, note: str, size=(8, 4.6)):
    theme.use_matplotlib()
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title, pad=26)
    theme.subtitle(ax, note)
    return fig, ax


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibration(calib: pd.DataFrame, path) -> None:
    fig, ax = figure(
        "Purchase forecasts match what customers actually did",
        "Mean purchases in the 26-week holdout, by repeat purchases before it",
    )
    x = np.arange(len(calib))
    w = 0.27
    ax.bar(x - w, calib["actual"], w, label="Actual", color=theme.secondary)
    ax.bar(x, calib["predicted"], w, label="BG/NBD forecast", color=theme.primary)
    ax.bar(x + w, calib["baseline"], w, label="Naive rate (no churn)", color=theme.context)
    ax.set_xticks(x, calib.index)
    ax.set_xlabel("Repeat purchases in the calibration period")
    ax.set_ylabel("Purchases in holdout")
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left")
    save(fig, path)


def qini(curves: dict, path) -> None:
    fig, ax = figure(
        "Causal models find the persuadable customers first",
        "Incremental visits captured as more customers are targeted · held-out test set",
    )
    for name in ("random", "tlearner", "slearner", "response", "drlearner", "xlearner"):
        frac, q = curves[name]
        color, lw, ls = upliftstyle[name]
        ax.plot(frac * 100, q, color=color, lw=lw, ls=ls, label=upliftlabels[name])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Customers targeted, ranked by model")
    ax.set_ylabel("Incremental visits")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(upliftlabels[n]) for n in upliftlabels if upliftlabels[n] in labels]
    ax.legend(
        [handles[i] for i in reversed(order)],
        [labels[i] for i in reversed(order)],
        loc="lower right",
    )
    save(fig, path)


def sensitivity(sens: pd.DataFrame, path) -> None:
    fig, ax = figure(
        "Only the profit rule stays profitable as contact costs rise",
        "Expected incremental profit of each targeting policy · independent pairing",
    )
    floor = -60_000
    for name in ("everyone", "top20_response", "top20_clv", "top20_uplift", "growthlab"):
        part = sens[sens.policy == name]
        color, lw, ls = policystyle[name]
        ax.plot(
            part["cost"],
            part["profit_mean"],
            color=color,
            lw=lw,
            ls=ls,
            marker="o",
            ms=3.5,
            label=policylabels[name],
        )
    ax.axhline(0, color=theme.mutedforeground, lw=0.9)
    worst = sens[sens.policy == "everyone"].iloc[-1]
    if worst["profit_mean"] < floor:
        ax.set_ylim(bottom=floor)
        ax.annotate(
            f"contact everyone ↓ {money(worst['profit_mean'])} at €{worst['cost']:g}",
            xy=(0.99, 0.02),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
            color=theme.mutedforeground,
        )
    costs = sens["cost"].unique()
    ax.set_xscale("log")
    ax.set_xticks(costs, [f"€{c:g}" for c in costs])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(euros)
    ax.set_xlabel("Cost per contacted customer")
    ax.set_ylabel("Expected incremental profit")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower left")
    save(fig, path)


def qini_curves_from_scores(scores: pd.DataFrame) -> dict:
    from growthlab.uplift import metrics

    outcome = "visit" if "visit" in scores.columns else "conversion"
    y, t = scores[outcome].to_numpy(), scores["treatment"].to_numpy()
    rng = np.random.default_rng(config.seed)
    curves = {"random": metrics.qini_curve(y, t, rng.random(len(y)))}
    for name in upliftlabels:
        if name != "random":
            curves[name] = metrics.qini_curve(y, t, scores[f"score_{name}"].to_numpy())
    return curves


def main() -> None:
    calib = pd.read_csv(reports / "clv" / "calibration.csv", index_col=0)
    calibration(calib, reports / "clv" / "calibration.png")

    scores = pd.read_parquet(config.processeddir / "uplift_scores.parquet")
    qini(qini_curves_from_scores(scores), reports / "uplift" / "qini.png")

    sens = pd.read_csv(reports / "decision" / "sensitivity.csv")
    sensitivity(sens, reports / "decision" / "sensitivity.png")
    print("✓ figures written to reports/")


if __name__ == "__main__":
    main()
