"""GrowthLab interactive demo.

uv run streamlit run src/growthlab/demo.py
"""

from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from growthlab import config, theme
from growthlab.decision import engine, simulate

repourl = "https://github.com/suelmapina/growthlab"
costgrid = [0.5, 1, 2, 5, 10, 20, 50]

display = {  # policy names as shown to a reader
    "Contact nobody": "Contact nobody",
    "Contact everyone": "Contact everyone",
    "Top 20% by CLV": "Top 20% by customer value",
    "Top 20% by response model": "Top 20% most likely to respond",
    "Top 20% by predicted uplift": "Top 20% by predicted uplift",
    "Top 20% by expected profit": "Top 20% by expected profit",
    "GrowthLab: profit > 0": "GrowthLab: profit rule",
}
ours = "GrowthLab: profit rule"

st.set_page_config(
    page_title="GrowthLab · Retention targeting", page_icon=":material/insights:", layout="wide"
)
st.markdown(theme.css, unsafe_allow_html=True)
alt.theme.register("portfolio", enable=True)(theme.altair_theme)


# --- helpers -------------------------------------------------------------------------------
def html(text: str) -> None:
    st.markdown(text, unsafe_allow_html=True)


def money(v: float, decimals: int = 0) -> str:
    sign = "−" if v < 0 else ""
    v = abs(v)
    if v >= 1000 and decimals == 0:
        return f"{sign}€{v / 1000:,.1f}k"
    return f"{sign}€{v:,.{decimals}f}"


def kpi(label: str, value: str, note: str, accent: bool = False, highlight: bool = False) -> str:
    return (
        f'<div class="card{" highlight" if highlight else ""}">'
        f'<div class="kpilabel">{label}</div>'
        f'<div class="kpivalue{" accent" if accent else ""}">{value}</div>'
        f'<div class="kpinote">{note}</div></div>'
    )


def section(title: str, note: str) -> None:
    html(f'<div class="sectiontitle">{title}</div><div class="sectionnote">{note}</div>')


eurolabels = (
    "datum.value == 0 ? '€0' : (datum.value < 0 ? '−€' : '€') + "
    "format(abs(datum.value) / 1000, ',.0f') + 'k'"
)


# --- data ----------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def population():
    customers, pool, cap = simulate.load()
    decisions = pd.read_parquet(config.datafile("decisions.parquet"))
    return customers, pool, cap, decisions


@st.cache_data(show_spinner=False)
def evidence() -> dict:
    reports = config.root / "reports"
    return {
        name: json.loads((reports / name / "metrics.json").read_text())
        for name in ("clv", "uplift", "decision")
    }


@st.cache_data(show_spinner=False)
def policy_table(margin: float, cost: float, mode: str, reps: int = 200) -> pd.DataFrame:
    customers, pool, _, _ = population()
    sim = simulate.simulate(customers, pool, mode, margin, cost, reps, config.seed)
    table = simulate.summarise(sim)
    table.index = [display[n] for n in table.index]
    return table


@st.cache_data(show_spinner=False)
def cost_curve(margin: float, mode: str, reps: int = 200) -> pd.DataFrame:
    customers, pool, _, _ = population()
    rows = []
    for cost in costgrid:
        table = policy_table(margin, cost, mode, reps)
        for name in (
            ours,
            "Top 20% by customer value",
            "Top 20% by predicted uplift",
            "Contact everyone",
        ):
            rows.append({"cost": cost, "policy": name, "profit": table.loc[name, "profit_mean"]})
    return pd.DataFrame(rows)


customers, pool, cap, decisions = population()
proof = evidence()

# --- header --------------------------------------------------------------------------------
html(
    '<div class="eyebrow">GrowthLab · causal machine learning case study</div>'
    '<div class="headline">Who should get the retention offer?</div>'
    '<div class="lede">Campaigns usually target the most valuable customers, or the ones most '
    "likely to buy. Both waste budget on people who would have stayed anyway. GrowthLab sends "
    "an offer only when it is expected to <i>change</i> a customer's behaviour by enough to pay "
    "for itself.</div>"
    '<div class="formula">expected profit = margin × <b>customer value</b> × '
    "<b>causal uplift</b> − cost per contact</div>"
)
st.write("")

with st.container(border=True):
    c1, c2, c3 = st.columns([1, 1, 1.2], gap="large")
    margin = (
        c1.slider("Gross margin", 5, 80, round(engine.defaultmargin * 100), 5, format="%d%%") / 100
    )
    cost = c2.select_slider("Cost per contacted customer (€)", costgrid, value=engine.defaultcost)
    modelabel = c3.segmented_control(
        "Value vs. persuadability",
        ["Independent", "Aligned"],
        default="Independent",
        help=(
            "The two public datasets share no customers, so they are paired in simulation. "
            "Independent: a customer's value says nothing about how persuadable they are. "
            "Aligned: the most active buyers are also the most responsive. "
            "The truth lies somewhere between the two."
        ),
    )
mode = (modelabel or "Independent").lower()
cost = float(cost)

table = policy_table(margin, cost, mode)
growth = table.loc[ours]
byvalue = table.loc["Top 20% by customer value"]
spend = cost * growth["contacted"]

st.write("")
k1, k2, k3, k4 = st.columns(4)
k1.markdown(
    kpi(
        "Incremental profit",
        money(growth["profit_mean"]),
        f"± {money(1.96 * growth['profit_se'])} (95% interval)",
        accent=True,
        highlight=True,
    ),
    unsafe_allow_html=True,
)
k2.markdown(
    kpi(
        "Customers contacted",
        f"{growth['contacted']:,.0f}",
        f"{growth['contacted'] / len(customers):.0%} of {len(customers):,} customers",
    ),
    unsafe_allow_html=True,
)
k3.markdown(
    kpi(
        "vs. targeting by value",
        money(-byvalue["vs_growthlab"]).replace("€", "+€", 1)
        if byvalue["vs_growthlab"] < 0
        else money(-byvalue["vs_growthlab"]),
        "paired, same customers",
    ),
    unsafe_allow_html=True,
)
k4.markdown(
    kpi(
        "Return per € spent",
        f"{growth['profit_mean'] / spend:,.1f}×" if spend else "n/a",
        f"on {money(spend)} of campaign cost",
    ),
    unsafe_allow_html=True,
)
st.write("")

policies, customerstab, method = st.tabs(["Policy comparison", "Customer decisions", "Method"])

# --- policies ------------------------------------------------------------------------------
with policies:
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        section(
            "Which targeting policy earns the most?",
            "Expected incremental profit, scored on observed outcomes of a randomised experiment.",
        )
        bars = table.drop(index="Contact nobody").reset_index(names="policy")
        bars["low"] = bars["profit_mean"] - 1.96 * bars["profit_se"]
        bars["high"] = bars["profit_mean"] + 1.96 * bars["profit_se"]
        bars["ours"] = bars["policy"] == ours
        bars["text"] = [money(v) for v in bars["profit_mean"]]
        bars["textx"] = bars["high"].clip(lower=0)
        order = bars.sort_values("profit_mean", ascending=False)["policy"].tolist()

        y = alt.Y("policy:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelPadding=8))
        base = alt.Chart(bars).encode(y=y)
        chart = (
            base.mark_bar(height=20, cornerRadiusEnd=4).encode(
                x=alt.X(
                    "profit_mean:Q", title=None, axis=alt.Axis(labelExpr=eurolabels, tickCount=5)
                ),
                color=alt.condition(
                    "datum.ours", alt.value(theme.primary), alt.value(theme.context)
                ),
                tooltip=[
                    alt.Tooltip("policy:N", title="Policy"),
                    alt.Tooltip("contacted:Q", format=",.0f", title="Contacted"),
                    alt.Tooltip("profit_mean:Q", format=",.0f", title="Profit (€)"),
                ],
            )
            + base.mark_rule(color=theme.mutedforeground, strokeWidth=1, opacity=0.55).encode(
                x="low:Q", x2="high:Q"
            )
            + base.mark_text(align="left", dx=8, fontSize=12).encode(
                x="textx:Q",
                text="text:N",
                color=alt.condition(
                    "datum.ours", alt.value(theme.foreground), alt.value(theme.mutedforeground)
                ),
            )
        ).properties(height=270)
        st.altair_chart(chart, width="stretch")

    with right:
        section(
            "How profit responds to the cost of contact",
            "The profit rule contacts fewer people as contact gets expensive.",
        )
        short = {
            ours: "GrowthLab",
            "Top 20% by predicted uplift": "Top 20% by uplift",
            "Top 20% by customer value": "Top 20% by value",
            "Contact everyone": "Everyone",
        }
        with st.spinner("Simulating 1,400 campaigns across contact costs…"):
            curve = cost_curve(margin, mode)
        curve = curve.assign(policy=lambda x: x["policy"].map(short))
        domain = list(short.values())
        colors = [theme.primary, theme.secondary, theme.light, theme.mutedforeground]
        floor = max(-60_000, curve["profit"].min())
        lines = (
            alt.Chart(curve)
            .mark_line(point=alt.OverlayMarkDef(size=28), strokeWidth=2)
            .encode(
                x=alt.X(
                    "cost:Q",
                    scale=alt.Scale(type="log"),
                    title="Cost per contact (€)",
                    axis=alt.Axis(values=costgrid, format="~g"),
                ),
                y=alt.Y(
                    "profit:Q",
                    title=None,
                    scale=alt.Scale(domainMin=floor, clamp=True),
                    axis=alt.Axis(labelExpr=eurolabels, tickCount=5),
                ),
                color=alt.Color(
                    "policy:N",
                    scale=alt.Scale(domain=domain, range=colors),
                    legend=alt.Legend(
                        orient="bottom",
                        columns=4,
                        title=None,
                        labelLimit=160,
                        symbolType="stroke",
                        symbolStrokeWidth=3,
                    ),
                ),
                strokeDash=alt.condition(
                    alt.datum.policy == "Everyone", alt.value([4, 3]), alt.value([1, 0])
                ),
                tooltip=["policy:N", "cost:Q", alt.Tooltip("profit:Q", format=",.0f")],
            )
        )
        marker = (
            alt.Chart(pd.DataFrame({"cost": [cost]}))
            .mark_rule(color=theme.mutedforeground, strokeDash=[2, 3])
            .encode(x="cost:Q")
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=theme.border).encode(y="y:Q")
        st.altair_chart((zero + lines + marker).properties(height=250), width="stretch")

    st.caption(
        "Each policy is scored on what paired users of a randomised experiment actually did, "
        "not on the model's own predictions; 200 simulated pairings per setting. Bars show the "
        "95% interval of the expected profit."
    )

# --- customers -----------------------------------------------------------------------------
with customerstab:
    d = decisions.copy()
    d["expected_profit"] = engine.expected_profit(d["clv"], d["uplift"], margin, cost)
    d["send"] = d["expected_profit"] > 0

    section(
        "Segments",
        "Persuadable customers are worth contacting; the rest mostly are not.",
    )
    cols = st.columns(4)
    notes = {
        "Persuadables": "respond because of the offer",
        "Sure Things": "would have stayed anyway",
        "Lost Causes": "will not respond either way",
        "Sleeping Dogs": "the offer does nothing, or harm",
    }
    for col, name in zip(
        cols, ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"], strict=True
    ):
        part = d[d["segment"] == name]
        col.markdown(
            kpi(
                name,
                f"{len(part) / len(d):.0%}",
                f"{notes[name]} · {part['send'].mean():.0%} contacted",
                highlight=name == "Persuadables",
            ),
            unsafe_allow_html=True,
        )
    st.write("")

    left, right = st.columns([1.45, 1], gap="large")
    ranked = d.sort_values("expected_profit", ascending=False).reset_index(drop=True)
    with left:
        section("Highest expected profit", "Select a row to see how the decision is made.")
        view = ranked.head(300).assign(
            uplift=lambda x: x["uplift"] * 100,
            action=lambda x: np.where(x["send"], "Send offer", "Do not contact"),
        )
        picked = st.dataframe(
            view[["customer_id", "segment", "clv", "uplift", "expected_profit", "action"]],
            hide_index=True,
            width="stretch",
            height=380,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "customer_id": st.column_config.NumberColumn("Customer", format="%d"),
                "segment": "Segment",
                "clv": st.column_config.NumberColumn("Value, 12 m", format="€%.0f"),
                "uplift": st.column_config.NumberColumn("Uplift", format="%+.2f pp"),
                "expected_profit": st.column_config.NumberColumn("Exp. profit", format="€%.2f"),
                "action": "Decision",
            },
        )
    rows = picked.selection.rows if picked else []
    row = view.iloc[rows[0]] if rows else view.iloc[0]

    with right:
        section(
            f"Customer {int(row['customer_id'])}",
            f"{row['segment']} · "
            f"{decisions.set_index('customer_id').loc[row['customer_id'], 'p_alive']:.0%} "
            "probability still active",
        )
        value = margin * row["clv"] * row["uplift"] / 100
        badge = (
            '<span class="badge go">SEND OFFER</span>'
            if row["send"]
            else '<span class="badge stop">DO NOT CONTACT</span>'
        )
        steps = [
            ("", "customer value, 12 months", money(row["clv"], 2)),
            ("×", "causal uplift", f"{row['uplift']:+.2f} pp"),
            ("×", "gross margin", f"{margin:.0%}"),
            ("=", "incremental value", money(value, 2)),
            ("−", "cost of contact", money(cost, 2)),
        ]
        body = "".join(
            f"<tr><td>{op}</td><td>{name}</td><td>{v}</td></tr>" for op, name, v in steps
        )
        html(
            '<div class="card"><table class="steps">'
            f"{body}"
            f'<tr class="total"><td>=</td><td>expected profit</td>'
            f"<td>{money(row['expected_profit'], 2)}</td></tr>"
            f"</table><div style='margin-top:0.8rem'>{badge}</div></div>"
        )
        st.caption(
            "Customer value: BG/NBD + Gamma-Gamma on two years of transactions (capped at "
            f"{money(cap)}). Uplift: X-learner trained on a randomised advertising experiment, "
            "assigned to this customer by simulation."
        )

# --- method --------------------------------------------------------------------------------
with method:
    clvm, upm = proof["clv"], proof["uplift"]
    vs = upm["best_vs_response_bootstrap"]["uplift_top10"]
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.markdown(
        kpi(
            "1 · Customer value",
            f"{clvm['purchases']['bgnbd']['total_error_pct']:+.1%}",
            "error in total purchases over a 26-week holdout (BG/NBD + Gamma-Gamma, "
            f"{clvm['customers']:,} customers)",
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        kpi(
            "2 · Causal uplift",
            f"+{vs['diff'] * 100:.1f} pp",
            "more uplift in the top 10% than a response model "
            f"(95% CI +{vs['ci_low'] * 100:.1f} to +{vs['ci_high'] * 100:.1f})",
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        kpi(
            "3 · Honest evaluation",
            "Transformed outcome",
            "policies are scored on observed outcomes of a randomised experiment, "
            "never on the model's own predictions",
        ),
        unsafe_allow_html=True,
    )
    st.write("")
    f1, f2 = st.columns(2, gap="large")
    f1.image(str(config.root / "reports" / "clv" / "calibration.png"))
    f2.image(str(config.root / "reports" / "uplift" / "qini.png"))

    with st.expander("Assumptions and limitations"):
        st.markdown(
            "- **The two datasets share no customers.** Value comes from a UK online retailer, "
            "uplift from a randomised advertising experiment; they are paired in simulation, "
            "bracketed by an independent and an aligned scenario.\n"
            "- **A response to the offer is assumed to retain the customer's 12-month value.** "
            "In production, this would be measured with a randomised hold-out group.\n"
            "- **Seasonality is not modelled**, so revenue is under-forecast in the holiday "
            "quarter (−16% on the holdout).\n"
            "- **Margin and cost are illustrative**; change them above."
        )
    st.markdown(
        f"Code, full write-up and data sources: [{repourl.removeprefix('https://')}]({repourl}) · "
        "Data: Online Retail II (UCI, CC BY 4.0) and Criteo Uplift v2.1 (CC BY-NC-SA 4.0)."
    )
