"""Clean Online Retail II transactions into one row per customer invoice."""

from __future__ import annotations

import pandas as pd

# Stock codes that are fees, postage or manual adjustments — not product purchases.
nonproductcodes = (
    r"POST|DOT|M|C2|D|S|B|CRUK|PADS|BANK CHARGES|AMAZONFEE|ADJUST\d*|TEST\d+|gift_\d+_\d+"
)


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep identifiable, positive-value product purchases.

    Returns the cleaned line items and a log with the rows remaining after each step,
    so every exclusion is visible in the case study.
    """
    log: list[tuple[str, int]] = [("raw line items", len(df))]

    out = df.dropna(subset=["customer_id"])
    log.append(("with a customer id", len(out)))

    out = out[~out["invoice"].str.startswith("C", na=False)]
    log.append(("not a cancellation (invoice 'C…')", len(out)))

    out = out[(out["quantity"] > 0) & (out["price"] > 0)]
    log.append(("positive quantity and price", len(out)))

    isfee = out["stock_code"].str.fullmatch(nonproductcodes, case=False, na=False)
    out = out[~isfee]
    log.append(("product lines only (no postage/fees)", len(out)))

    out = out.drop_duplicates()
    log.append(("exact duplicates removed", len(out)))

    out = out.assign(
        customer_id=out["customer_id"].astype("int64"),
        revenue=out["quantity"] * out["price"],
    ).reset_index(drop=True)

    return out, pd.DataFrame(log, columns=["step", "rows"])


def to_invoices(lines: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line items to one row per (customer, invoice) with its date and revenue."""
    return (
        lines.groupby(["customer_id", "invoice"], as_index=False)
        .agg(date=("invoice_date", "min"), revenue=("revenue", "sum"))
        .sort_values(["customer_id", "date"], ignore_index=True)
    )
