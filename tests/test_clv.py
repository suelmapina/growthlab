import numpy as np
import pandas as pd
import pytest

from growthlab.clv import evaluate, prepare


@pytest.fixture
def lines():
    ts = pd.Timestamp("2011-01-01")
    return pd.DataFrame(
        {
            "invoice": ["1", "1", "C2", "3", "4", "5", "5"],
            "stock_code": ["A", "POST", "A", "A", "B", "A", "A"],
            "description": ["x"] * 7,
            "quantity": [2, 1, -2, 1, 0, 3, 3],
            "invoice_date": [ts] * 7,
            "price": [5.0, 10.0, 5.0, 4.0, 1.0, 2.0, 2.0],
            "customer_id": pd.array([1, 1, 1, None, 2, 2, 2], dtype="Int64"),
            "country": ["UK"] * 7,
        }
    ).astype({"invoice": "string", "stock_code": "string"})


def test_clean_transactions_drops_non_purchases(lines):
    clean, log = prepare.clean_transactions(lines)
    # kept: invoice 1 product line, and one copy of the duplicated invoice-5 line
    assert sorted(clean["invoice"]) == ["1", "5"]
    assert clean["revenue"].tolist() == [10.0, 6.0]
    assert log["rows"].iloc[0] == len(lines)
    assert log["rows"].is_monotonic_decreasing


def test_to_invoices_aggregates_lines(lines):
    clean, _ = prepare.clean_transactions(lines)
    inv = prepare.to_invoices(clean)
    assert list(inv.columns) == ["customer_id", "invoice", "date", "revenue"]
    assert len(inv) == 2


def test_top_share_captured():
    actual = np.array([10, 0, 0, 0, 90])
    assert evaluate.top_share_captured(actual, actual, top=0.2) == 0.9
    assert evaluate.top_share_captured(actual, -actual, top=0.2) == 0.0


def test_error_metrics_perfect_prediction():
    a = np.array([0.0, 1.0, 3.0])
    m = evaluate.error_metrics(a, a)
    assert m["mae"] == 0 and m["total_error_pct"] == 0 and m["spearman"] == pytest.approx(1)


def test_calibration_table_buckets():
    t = evaluate.calibration_table(
        np.array([0, 1, 9, 12]), np.array([0, 1, 5, 7]), np.array([0, 1, 6, 6]), maxbucket=7
    )
    assert list(t.index) == ["0", "1", "7+"]
    assert t.loc["7+", "customers"] == 2
    assert t.loc["7+", "actual"] == 6
