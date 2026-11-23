"""Copy the columns the demo needs into app/data, so the hosted demo runs without the pipeline.

uv run python -m growthlab.exportdemo
"""

from __future__ import annotations

import pandas as pd

from growthlab import config

columns = {
    "clv_scores.parquet": ["customer_id", "clv_12m", "expected_purchases_52w", "p_alive"],
    "uplift_scores.parquet": ["treatment", "visit", "score_xlearner", "score_response", "baseline"],
    "decisions.parquet": None,  # already small: one row per customer
}


def main() -> None:
    config.appdatadir.mkdir(parents=True, exist_ok=True)
    for name, cols in columns.items():
        df = pd.read_parquet(config.processeddir / name, columns=cols)
        out = config.appdatadir / name
        df.to_parquet(out, index=False, compression="zstd")
        print(f"✓ {name}: {len(df):,} rows, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
