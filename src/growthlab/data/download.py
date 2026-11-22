"""Download the raw datasets and convert them to parquet.

Data is never committed to git. Run:

    uv run python -m growthlab.data.download --dataset all
    uv run python -m growthlab.data.download --dataset criteo --criteo-rows 1000000
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd
import requests

from growthlab import config

blocksize = 1 << 20  # 1 MiB
criteorows = 13_979_592  # rows in Criteo Uplift v2.1


def download(url: str, dest: Path) -> Path:
    """Stream `url` to `dest`, skipping the download if the file already exists."""
    if dest.exists():
        print(f"✓ {dest.name} already downloaded")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"↓ {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for block in r.iter_content(blocksize):
                f.write(block)
                done += len(block)
                if total:
                    print(f"\r  {done / total:6.1%} of {total / 1e6:.0f} MB", end="")
    print()
    tmp.rename(dest)
    return dest


def fetch_online_retail() -> Path:
    """Online Retail II: UK online retailer transactions, Dec 2009 – Dec 2011."""
    out = config.retailparquet
    if out.exists():
        print(f"✓ {out.name} ready")
        return out

    if not config.retailxlsx.exists():
        zpath = download(config.retailurl, config.rawdir / "online_retail_II.zip")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(config.rawdir)

    print("  reading xlsx (both sheets, ~1M rows — takes a minute)…")
    sheets = pd.read_excel(config.retailxlsx, sheet_name=None, dtype={"Invoice": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df.rename(
        columns={
            "Invoice": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Customer ID": "customer_id",
            "Country": "country",
        }
    )
    # a few cells in these text columns are parsed as numbers by Excel
    for col in ("invoice", "stock_code", "description", "country"):
        df[col] = df[col].astype("string")
    df["customer_id"] = df["customer_id"].astype("Int64")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✓ {out.name}: {len(df):,} rows")
    return out


def fetch_criteo(nrows: int | None = None) -> Path:
    """Criteo Uplift v2.1: ~14M users from a randomized advertising experiment.

    `nrows` keeps a uniform random sample of ~N rows. The file is NOT shuffled (it is ordered by
    treatment group), so taking the first N rows would give a treatment-only sample.
    """
    out = config.criteoparquet
    if out.exists():
        print(f"✓ {out.name} ready")
        return out

    gz = download(config.criteourl, config.rawdir / "criteo-uplift-v2.1.csv.gz")
    frac = None if nrows is None else min(1.0, nrows / criteorows)
    dtypes = {f"f{i}": "float32" for i in range(12)} | {
        c: "int8" for c in ("treatment", "conversion", "visit", "exposure")
    }

    print("  reading csv in chunks…")
    chunks = []
    for chunk in pd.read_csv(gz, dtype=dtypes, chunksize=1_000_000):
        if frac is not None:
            chunk = chunk.sample(frac=frac, random_state=config.seed)
        chunks.append(chunk)
    # shuffle so that downstream splits never depend on the file's original ordering
    df = pd.concat(chunks).sample(frac=1, random_state=config.seed).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✓ {out.name}: {len(df):,} rows, treatment rate {df['treatment'].mean():.1%}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["retail", "criteo", "all"], default="all")
    parser.add_argument(
        "--criteo-rows",
        dest="criteorows",
        type=int,
        default=None,
        help="keep a random sample of ~N Criteo rows",
    )
    args = parser.parse_args()

    if args.dataset in ("retail", "all"):
        fetch_online_retail()
    if args.dataset in ("criteo", "all"):
        fetch_criteo(args.criteorows)


if __name__ == "__main__":
    main()
