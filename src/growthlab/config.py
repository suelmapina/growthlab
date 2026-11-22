"""Project-wide paths and constants."""

from pathlib import Path

root = Path(__file__).resolve().parents[2]
datadir = root / "data"
rawdir = datadir / "raw"
processeddir = datadir / "processed"
modelsdir = root / "models"

seed = 42

# Online Retail II (UCI ML Repository, CC BY 4.0)
retailurl = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
retailxlsx = rawdir / "online_retail_II.xlsx"
retailparquet = processeddir / "online_retail.parquet"

# Criteo Uplift Prediction Dataset v2.1 (Criteo AI Lab, CC BY-NC-SA 4.0)
# official mirror on the Criteo Hugging Face account (the old go.criteo.net link is dead)
criteourl = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
criteoparquet = processeddir / "criteo_uplift.parquet"
