.PHONY: help install data data-sample clv clv-mcmc uplift decision all figures api demo lint format test mlflow clean

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create .venv and install all dependencies from uv.lock
	uv sync

data:  ## Download Online Retail II + full Criteo Uplift (~300 MB compressed)
	uv run python -m growthlab.data.download --dataset all

data-sample:  ## Download Online Retail II + a random 1M-row Criteo sample
	uv run python -m growthlab.data.download --dataset all --criteo-rows 1000000

clv:  ## Module 1: train & evaluate CLV models (fast MAP fit)
	uv run python -m growthlab.clv.train

clv-mcmc:  ## Module 1 with full posterior (slower, adds credible intervals)
	uv run python -m growthlab.clv.train --method mcmc

uplift:  ## Module 2: fit and compare uplift models on Criteo (~4 min)
	uv run python -m growthlab.uplift.train

decision:  ## Module 3: simulate targeting policies, write per-customer decisions
	uv run python -m growthlab.decision.simulate

all: clv uplift decision  ## Run the full pipeline (value → uplift → decision)

figures:  ## Re-render README figures from saved results (no retraining)
	uv run python -m growthlab.figures

api:  ## Serve the decision API on http://localhost:8000/docs
	uv run uvicorn growthlab.api.main:app --reload

demo:  ## Open the Streamlit demo on http://localhost:8501
	uv run streamlit run src/growthlab/demo.py

lint:  ## Lint with ruff
	uv run ruff check src tests

format:  ## Auto-format with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:  ## Run the test suite
	uv run pytest -q

mlflow:  ## Open the MLflow UI on http://localhost:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:  ## Remove caches (keeps data and .venv)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
