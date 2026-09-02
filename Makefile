# Pipeline orchestration.
#
# Run order used to live only in the README, which is why the features table
# fell eight months behind the raw data: nothing enforced that it rebuild after
# ingestion. Make encodes the dependencies instead, so `make features` cannot
# run against data that has not been ingested.
#
# This is deliberately not Airflow. For a single-machine pipeline with four
# steps, Make plus cron does the job and adds no operational surface. Reach for
# an orchestrator when you need retries, backfill windows and a scheduler UI.

PYTHON ?= python
DB     ?= market_data.db
ARTIFACT ?= forecast_model.joblib
YEARS  ?= 2024 2025 2026
HUB    ?= HB_HOUSTON

.PHONY: help setup schema check-schema backfill ingest features train \
        backtest dispatch serve test lint clean all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies
	$(PYTHON) -m pip install -r requirements_minimal.txt

schema:  ## Apply the database schema (safe to re-run)
	$(PYTHON) data-ingestion/schema.py --migrate

check-schema:  ## Report schema drift without changing anything
	$(PYTHON) data-ingestion/schema.py --check

$(DB): schema

backfill: schema  ## Backfill history from ERCOT annual archives
	$(PYTHON) data-ingestion/ingest_ercot_history.py --years $(YEARS)

ingest: schema  ## Top up to the current interval (cron target)
	$(PYTHON) data-ingestion/ingest_recent.py

features: ingest  ## Rebuild the feature matrix from current data
	$(PYTHON) feature-engineering/phase_2_4_feature_matrix_sql.py

train: features  ## Retrain the served model if it has fallen behind
	$(PYTHON) forecasting-model/train_model.py --if-stale-days 7 --hub $(HUB)

backtest:  ## Score the baselines over all history
	$(PYTHON) forecasting-model/backtest.py --hub $(HUB)

walk-forward:  ## Rolling-origin validation of the model against baselines
	$(PYTHON) forecasting-model/walk_forward.py --hub $(HUB)

dispatch:  ## Simulate dispatch decisions and report profit
	$(PYTHON) forecasting-model/dispatch.py --hub $(HUB)

serve:  ## Run the forecast API on :5001
	$(PYTHON) backend/phase_5_1_forecast_api.py

ui:  ## Run the Django dashboard on :8000
	cd frontend && $(PYTHON) manage.py runserver

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

lint:  ## Byte-compile every tracked Python file
	@git ls-files '*.py' | xargs -n1 $(PYTHON) -m py_compile && echo "all files compile"

clean:  ## Remove caches and compiled files
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete

all: train test  ## Ingest, rebuild features, retrain if stale, and test
