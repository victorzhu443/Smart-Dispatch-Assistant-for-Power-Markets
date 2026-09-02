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
        quality check-fresh monitor backtest dispatch serve test lint clean all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies
	$(PYTHON) -m pip install -r requirements_minimal.txt

schema:  ## Apply the database schema (safe to re-run)
	$(PYTHON) -m data_ingestion.schema --migrate

check-schema:  ## Report schema drift without changing anything
	$(PYTHON) -m data_ingestion.schema --check

$(DB): schema

backfill: schema  ## Backfill history from ERCOT annual archives
	$(PYTHON) -m data_ingestion.ingest_ercot_history --years $(YEARS)

ingest: schema  ## Top up to the current interval (cron target)
	$(PYTHON) -m data_ingestion.ingest_recent

features: ingest  ## Rebuild the feature matrix from current data
	$(PYTHON) -m feature_engineering.phase_2_4_feature_matrix_sql

train: features  ## Retrain the served model if it has fallen behind
	$(PYTHON) -m forecasting_model.train_model --if-stale-days 7 --hub $(HUB)

quality:  ## Report table freshness and recent quality checks
	$(PYTHON) -m data_ingestion.quality --report

check-fresh:  ## Exit non-zero if any table is stale (cron/alerting target)
	$(PYTHON) -m data_ingestion.quality --check

monitor:  ## Alert if the pipeline has stopped or is failing (cron target)
	$(PYTHON) -m data_ingestion.monitor

backtest:  ## Score the baselines over all history
	$(PYTHON) -m forecasting_model.backtest --hub $(HUB)

walk-forward:  ## Rolling-origin validation of the model against baselines
	$(PYTHON) -m forecasting_model.walk_forward --hub $(HUB)

dispatch:  ## Simulate dispatch decisions and report profit
	$(PYTHON) -m forecasting_model.dispatch --hub $(HUB)

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
