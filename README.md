# Smart Dispatch Assistant for Power Markets

An end-to-end pipeline that ingests ERCOT wholesale electricity prices, engineers
time-series features, trains a price forecaster, and exposes both a `/forecast`
and a RAG-backed `/query` API behind a Django UI.

This is a learning/portfolio build following the phased plan in [`PRD.md`](PRD.md).
It is mid-rebuild, and the rebuild is working. The forecaster trains on
**24 months of real ERCOT prices** and beats the day-ahead market by 8.6% under
walk-forward validation, where the previous version trained on generated data
and lost to a constant. Turned into dispatch decisions, that edge captures
**79.8% of the profit perfect foresight would earn** — 85% more than bidding
off the day-ahead price. It still cannot predict scarcity hours, which is the
next problem. The RAG fine-tuning result is real and is worse than its base model.
See [Results](#results) and
[Known limitations](#known-limitations); every number below is reproducible with
the command next to it.

## Results

### Price forecasting (Phase 3) — beats the market by 8.6%, fails on scarcity

Evaluated **walk-forward**: retrain each month on everything prior, predict the
next month, step forward. 19 folds, **13,182 out-of-sample hours** at
HB_HOUSTON. Every predictor scored on exactly the same hours.

```bash
python data-ingestion/ingest_ercot_history.py     # ~1 min, no credentials
python forecasting-model/walk_forward.py
```

| Predictor | RMSE | MAE | Peak hours | **Scarcity hours** |
| --- | --- | --- | --- | --- |
| **Ridge regression** | **$37.45** | $8.29 | $63.59 | $619.06 |
| Gradient boosting | $39.12 | $7.77 | $66.30 | $635.17 |
| Day-ahead (the market) | $40.98 | $9.80 | $65.89 | $660.22 |
| Persistence | $45.96 | $8.04 | $67.53 | $717.00 |
| Same hour yesterday | $58.16 | $14.49 | $95.94 | $658.49 |

Three things this says, and the third is the important one.

**It beats the market, modestly and consistently.** Ridge is 8.6% better than
the day-ahead price across 19 independent monthly folds. Day-ahead is the
market's own published forecast, so a consistent edge over it is a real
result — but 8.6%, not the 25% a single calm test window suggested.

**The simplest model won.** Ridge regression beat gradient boosting. Whatever
non-linear structure the trees found did not survive contact with out-of-sample
months, and the extra complexity earned nothing. Worth remembering before
reaching for a bigger architecture.

**Nobody can predict scarcity.** On the 41 hours at or above $200/MWh, every
predictor is wrong by roughly $600–700. Those are the hours that decide whether
a peaker earns its year, and they are effectively unforecast — by this model
and by the market alike. The headline RMSE is dominated by the 99.7% of hours
where the decision is easy anyway.

So the useful conclusion is not "the model works." It is that the remaining
value in this problem is entirely in the tail, and a point forecast is the
wrong tool for it. That is what motivates quantile forecasting and an explicit
dispatch rule as the next step.

<details>
<summary><strong>The earlier single-split result, and why it overstated things</strong></summary>

A single chronological split (train through 2025-08-08, test on the 3,504
hours after) gave an LSTM $11.70 RMSE against day-ahead's $15.62 — a 25% edge,
R² 0.695.

That window turned out to be unusually calm: it peaks at $214/MWh and holds
only 3 hours above $200, against 74 across the full two years. Day-ahead scores
$15.62 there versus $40.98 walking forward, which confirms the window was easy
rather than the model being exceptional.

The single-split number was not wrong, but it was not representative, and it is
the reason walk-forward exists. Reproduce it with
`python forecasting-model/phase_3_4_evaluate_rmse.py`.

Note also that `phase_3_4`'s "last-hour baseline" is not persistence — it is
`np.mean(y_train[-3:])` held constant across the whole test set, which is why
it scores almost identically to the training mean. The real baselines are in
`backtest.py` and `walk_forward.py`.

</details>

### Dispatch decisions (Phase 8) — 79.8% of theoretically available profit

The forecast only matters if it changes a decision. This simulates a 100 MW gas
peaker at $45/MWh marginal cost and $5,000 per start over the same 13,182
out-of-sample hours, charging a start cost on every off-to-on transition.

```bash
python forecasting-model/dispatch.py
```

Quantiles come from gradient boosting fit with pinball loss at P10/P50/P90,
retrained monthly. **Calibration is checked, not assumed** — if the P90 only
covered 70% of outcomes, every expected value built on it would be wrong:

| Quantile | Nominal | Observed | Error |
| --- | --- | --- | --- |
| P10 | 10% | 12.3% | +2.3pp |
| P50 | 50% | 50.6% | +0.6pp |
| P90 | 90% | 87.3% | −2.7pp |

| Strategy | Profit | vs perfect | Hours run | Starts |
| --- | --- | --- | --- | --- |
| Perfect foresight (ceiling) | $3,466,014 | 100.0% | 1,700 | 518 |
| **Model — P50 threshold** | **$2,766,476** | **79.8%** | 1,460 | 487 |
| Bidding off the day-ahead price | $1,494,104 | 43.1% | 1,897 | 521 |
| Model — expected margin | $1,203,587 | 34.7% | 2,734 | 723 |
| Model — P90 (aggressive) | $950,661 | 27.4% | 2,948 | 740 |
| Never run | $0 | 0.0% | 0 | 0 |
| Always run | −$18,654,296 | −538.2% | 13,182 | 1 |

**A modest accuracy edge becomes a large economic one.** The model beats
day-ahead by 8.6% on RMSE but by **85%** on realised profit. That is not a
contradiction: dispatch is a threshold decision, so being right *near the
threshold* matters enormously and being wrong in the middle of a distribution
barely matters at all. It is the clearest argument in this project for scoring
in dollars rather than RMSE.

**The sophisticated rules lost to the simple one.** The expected-margin rule
that uses the full P10/P50/P90 distribution earned $1.2M against the plain P50
threshold's $2.8M, and the aggressive P90 rule did worse still. Both over-commit
— 2,700–2,900 hours run against P50's 1,460 — and bleed the difference in start
costs. The distribution is well calibrated; the decision rule built on it is
simply too permissive. Tuning its threshold on this same data would be fitting
to the evaluation set, so it is left as reported.

### RAG fine-tuning (Phase 4) — worse than the base model

GPT-2 fine-tuned on ~100 dispatch Q&A pairs, evaluated against GPT-generated
reference answers. Numbers from
[`perplexity_evaluation_results.json`](perplexity_evaluation_results.json)
(`"test_passed": false`), charted in `perplexity_analysis.png`:

| Model | Avg. perplexity |
| --- | --- |
| Base GPT-2 | 54.0 |
| Fine-tuned GPT-2 | 139.4 (**−158%**) |

**0 of 10** evaluation questions improved. The fine-tuned model degenerates into
repeating settlement-point identifiers, e.g.:

> No dispatch recommended. Price of $32.61/MWh is below marginal costs. Current
> market conditions show moderate pricing at
> AMOCO_AMOCO_RN.RN.SUB_ALL.GBP.GBP.GB_RN.RN.SUB_ALL.GBP…

Training loss fell from 4.4 to 0.55 over ~110 steps (`training_history.png`) while
output quality dropped — the model memorized a tiny corpus instead of learning the
task.

## Repository layout

| Path | Phase | Contents |
| --- | --- | --- |
| `data-ingestion/` | 1 | ERCOT OAuth client and price fetch (`phase_1_3` → `phase_1_5`), NaN handling, SQL persistence. Additional PJM (`ingest_jpm.py`) and MISO (`ingest_miso.py`) ingesters. |
| `feature-engineering/` | 2 | SQL load → hourly resample → 24h sliding windows → technical features (mean, std, trend slope, moving averages, momentum) → `features` table. |
| `forecasting-model/` | 3 | `PowerMarketLSTM` in PyTorch: load features, define architecture, train, evaluate RMSE vs. baselines, serialize to `model.pt`. |
| `llm_rag/` | 4 | SentenceTransformers embeddings → `market_embeddings.json`, GPT-2 fine-tuning, retrieval `/query` endpoint, perplexity evaluation. |
| `backend/` | 5 | Flask services: `phase_5_1_forecast_api.py` (port 5001) and `phase_5_2_minimal.py` / `phase_5_2_query_api.py` (port 5002), plus Docker and minikube deploy scripts. |
| `frontend/` | 6 | Django project (`smart_ui`) with a `dashboard` app: the dispatch outlook (P10–P90 band, marginal-cost line, per-hour run/hold call) and a chat page, proxying to the Flask APIs server-side to avoid CORS. |
| `k8s-*.yaml` | 5.3 | Namespace, deployments, services and ingress for local minikube. |

Phase 7 (monitoring) from the PRD is not implemented. Phase 8 (dispatch
simulation) is covered by `forecasting-model/dispatch.py`.

## Setup

Requires **Python 3.10+** (the Django pin needs ≥3.10) and, for the containerized
services, Docker.

```bash
git clone https://github.com/victorzhu443/Smart-Dispatch-Assistant-for-Power-Markets.git
cd Smart-Dispatch-Assistant-for-Power-Markets

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_minimal.txt   # both APIs, incl. torch + transformers

cp .env.example .env                      # then fill in your credentials
```

`.env.example` documents every variable and which script consumes it. An ERCOT
API subscription (free, from <https://apiexplorer.ercot.com/>) is needed to ingest
new data. Every pipeline script attempts PostgreSQL first and silently falls back
to a local SQLite file, so Postgres is optional for local work.

### Generating the data and models

`market_data.db`, `model.pt`, `market_embeddings.json` and `gpt2_dispatch_model/`
are build artifacts and are not tracked in git. Regenerate them in order:

```bash
# Backfill from ERCOT's annual archives (no credentials needed)
python data-ingestion/ingest_ercot_history.py --years 2024 2025 2026

# Top up to the current interval; safe to re-run, safe on cron
python data-ingestion/ingest_recent.py

# Train the served model
python forecasting-model/train_model.py               # → forecast_model.joblib
```

Keep it current with cron:

```cron
*/15 * * * * cd /path/to/repo && python data-ingestion/ingest_recent.py >> logs/ingest.log 2>&1
```

Both ingestion paths are idempotent on `(timestamp, settlement_point)`, so a
re-run or an overlapping window converges rather than duplicating.

Phases 1 and 4.2 are the slow ones; 4.2 needs a GPU to be comfortable. The query
API refuses to start without `gpt2_dispatch_model/` and `market_embeddings.json`.

### Running the services

The forecast API serves the trained artifact, so train it first:

```bash
python forecasting-model/train_model.py     # -> forecast_model.joblib
python backend/phase_5_1_forecast_api.py    # http://localhost:5001
python backend/phase_5_2_minimal.py         # http://localhost:5002 (+ /chat)
```

```bash
curl "http://localhost:5001/forecast?timestamp=2025-12-15T18:00:00Z"
curl "http://localhost:5001/forecast?timestamp=2025-12-15T18:00:00Z&marginal_cost=55"
curl http://localhost:5001/health
curl -X POST http://localhost:5002/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Should we dispatch the gas peaker?"}'
```

Every forecast response carries provenance — which model answered, its
version, the training cutoff, and whether any fallback was engaged:

```json
"provenance": {
  "model": "quantile_gbm",
  "model_version": "93af19a3f79c",
  "degraded": false,
  "training_cutoff": "2026-01-01T05:00:00+00:00",
  "data_age_hours": 5856.1
}
```

The Django UI reads the same API:

```bash
cd frontend && python manage.py runserver    # http://localhost:8000
```

The dashboard draws the **P10–P90 band with the marginal-cost line across it**,
colours each hour by the dispatch call, and lets you change the marginal cost to
see the call move. If the API is serving a fallback, the page says so in a banner
rather than presenting degraded numbers as a model output.

### Fallback policy — degrade visibly, never invent

Every degraded path is detectable by the caller through
`provenance.degraded` and `provenance.model`. The service will not return
something that looks like a full-confidence answer when it is not.

| Failure | Response |
| --- | --- |
| Model artifact missing or unreadable | Serves seasonal-naive, `"model": "seasonal_naive_fallback"`, `degraded: true` |
| Live forecast requested on stale data | `503` naming the age and the limit; an explicit historical `?timestamp=` still works |
| Requested hour not in the data | `422` with `latest_available` |
| Invalid timestamp or marginal cost | `422` naming the field |
| P10–P90 spread beyond the threshold | Forecast returned, `"action": "no_action"`, `"reason": "interval_too_wide"` |
| Database unreachable | `503`, health check red |

Health reflects real readiness rather than a constant. Stale data reports
`degraded`, not `healthy` — otherwise an orchestrator keeps routing live
traffic that will 503.

Kubernetes (minikube) deployment: `bash backend/deploy_k8s.sh`.

## Known limitations

Ordered by how much they'd need fixing before this is more than a demo.

1. ~~**The pipeline silently substitutes synthetic data when real data is
   thin.**~~ **Fixed.** Every feature-engineering script carried an
   `extend_hourly_data()` / `simulate_historical_data()` path that fabricated
   random-walk prices when the real pull was too small, and it fired without
   failing or flagging the output — the `features` table looked identical
   whether it held real or generated prices. Those 279 lines are removed; the
   pipeline now raises `InsufficientDataError` with the actual and required row
   counts, and the Phase 3 scripts refuse to train on fewer than 1,000 windows.
2. ~~**The train/test split leaked future data.**~~ **Fixed.** Every Phase 3
   script called `train_test_split()` without `shuffle=False`, so a time series
   was randomly shuffled and the model was scored on hours preceding ones it
   trained on. Now a chronological split, asserted in
   `tests/test_chronological_split.py`.
3. ~~**Only five minutes of real market data was ever ingested.**~~
   **Fixed.** `data-ingestion/ingest_ercot_history.py` pulls 24 months of
   hourly hub prices plus day-ahead prices from ERCOT's public MIS, which
   needs no credentials, and `ingest_recent.py` tops up from the rolling
   15-minute feed (PRD step 1.6), so a cron entry keeps the window current.
4. **Scarcity hours are unforecast.** Walk-forward across 19 folds, every
   predictor — including the market's own day-ahead price — is wrong by
   roughly $600–700 on hours above $200/MWh. Those are the hours that decide
   whether a peaker earns its year, so the 8.6% headline edge is earned almost
   entirely on hours where the dispatch decision is easy anyway. A point
   forecast is the wrong tool here; quantile forecasting plus an explicit
   dispatch rule is the next step.
5. **The RAG fine-tuning made the model worse** and is still in the tree.
   ~100 Q&A pairs is far too few to fine-tune on without catastrophic
   degradation; the retrieval half works and the fine-tuning half should
   probably be dropped.
6. **The Django UI does not start.** `manage.py` and `settings.py` reference a
   `smartui` module while the package directory is `smart_ui`, giving
   `ModuleNotFoundError: No module named 'smartui'`. Correcting that surfaces a
   second problem: `frontend/dashboard/` has no `__init__.py`, so Django raises
   `ImproperlyConfigured`. `smart_ui/urls.py` also imports `views` from its own
   package, but the views live in `dashboard/views.py`. PRD test case 6.1
   ("localhost:8000 loads basic UI") therefore does not pass.
7. ~~**The forecast API does not use the trained model.**~~ **Fixed.** It
   trained a `RandomForestRegressor` at import time and predicted from
   hardcoded feature values, so the thing served was never the thing
   evaluated. It now loads the versioned artifact from
   `forecasting-model/train_model.py` and reports which model answered.
8. ~~**The forecast API's features are mostly hardcoded.**~~ **Fixed.**
   Features are now built from real history by the same code path used in
   training, so serving and evaluation cannot drift apart.
9. **`docker-compose.yml` cannot build.** It specifies `dockerfile: Dockerfile`,
   but the file in the repo is named `Dockerfile.txt`.
10. **Heavy duplication across phase scripts.** `setup_database_connection()` and
   the whole `ERCOTClient` class are copy-pasted verbatim into roughly ten files;
   a change to the auth flow means ten edits.
11. **Test coverage is one module deep.** `tests/test_chronological_split.py`
   covers the train/test split; everything else still "self-tests" by printing
   its own PASS/FAIL to stdout, which passes just as happily on fabricated data.
   Data-validation and pipeline tests are the gap. Run what exists with
   `pytest tests/ -v`.
12. **Pinecone is not used.** Phase 4.1 writes embeddings to a local
   `market_embeddings.json` and retrieval does an in-memory cosine similarity,
   despite the PRD specifying a vector store.

## Next steps

- ~~Make the synthetic-data fallback fail loudly instead of silently padding.~~
  Done — the pipeline now raises `InsufficientDataError`.
- ~~Seed torch's RNG so runs are comparable.~~ Done.
- **Ingest real ERCOT history.** This is the only thing that unblocks a real
  forecasting result. Target 24 months of hourly settlement point prices for a
  handful of hubs, plus scheduled ingestion (PRD step 1.6), then re-run
  Phases 2–3.
- Build a walk-forward backtest and baseline set (persistence, seasonal-naive,
  and the day-ahead price) *before* tuning any model, so improvements can be
  distinguished from noise.
- Fix the Django package naming and add the missing `__init__.py` so Phase 6 runs.
- Wire the forecast API to `model.pt` (or state plainly that it serves a
  RandomForest) so the served and evaluated models agree.
- Extract the shared DB and ERCOT client code into one importable module.

## License

MIT.

---

*Author: Victor Zhu. Requirements and per-step test cases are in [`PRD.md`](PRD.md).*
