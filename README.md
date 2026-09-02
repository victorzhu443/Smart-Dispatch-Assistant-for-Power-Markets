# Smart Dispatch Assistant for Power Markets

An end-to-end pipeline that ingests ERCOT wholesale electricity prices, engineers
time-series features, trains a price forecaster, and exposes both a `/forecast`
and a RAG-backed `/query` API behind a Django UI.

This is a learning/portfolio build following the phased plan in [`PRD.md`](PRD.md).
The data pipeline and the forecast service work. **The two ML results do not beat
their baselines, and the Django UI does not currently start** — see
[Results](#results) and [Known limitations](#known-limitations). Those numbers are
reproducible with the commands given below.

## Results

### Price forecasting (Phase 3) — trained on synthetic data, does not beat baseline

**Read this before quoting any forecasting number from this repo.** The database
holds 2,100 rows, but they span **two timestamps five minutes apart** across 1,034
settlement points. Feature engineering filters to a single hub, which leaves *one*
real hourly price point. `extend_hourly_data()` in
`feature-engineering/phase_2_4_feature_matrix_sql.py` then pads that up to 75
points with a seeded random walk (`last_price + N(0,3) + 5·sin(hour)`).

So **74 of 75 hourly points — and all 51 training windows — are synthetic.** The
`features` table's Aug 1–3 timestamps come from `datetime.now()` at generation
time, which is why they do not match the Jul 31 raw data. The forecasting result
below measures how well an LSTM fits a random walk, not how well it predicts
ERCOT prices.

The LSTM is trained for 5 epochs on those **51 sliding windows** (40 train / 11
test). Reproduce with:

```bash
python forecasting-model/phase_3_4_evaluate_rmse.py
```

| Predictor | RMSE | Beats model? |
| --- | --- | --- |
| **LSTM** | **$19.97 – $38.21** (varies per run) | — |
| Mean-of-training baseline | $12.51 | yes |
| Last-hour baseline | $13.22 | yes |

Across four runs the model scored $19.97, $38.21, $29.59 and $26.67, with R²
between −1.60 and −8.53. A negative R² means the model does worse than always
predicting the training mean. The baselines are identical every run because the
train/test split is seeded (`random_state=42`); the model's spread comes from
training being unseeded, so a single run is not a meaningful number to quote.

The failure mode is visible in the per-sample output: predictions collapse into a
narrow $18–$23 band while actuals range $18–$56, giving a −$16 mean bias. With 40
training samples and 57,441 parameters, the model has nowhere near enough data to
learn anything — and the increments it is being asked to learn are drawn from
`N(0, 3)`, which is unlearnable by construction.

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
| `frontend/` | 6 | Django project (`smart_ui`) with a `dashboard` app: Chart.js forecast page and chat page, proxying to the Flask APIs server-side to avoid CORS. |
| `k8s-*.yaml` | 5.3 | Namespace, deployments, services and ingress for local minikube. |

Phases 7 (monitoring) and 8 (dispatch simulation) from the PRD are not implemented.

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
python data-ingestion/phase_1_5_sql_storage.py        # → market_data table
python feature-engineering/phase_2_4_feature_matrix_sql.py  # → features table
python forecasting-model/phase_3_5_save_model.py      # → model.pt
python llm_rag/phase_4_1_embed_market_data.py         # → market_embeddings.json
python llm_rag/phase_4_2_finetune_llm.py              # → gpt2_dispatch_model/
```

Phases 1 and 4.2 are the slow ones; 4.2 needs a GPU to be comfortable. The query
API refuses to start without `gpt2_dispatch_model/` and `market_embeddings.json`.

### Running the services

```bash
python backend/phase_5_1_forecast_api.py    # http://localhost:5001
python backend/phase_5_2_minimal.py         # http://localhost:5002 (+ /chat)
```

```bash
curl http://localhost:5001/forecast
curl "http://localhost:5001/forecast?timestamp=2025-07-31T15:30:00"
curl -X POST http://localhost:5002/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Should we dispatch the gas peaker?"}'
```

Kubernetes (minikube) deployment: `bash backend/deploy_k8s.sh`.

## Known limitations

Ordered by how much they'd need fixing before this is more than a demo.

1. **The pipeline silently substitutes synthetic data when real data is thin.**
   Documented above and the most serious issue here. Every feature-engineering
   script carries an `extend_hourly_data()` / `simulate_historical_data()` path
   that fabricates random-walk prices when the real pull is too small, and it
   fires without failing or flagging the output — the `features` table looks
   identical whether it holds real or generated prices. This is what makes the
   forecasting result meaningless rather than merely bad. The pipeline should
   fail loudly on insufficient data, or at minimum tag generated rows so
   downstream consumers can tell the difference.
2. **Only five minutes of real market data was ever ingested.** The
   `market_data` table covers two timestamps (`2025-07-31 00:25:10` and
   `00:30:10`). The Phase 1 scripts pull with `size=100`–`1000` in a single
   request and were run once; there is no scheduled ingestion (PRD step 1.6 is
   unimplemented), so history never accumulated.
3. **Neither ML result beats its baseline.** Both are documented above. For the
   forecaster the cause is issues 1 and 2; for the LLM it is ~100 Q&A pairs,
   which is far too few to fine-tune on without catastrophic degradation.
4. **The Django UI does not start.** `manage.py` and `settings.py` reference a
   `smartui` module while the package directory is `smart_ui`, giving
   `ModuleNotFoundError: No module named 'smartui'`. Correcting that surfaces a
   second problem: `frontend/dashboard/` has no `__init__.py`, so Django raises
   `ImproperlyConfigured`. `smart_ui/urls.py` also imports `views` from its own
   package, but the views live in `dashboard/views.py`. PRD test case 6.1
   ("localhost:8000 loads basic UI") therefore does not pass.
5. **The forecast API does not use the trained LSTM.** `phase_5_1_forecast_api.py`
   trains its own `RandomForestRegressor` at startup and falls back to a
   mean-price constant if that fails. `model.pt` is only loaded by
   `model_usage_example.py`. The served forecast and the evaluated model are two
   different things.
6. **The forecast API's features are mostly hardcoded.** `predict_price()` passes
   fixed values for `price_mean`, `price_std`, `trend_slope` and momentum, varying
   only the time-derived features, then applies hand-tuned peak/off-peak
   multipliers. It is closer to a time-of-day heuristic than a learned model.
7. **`docker-compose.yml` cannot build.** It specifies `dockerfile: Dockerfile`,
   but the file in the repo is named `Dockerfile.txt`.
8. **Heavy duplication across phase scripts.** `setup_database_connection()` and
   the whole `ERCOTClient` class are copy-pasted verbatim into roughly ten files;
   a change to the auth flow means ten edits.
9. **No automated tests.** Each phase script self-checks by printing its own
   PASS/FAIL to stdout; there is no test runner, and `pytest` in
   `requirements.txt` is unused.
10. **Pinecone is not used.** Phase 4.1 writes embeddings to a local
   `market_embeddings.json` and retrieval does an in-memory cosine similarity,
   despite the PRD specifying a vector store.

## Next steps

- Make the synthetic-data fallback fail loudly instead of silently padding, or
  add an `is_synthetic` column so no result can be reported without knowing what
  it was computed on. Nothing else on this list matters until this is done.
- Implement scheduled ingestion (PRD step 1.6) and accumulate real ERCOT history,
  then re-run Phases 2–3. Only then is the forecasting number worth reading.
- Seed torch's RNG in the training scripts so runs are comparable.
- Fix the Django package naming and add the missing `__init__.py` so Phase 6 runs.
- Wire the forecast API to `model.pt` (or state plainly that it serves a
  RandomForest) so the served and evaluated models agree.
- Extract the shared DB and ERCOT client code into one importable module.

## License

MIT.

---

*Author: Victor Zhu. Requirements and per-step test cases are in [`PRD.md`](PRD.md).*
