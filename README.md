# Smart Dispatch Assistant for Power Markets

An end-to-end pipeline that ingests ERCOT wholesale electricity prices, engineers
time-series features, trains a price forecaster, and exposes both a `/forecast`
and a RAG-backed `/query` API behind a Django UI.

This is a learning/portfolio build following the phased plan in [`PRD.md`](PRD.md).
It is mid-rebuild. **The forecasting pipeline deliberately refuses to run on the
current database** — there is only five minutes of real market data, and it now
fails loudly rather than padding the gap with generated prices, which is what it
used to do. The RAG fine-tuning result is real and is worse than its base model.
The Django UI does not start. See [Results](#results) and
[Known limitations](#known-limitations); every number below is reproducible with
the command next to it.

## Results

### Price forecasting (Phase 3) — no current result

**The pipeline does not currently produce a forecasting number, by design.**
There is not enough real data to train on, and it now says so instead of
inventing the difference.

The database holds 2,100 rows, but they span **two timestamps five minutes
apart** across 1,034 settlement points. Feature engineering filters to a single
hub, which leaves *one* real hourly price point — far below the 75 needed to
build 24-hour windows. Running either stage fails loudly:

```console
$ python feature-engineering/phase_2_4_feature_matrix_sql.py
InsufficientDataError: Need 75 hourly observations to build 24-hour windows,
have 1 for settlement point '7RNCHSLR_ALL'.

$ python forecasting-model/phase_3_4_evaluate_rmse.py
InsufficientDataError: Refusing to train on 51 feature windows; 1000 required.
```

Restoring a real number requires ingesting real history, not changing the model.

<details>
<summary><strong>The result this replaced, and why it was withdrawn</strong></summary>

Until the M0 fix, `extend_hourly_data()` padded that single real observation up
to 75 points with a seeded random walk (`last_price + N(0,3) + 5·sin(hour)`), so
**74 of 75 hourly points — and all 51 training windows — were synthetic.** The
`features` table's Aug 1–3 timestamps came from `datetime.now()` at generation
time, which is why they never matched the Jul 31 raw data.

The LSTM trained for 5 epochs on those 51 windows scored as follows — a measure
of how well it fit a random walk, not of anything about ERCOT prices:

| Predictor | RMSE | Beats model? |
| --- | --- | --- |
| **LSTM** | **$19.97 – $38.21** (varies per run) | — |
| Mean-of-training baseline | $12.51 | yes |
| Last-hour baseline | $13.22 | yes |

Across four runs the model scored $19.97, $38.21, $29.59 and $26.67, with R²
between −1.60 and −8.53. A negative R² means it did worse than always predicting
the training mean. The baselines were identical every run because the split was
seeded; the model's spread came from training being unseeded, so no single run
was a meaningful number to quote.

Predictions collapsed into a narrow $18–$23 band while actuals ranged $18–$56, a
−$16 mean bias. With 40 training samples and 57,441 parameters there was nothing
to learn — and the increments it was asked to learn were drawn from `N(0, 3)`,
unlearnable by construction. The split was also shuffled, so the model was
scored on hours that preceded ones it trained on; that leakage *inflates*
scores, and it still lost to a constant.

</details>

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
3. **Only five minutes of real market data was ever ingested.** The
   `market_data` table covers two timestamps (`2025-07-31 00:25:10` and
   `00:30:10`). The Phase 1 scripts pull with `size=100`–`1000` in a single
   request and were run once; there is no scheduled ingestion (PRD step 1.6 is
   unimplemented), so history never accumulated.
4. **Neither ML result beat its baseline.** Both are documented above. For the
   forecaster the cause was issues 1 and 2; for the LLM it is ~100 Q&A pairs,
   which is far too few to fine-tune on without catastrophic degradation.
5. **The Django UI does not start.** `manage.py` and `settings.py` reference a
   `smartui` module while the package directory is `smart_ui`, giving
   `ModuleNotFoundError: No module named 'smartui'`. Correcting that surfaces a
   second problem: `frontend/dashboard/` has no `__init__.py`, so Django raises
   `ImproperlyConfigured`. `smart_ui/urls.py` also imports `views` from its own
   package, but the views live in `dashboard/views.py`. PRD test case 6.1
   ("localhost:8000 loads basic UI") therefore does not pass.
6. **The forecast API does not use the trained LSTM.** `phase_5_1_forecast_api.py`
   trains its own `RandomForestRegressor` at startup and falls back to a
   mean-price constant if that fails. `model.pt` is only loaded by
   `model_usage_example.py`. The served forecast and the evaluated model are two
   different things.
7. **The forecast API's features are mostly hardcoded.** `predict_price()` passes
   fixed values for `price_mean`, `price_std`, `trend_slope` and momentum, varying
   only the time-derived features, then applies hand-tuned peak/off-peak
   multipliers. It is closer to a time-of-day heuristic than a learned model.
8. **`docker-compose.yml` cannot build.** It specifies `dockerfile: Dockerfile`,
   but the file in the repo is named `Dockerfile.txt`.
9. **Heavy duplication across phase scripts.** `setup_database_connection()` and
   the whole `ERCOTClient` class are copy-pasted verbatim into roughly ten files;
   a change to the auth flow means ten edits.
10. **Test coverage is one module deep.** `tests/test_chronological_split.py`
   covers the train/test split; everything else still "self-tests" by printing
   its own PASS/FAIL to stdout, which passes just as happily on fabricated data.
   Data-validation and pipeline tests are the gap. Run what exists with
   `pytest tests/ -v`.
11. **Pinecone is not used.** Phase 4.1 writes embeddings to a local
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
