"""Train the quantile forecaster and save a versioned artifact for serving.

Separating training from serving is the point of this module. The forecast API
previously trained a RandomForest at import time, on hardcoded feature values,
every time the process started -- so the thing being served was never the thing
that had been evaluated, and no two deployments served the same model.

Here the model is trained once, deliberately, and written to disk with enough
metadata for the API to describe exactly what it is answering with:

    version        content hash of hub + cutoff + features + library versions
    trained_at     when this artifact was produced
    data_cutoff    the last hour of training data, so staleness is measurable
    calibration    measured on a held-out tail, not assumed
    feature_names  so the server can fail loudly on a schema mismatch

Usage:
    python forecasting-model/train_model.py
    python forecasting-model/train_model.py --hub HB_WEST --holdout-months 3

Cron (retrain only once the model has fallen a week behind the data):
    0 4 * * * cd /path/to/repo && python forecasting-model/train_model.py \\
              --if-stale-days 7 >> logs/train.log 2>&1

Without something like that entry, the served model drifts away from the
market exactly as the data drifted away from reality before scheduled
ingestion existed. The API reports the lag either way, so the failure is at
least visible rather than silent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

# Sibling module: this file runs from inside forecasting-model/, which is on
# sys.path when invoked as a script.
from forecasting_model import walk_forward as wf

QUANTILES = (0.10, 0.50, 0.90)
SEED = 42
DEFAULT_ARTIFACT = Path("forecast_model.joblib")


class TrainingError(RuntimeError):
    pass


def _version_hash(hub: str, cutoff: pd.Timestamp, features: list[str]) -> str:
    """Stable identifier for what this artifact actually is.

    Includes library versions because a model pickled by one sklearn and loaded
    by another is a real source of silent behaviour change.
    """
    payload = json.dumps({
        "hub": hub,
        "cutoff": cutoff.isoformat(),
        "features": features,
        "quantiles": list(QUANTILES),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def train(df: pd.DataFrame, hub: str, holdout_months: int) -> dict:
    """Fit quantile models, measuring calibration on a held-out tail."""
    feat = wf.build_features(df).dropna(subset=wf.FEATURE_COLUMNS + ["price"])
    if len(feat) < 1000:
        raise TrainingError(
            f"Refusing to train on {len(feat)} rows; 1000 required. "
            f"Run data-ingestion/ingest_ercot_history.py first."
        )

    boundaries = wf.month_starts(feat["timestamp_utc"])
    if len(boundaries) <= holdout_months:
        raise TrainingError(
            f"Need more than {holdout_months} months of data to hold any out"
        )

    holdout_start = boundaries[-holdout_months]
    is_fit = feat["timestamp_utc"] < holdout_start
    is_holdout = ~is_fit

    X_fit = feat.loc[is_fit, wf.FEATURE_COLUMNS].to_numpy()
    y_fit = feat.loc[is_fit, "price"].to_numpy()
    X_holdout = feat.loc[is_holdout, wf.FEATURE_COLUMNS].to_numpy()
    y_holdout = feat.loc[is_holdout, "price"].to_numpy()

    print(f"Fitting on {is_fit.sum():,} hours, "
          f"holding out {is_holdout.sum():,} from {holdout_start:%Y-%m-%d}")

    models = {}
    for q in QUANTILES:
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=300, learning_rate=0.06,
            min_samples_leaf=40, l2_regularization=1.0, random_state=SEED,
        )
        model.fit(X_fit, y_fit)
        models[q] = model
        print(f"  fitted P{int(q * 100)}")

    # Calibration on data the models never saw.
    calibration = {}
    for q, model in models.items():
        predicted = model.predict(X_holdout)
        covered = float((y_holdout <= predicted).mean())
        calibration[f"p{int(q * 100)}"] = {
            "nominal": q, "observed": round(covered, 4),
            "error_pp": round((covered - q) * 100, 2),
        }

    # Refit on everything now that calibration is measured, so the served model
    # uses all available history.
    X_all = feat[wf.FEATURE_COLUMNS].to_numpy()
    y_all = feat["price"].to_numpy()
    for q, model in models.items():
        model.fit(X_all, y_all)

    cutoff = feat["timestamp_utc"].max()
    return {
        "models": models,
        "quantiles": list(QUANTILES),
        "feature_names": list(wf.FEATURE_COLUMNS),
        "hub": hub,
        "data_cutoff": cutoff,
        "trained_at": datetime.now(timezone.utc),
        "calibration": calibration,
        "training_rows": int(len(feat)),
        "sklearn_version": sklearn.__version__,
        "version": _version_hash(hub, cutoff, list(wf.FEATURE_COLUMNS)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", default="HB_HOUSTON")
    parser.add_argument("--holdout-months", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--if-stale-days", type=float, default=None,
        help=("only retrain when the existing artifact is this many days "
              "behind the data; exit 0 otherwise. Makes this safe on cron."),
    )
    args = parser.parse_args()

    engine = wf.setup_database_connection()
    df = wf.load_prices(engine, args.hub)
    print(f"Loaded {len(df):,} hours for {args.hub}")

    if args.if_stale_days is not None and args.out.exists():
        # Cron calls this every day; retraining daily is wasted work, and never
        # retraining is how the served model silently falls behind the market.
        try:
            existing = joblib.load(args.out)
            cutoff = pd.Timestamp(existing["data_cutoff"])
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize("UTC")
            lag_days = (df["timestamp_utc"].max() - cutoff).total_seconds() / 86400
            if lag_days < args.if_stale_days:
                print(f"Artifact {existing['version']} is {lag_days:,.1f} days "
                      f"behind the data (threshold {args.if_stale_days:,.0f}); "
                      f"nothing to do.")
                return 0
            print(f"Artifact {existing['version']} is {lag_days:,.1f} days "
                  f"behind; retraining.")
        except Exception as exc:
            print(f"Could not read the existing artifact ({exc}); retraining.")

    artifact = train(df, args.hub, args.holdout_months)
    joblib.dump(artifact, args.out)

    size_kb = args.out.stat().st_size / 1024
    print(f"\nSaved {args.out} ({size_kb:,.0f} KB)")
    print(f"  version      {artifact['version']}")
    print(f"  hub          {artifact['hub']}")
    print(f"  data cutoff  {artifact['data_cutoff']}")
    print(f"  rows         {artifact['training_rows']:,}")
    print("  calibration")
    for name, c in artifact["calibration"].items():
        print(f"    {name}  nominal {c['nominal']*100:>3.0f}%  "
              f"observed {c['observed']*100:>5.1f}%  {c['error_pp']:+.1f}pp")

    worst = max(abs(c["error_pp"]) for c in artifact["calibration"].values())
    if worst >= 5:
        print(f"\nWARNING: miscalibrated by up to {worst:.1f}pp. Expected values "
              f"built on these quantiles will be correspondingly wrong.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (TrainingError, wf.WalkForwardError) as exc:
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        sys.exit(1)
