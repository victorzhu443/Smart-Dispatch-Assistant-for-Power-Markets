"""Walk-forward evaluation: does the model beat the market across all conditions?

The single-split result in phase_3_4 (LSTM, $11.70 RMSE, beating day-ahead's
$15.62) was measured on one held-out window that happened to be calm -- it
peaks at $214/MWh and holds 3 hours above $200, against 74 across the full two
years. A model can look excellent on a quiet stretch and fall apart in the
hours that actually decide a peaker's year.

This retrains repeatedly, walking forward through history the way the system
would run in production:

    fold 1   train 2024-01..2024-06   predict 2024-07
    fold 2   train 2024-01..2024-07   predict 2024-08
    ...

Every fold trains only on hours before the ones it predicts, so no fold can
see its own future. Baselines are scored on exactly the same predicted hours,
because comparing errors computed over different populations is meaningless
however similar the units look.

Feature timing is the part to check when reading this. Every feature is
knowable before the target hour:

    lag_*, roll_*   past real-time prices, shifted strictly backwards
    dam             the day-ahead clearing price for the target hour, which
                    clears around 13:30 the day BEFORE delivery, so it is
                    known 10-34 hours ahead. Legitimate, and the single
                    strongest predictor available.
    hour, dow       calendar facts

Usage:
    python forecasting-model/walk_forward.py
    python forecasting-model/walk_forward.py --hub HB_WEST --initial-months 6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

from common.db import setup_database_connection

MARKET_TZ = "America/Chicago"
SEED = 42
SPIKE_THRESHOLD = 200.0

# Lags in hours. 1-3 capture short persistence, 24 the daily cycle, 168 weekly.
LAGS = [1, 2, 3, 24, 168]
ROLL_WINDOWS = [24, 168]


class WalkForwardError(RuntimeError):
    pass


def load_prices(engine, hub: str) -> pd.DataFrame:
    rt = pd.read_sql(
        text("SELECT timestamp_utc, price FROM market_data_hourly "
             "WHERE settlement_point = :hub ORDER BY timestamp_utc"),
        engine, params={"hub": hub},
    )
    if rt.empty:
        raise WalkForwardError(
            f"No prices for {hub!r}. Run "
            f"'python -m data_ingestion.ingest_ercot_history' first."
        )
    dam = pd.read_sql(
        text("SELECT timestamp_utc, price AS dam FROM dam_prices_hourly "
             "WHERE settlement_point = :hub ORDER BY timestamp_utc"),
        engine, params={"hub": hub},
    )
    for frame in (rt, dam):
        frame["timestamp_utc"] = pd.to_datetime(
            frame["timestamp_utc"], utc=True, format="mixed"
        )
    df = rt.merge(dam, on="timestamp_utc", how="left")
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every column here is knowable before the hour it describes."""
    out = df.copy()
    local = out["timestamp_utc"].dt.tz_convert(MARKET_TZ)

    for lag in LAGS:
        out[f"lag_{lag}"] = out["price"].shift(lag)

    for window in ROLL_WINDOWS:
        # shift(1) first: a rolling window ending at t would include t itself.
        past = out["price"].shift(1)
        out[f"roll_mean_{window}"] = past.rolling(window).mean()
        out[f"roll_std_{window}"] = past.rolling(window).std()
        out[f"roll_max_{window}"] = past.rolling(window).max()

    out["dam_minus_lag1"] = out["dam"] - out["lag_1"]
    out["hour"] = local.dt.hour
    out["dow"] = local.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["month"] = local.dt.month

    return out


FEATURE_COLUMNS = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_{stat}_{w}" for w in ROLL_WINDOWS for stat in ("mean", "std", "max")]
    + ["dam", "dam_minus_lag1", "hour", "dow", "is_weekend", "month"]
)


def month_starts(timestamps: pd.Series) -> list[pd.Timestamp]:
    """First instant of each month present, as UTC-aware timestamps."""
    months = timestamps.dt.tz_convert("UTC").dt.to_period("M").unique()
    return [p.to_timestamp().tz_localize("UTC") for p in sorted(months)]


def walk_forward(df: pd.DataFrame, initial_months: int) -> pd.DataFrame:
    """Retrain monthly on an expanding window; collect out-of-sample predictions."""
    feat = build_features(df).dropna(subset=FEATURE_COLUMNS + ["price"])
    if feat.empty:
        raise WalkForwardError("No rows survive feature construction")

    boundaries = month_starts(feat["timestamp_utc"])
    if len(boundaries) <= initial_months:
        raise WalkForwardError(
            f"Need more than {initial_months} months of data, have {len(boundaries)}"
        )

    collected = []
    for i in range(initial_months, len(boundaries)):
        cut = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else None

        is_train = feat["timestamp_utc"] < cut
        is_test = (feat["timestamp_utc"] >= cut) & (
            (feat["timestamp_utc"] < end) if end is not None else True
        )
        if is_test.sum() == 0 or is_train.sum() < 500:
            continue

        X_train = feat.loc[is_train, FEATURE_COLUMNS].to_numpy()
        y_train = feat.loc[is_train, "price"].to_numpy()
        X_test = feat.loc[is_test, FEATURE_COLUMNS].to_numpy()

        # Rung 1: linear, on scaled inputs.
        scaler = StandardScaler().fit(X_train)
        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(scaler.transform(X_train), y_train)

        # Rung 2: gradient-boosted trees. Same family as LightGBM, but ships
        # with scikit-learn and needs no OpenMP runtime.
        gbm = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=None,
            min_samples_leaf=40, l2_regularization=1.0, random_state=SEED,
        )
        gbm.fit(X_train, y_train)

        fold = feat.loc[is_test, ["timestamp_utc", "price", "dam"]].copy()
        fold["ridge"] = ridge.predict(scaler.transform(X_test))
        fold["gbm"] = gbm.predict(X_test)
        fold["persistence"] = feat.loc[is_test, "lag_1"].to_numpy()
        fold["seasonal_24h"] = feat.loc[is_test, "lag_24"].to_numpy()
        fold["fold"] = i - initial_months + 1
        fold["train_hours"] = int(is_train.sum())
        collected.append(fold)

        print(f"  fold {i - initial_months + 1:>2}  train {is_train.sum():>6,}h  "
              f"predict {is_test.sum():>4,}h  from {cut:%Y-%m}")

    if not collected:
        raise WalkForwardError("No usable folds")
    return pd.concat(collected, ignore_index=True)


def rmse(actual, predicted) -> float:
    mask = pd.notna(actual) & pd.notna(predicted)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(((predicted[mask] - actual[mask]) ** 2).mean()))


def report(results: pd.DataFrame, hub: str) -> None:
    predictors = ["gbm", "ridge", "dam", "persistence", "seasonal_24h"]
    labels = {
        "gbm": "gradient boosting", "ridge": "ridge regression",
        "dam": "day-ahead (market)", "persistence": "persistence",
        "seasonal_24h": "same hour yesterday",
    }
    actual = results["price"]

    local_hour = results["timestamp_utc"].dt.tz_convert(MARKET_TZ).dt.hour
    peak = local_hour.between(14, 19)
    spike = actual >= SPIKE_THRESHOLD

    print(f"\nWalk-forward — {hub}")
    print(f"{len(results):,} out-of-sample hours across {results['fold'].nunique()} "
          f"monthly folds, {results['timestamp_utc'].min():%Y-%m-%d} to "
          f"{results['timestamp_utc'].max():%Y-%m-%d}")
    print(f"{int(spike.sum())} hours at or above ${SPIKE_THRESHOLD:,.0f}/MWh\n")

    head = f"{'predictor':<22}{'RMSE':>10}{'MAE':>10}{'peak RMSE':>12}{'spike RMSE':>13}"
    print(head)
    print("-" * len(head))

    scored = []
    for name in predictors:
        pred = results[name]
        mask = pd.notna(actual) & pd.notna(pred)
        overall = rmse(actual, pred)
        mae = float((pred[mask] - actual[mask]).abs().mean())
        scored.append((overall, name, mae,
                       rmse(actual[peak], pred[peak]),
                       rmse(actual[spike], pred[spike])))

    for overall, name, mae, peak_rmse, spike_rmse in sorted(scored):
        print(f"{labels[name]:<22}{'$'+format(overall, ',.2f'):>10}"
              f"{'$'+format(mae, ',.2f'):>10}"
              f"{'$'+format(peak_rmse, ',.2f'):>12}"
              f"{'$'+format(spike_rmse, ',.2f'):>13}")

    best = min(scored)
    dam_score = next(s for s in scored if s[1] == "dam")[0]
    print(f"\nBest overall: {labels[best[1]]} at ${best[0]:,.2f}")
    if best[1] != "dam":
        delta = (dam_score - best[0]) / dam_score * 100
        print(f"vs day-ahead ${dam_score:,.2f} — {delta:+.1f}% better across all "
              f"conditions, not one window.")
    else:
        print("The market's own forward price is still the best predictor here.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", default="HB_HOUSTON")
    parser.add_argument("--initial-months", type=int, default=6)
    args = parser.parse_args()

    engine = setup_database_connection()
    df = load_prices(engine, args.hub)
    print(f"Loaded {len(df):,} hours for {args.hub}; walking forward...")
    results = walk_forward(df, args.initial_months)
    report(results, args.hub)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except WalkForwardError as exc:
        print(f"\nWalk-forward failed: {exc}", file=sys.stderr)
        sys.exit(1)
