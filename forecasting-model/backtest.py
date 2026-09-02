"""Evaluation harness and baselines for real-time price forecasting.

Built before any model, deliberately. Without a scoreboard there is no way to
tell a real improvement from noise, and tuning against noise is how the
previous version of this project ended up with a 57,441-parameter LSTM losing
to a constant.

Every predictor here uses only information available before the hour it
predicts, so the comparison is honest:

    persistence      the previous hour's real-time price
    seasonal_24h     the same hour yesterday
    seasonal_168h    the same hour last week
    day_ahead        the day-ahead clearing price for that hour

The last one is the bar that matters. Day-ahead is the market's own published
forecast of real-time, set by participants with money at stake. Beating naive
baselines shows a model learned something; beating day-ahead would mean it
knows something the market does not.

Results are broken out by regime because an average over 17,544 hours hides
the only hours that pay. A predictor can post an excellent overall RMSE by
predicting the mean forever and missing every scarcity event.

Usage:
    python forecasting-model/backtest.py
    python forecasting-model/backtest.py --hub HB_NORTH --spike-threshold 100
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

MARKET_TZ = "America/Chicago"

# Hours are labelled by local Central time, since the daily demand cycle that
# drives price is a local-clock phenomenon.
REGIMES = {
    "overnight": range(0, 6),     # 00:00-05:59, minimum demand
    "morning": range(6, 14),      # 06:00-13:59, ramp
    "peak": range(14, 20),        # 14:00-19:59, afternoon peak
    "evening": range(20, 24),     # 20:00-23:59, decline
}

# Above this, an hour is a scarcity event worth catching. Roughly the point
# where a peaker's margin becomes large relative to its start cost.
DEFAULT_SPIKE_THRESHOLD = 200.0


class BacktestError(RuntimeError):
    """The backtest cannot produce a trustworthy comparison."""


def setup_database_connection():
    try:
        pg = (
            f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'password')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DATABASE', 'smart_dispatch')}"
        )
        engine = create_engine(pg)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception:
        return create_engine("sqlite:///market_data.db")


def load_prices(engine, hub: str) -> pd.DataFrame:
    """Load aligned real-time and day-ahead hourly prices for one hub."""
    try:
        rt = pd.read_sql(
            text(
                "SELECT timestamp_utc, price FROM market_data_hourly "
                "WHERE settlement_point = :hub ORDER BY timestamp_utc"
            ),
            engine,
            params={"hub": hub},
        )
    except Exception as exc:
        raise BacktestError(
            f"Could not read market_data_hourly: {exc}. Run "
            f"'python data-ingestion/ingest_ercot_history.py' first."
        ) from exc

    if rt.empty:
        raise BacktestError(
            f"No real-time prices for {hub!r}. Ingest it first, or pick another "
            f"hub with --hub."
        )

    dam = pd.read_sql(
        text(
            "SELECT timestamp_utc, price AS dam_price FROM dam_prices_hourly "
            "WHERE settlement_point = :hub ORDER BY timestamp_utc"
        ),
        engine,
        params={"hub": hub},
    )

    for frame in (rt, dam):
        frame["timestamp_utc"] = pd.to_datetime(
            frame["timestamp_utc"], utc=True, format="mixed"
        )

    df = rt.rename(columns={"price": "rt_price"}).merge(
        dam, on="timestamp_utc", how="left"
    )
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Regime labels come from local time, not UTC.
    local_hour = df["timestamp_utc"].dt.tz_convert(MARKET_TZ).dt.hour
    df["local_hour"] = local_hour
    df["regime"] = "unclassified"
    for name, hours in REGIMES.items():
        df.loc[local_hour.isin(list(hours)), "regime"] = name

    return df


def build_baselines(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Predictions from each baseline, aligned to df's index.

    Every one uses only information available before the target hour.
    """
    return {
        "persistence": df["rt_price"].shift(1),
        "seasonal_24h": df["rt_price"].shift(24),
        "seasonal_168h": df["rt_price"].shift(168),
        "day_ahead": df["dam_price"],
    }


def score(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """RMSE, MAE and bias over the hours where both are present."""
    mask = actual.notna() & predicted.notna()
    if not mask.any():
        return {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "n": 0}

    error = predicted[mask] - actual[mask]
    return {
        "rmse": float(np.sqrt((error**2).mean())),
        "mae": float(error.abs().mean()),
        "bias": float(error.mean()),
        "n": int(mask.sum()),
    }


def spike_recall(
    actual: pd.Series, predicted: pd.Series, threshold: float
) -> dict[str, float]:
    """How many scarcity hours a predictor would have flagged in advance.

    Treated as a detection problem, because that is what the dispatch decision
    actually is: the operator needs to know whether to commit, not the price to
    the cent.
    """
    mask = actual.notna() & predicted.notna()
    actual_spike = actual[mask] >= threshold
    predicted_spike = predicted[mask] >= threshold

    true_positive = int((actual_spike & predicted_spike).sum())
    false_negative = int((actual_spike & ~predicted_spike).sum())
    false_positive = int((~actual_spike & predicted_spike).sum())

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else np.nan
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else np.nan
    )
    return {
        "spikes": int(actual_spike.sum()),
        "caught": true_positive,
        "missed": false_negative,
        "false_alarms": false_positive,
        "recall": recall,
        "precision": precision,
    }


def run(df: pd.DataFrame, predictions: dict[str, pd.Series],
        spike_threshold: float) -> pd.DataFrame:
    """Score every predictor overall and per regime."""
    rows = []
    for name, predicted in predictions.items():
        overall = score(df["rt_price"], predicted)
        row = {"predictor": name, "rmse": overall["rmse"], "mae": overall["mae"],
               "bias": overall["bias"], "n": overall["n"]}
        for regime in REGIMES:
            in_regime = df["regime"] == regime
            row[f"rmse_{regime}"] = score(
                df.loc[in_regime, "rt_price"], predicted[in_regime]
            )["rmse"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def _fmt(value, width=9, money=True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—".rjust(width)
    return (f"${value:,.2f}" if money else f"{value:,.0f}").rjust(width)


def report(df: pd.DataFrame, results: pd.DataFrame,
           predictions: dict[str, pd.Series], hub: str,
           spike_threshold: float) -> None:
    start = df["timestamp_utc"].min().tz_convert(MARKET_TZ)
    end = df["timestamp_utc"].max().tz_convert(MARKET_TZ)

    print(f"\nBacktest — {hub}")
    print(f"{len(df):,} hours, {start:%Y-%m-%d} to {end:%Y-%m-%d} (local)")
    print(f"Real-time price: mean ${df['rt_price'].mean():.2f}, "
          f"median ${df['rt_price'].median():.2f}, "
          f"max ${df['rt_price'].max():,.2f}, "
          f"negative in {(df['rt_price'] < 0).mean() * 100:.1f}% of hours")

    print("\nAccuracy (lower RMSE is better)")
    header = (f"{'predictor':<15}{'RMSE':>10}{'MAE':>10}{'bias':>10}"
              f"{'overnight':>11}{'morning':>10}{'peak':>10}{'evening':>10}")
    print(header)
    print("-" * len(header))
    for _, r in results.iterrows():
        print(f"{r['predictor']:<15}"
              f"{_fmt(r['rmse'], 10)}{_fmt(r['mae'], 10)}{_fmt(r['bias'], 10)}"
              f"{_fmt(r['rmse_overnight'], 11)}{_fmt(r['rmse_morning'], 10)}"
              f"{_fmt(r['rmse_peak'], 10)}{_fmt(r['rmse_evening'], 10)}")

    print(f"\nScarcity detection (hours at or above ${spike_threshold:,.0f}/MWh)")
    header = (f"{'predictor':<15}{'spikes':>8}{'caught':>8}{'missed':>8}"
              f"{'false alarms':>14}{'recall':>9}{'precision':>11}")
    print(header)
    print("-" * len(header))
    for name, predicted in predictions.items():
        s = spike_recall(df["rt_price"], predicted, spike_threshold)
        recall = f"{s['recall'] * 100:.1f}%" if not np.isnan(s["recall"]) else "—"
        prec = f"{s['precision'] * 100:.1f}%" if not np.isnan(s["precision"]) else "—"
        print(f"{name:<15}{s['spikes']:>8,}{s['caught']:>8,}{s['missed']:>8,}"
              f"{s['false_alarms']:>14,}{recall:>9}{prec:>11}")

    best = results.iloc[0]
    print(f"\nBar to beat: {best['predictor']} at ${best['rmse']:,.2f} RMSE.")
    print("A model is only worth deploying if it beats this on the same hours.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", default="HB_HOUSTON")
    parser.add_argument("--spike-threshold", type=float,
                        default=DEFAULT_SPIKE_THRESHOLD)
    args = parser.parse_args()

    engine = setup_database_connection()
    df = load_prices(engine, args.hub)
    predictions = build_baselines(df)
    results = run(df, predictions, args.spike_threshold)
    report(df, results, predictions, args.hub, args.spike_threshold)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BacktestError as exc:
        print(f"\nBacktest failed: {exc}", file=sys.stderr)
        sys.exit(1)
