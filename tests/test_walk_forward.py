"""Tests for walk-forward evaluation.

The feature builder is where leakage would hide: a rolling window that
includes its own target, or a lag shifted the wrong direction, produces a
model that looks excellent and cannot work in production. These tests assert
the timing directly rather than trusting the shifts by eye.

Run with:  pytest tests/ -v
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(relative_path, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wf = _load("forecasting-model/walk_forward.py", "walk_forward")


def _frame(n=400):
    """Prices that encode their own position, so leakage is visible."""
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp_utc": idx,
        "price": np.arange(n, dtype=float),
        "dam": np.arange(n, dtype=float) + 1000.0,   # offset so it is distinguishable
    })


class TestFeatureTiming:
    def test_lags_look_strictly_backwards(self):
        feat = wf.build_features(_frame())

        for lag in wf.LAGS:
            column = feat[f"lag_{lag}"]
            valid = column.notna()
            # price[i] == i, so lag_k at row i must equal i - k.
            expected = feat.loc[valid, "price"] - lag
            np.testing.assert_array_equal(column[valid].to_numpy(),
                                          expected.to_numpy())

    def test_rolling_windows_exclude_the_target_hour(self):
        """A window ending at t that includes t leaks the answer."""
        feat = wf.build_features(_frame())

        for window in wf.ROLL_WINDOWS:
            column = feat[f"roll_max_{window}"]
            valid = column.notna()
            # Prices increase, so the max of the window ending strictly before
            # row i is price[i-1] == i-1. If it leaked, it would equal i.
            assert (column[valid] < feat.loc[valid, "price"]).all(), (
                f"roll_max_{window} includes the target hour"
            )

    def test_rolling_mean_never_reaches_the_current_price(self):
        feat = wf.build_features(_frame())
        for window in wf.ROLL_WINDOWS:
            column = feat[f"roll_mean_{window}"]
            valid = column.notna()
            assert (column[valid] < feat.loc[valid, "price"]).all()

    def test_day_ahead_is_used_for_the_target_hour(self):
        """DAM clears the day before delivery, so using it at t is legitimate."""
        feat = wf.build_features(_frame())
        np.testing.assert_array_equal(feat["dam"].to_numpy(),
                                      feat["price"].to_numpy() + 1000.0)

    def test_every_declared_feature_exists(self):
        feat = wf.build_features(_frame())
        missing = [c for c in wf.FEATURE_COLUMNS if c not in feat.columns]
        assert not missing, f"declared but not built: {missing}"

    def test_calendar_features_use_local_time(self):
        feat = wf.build_features(_frame())
        expected = (
            feat["timestamp_utc"].dt.tz_convert(wf.MARKET_TZ).dt.hour.to_numpy()
        )
        np.testing.assert_array_equal(feat["hour"].to_numpy(), expected)


class TestMonthBoundaries:
    def test_returns_utc_aware_month_starts(self):
        idx = pd.Series(pd.date_range("2024-01-15", periods=90, freq="1D", tz="UTC"))
        starts = wf.month_starts(idx)

        assert len(starts) == 4                       # Jan, Feb, Mar, Apr
        assert all(t.tzinfo is not None for t in starts)
        assert all(t.day == 1 and t.hour == 0 for t in starts)
        assert starts == sorted(starts)


class TestWalkForward:
    def test_folds_never_train_on_their_own_test_hours(self):
        results = wf.walk_forward(_frame(24 * 200), initial_months=2)

        # Every fold's predicted hours must be disjoint from every other fold's.
        counts = results.groupby("timestamp_utc").size()
        assert (counts == 1).all(), "an hour was predicted by more than one fold"

    def test_training_set_grows_each_fold(self):
        results = wf.walk_forward(_frame(24 * 200), initial_months=2)
        sizes = results.groupby("fold")["train_hours"].first()

        assert (sizes.diff().dropna() > 0).all(), "expanding window did not expand"

    def test_produces_predictions_for_every_predictor(self):
        results = wf.walk_forward(_frame(24 * 200), initial_months=2)

        for column in ("gbm", "ridge", "dam", "persistence", "seasonal_24h"):
            assert column in results.columns
            assert results[column].notna().any()

    def test_refuses_when_history_is_too_short(self):
        with pytest.raises(wf.WalkForwardError):
            wf.walk_forward(_frame(24 * 20), initial_months=6)


class TestScoring:
    def test_rmse_matches_a_hand_computed_value(self):
        actual = pd.Series([10.0, 20.0, 30.0])
        predicted = pd.Series([12.0, 18.0, 33.0])

        assert wf.rmse(actual, predicted) == pytest.approx(
            np.sqrt((4 + 4 + 9) / 3)
        )

    def test_rmse_ignores_missing_pairs(self):
        actual = pd.Series([10.0, np.nan])
        predicted = pd.Series([10.0, 99.0])

        assert wf.rmse(actual, predicted) == pytest.approx(0.0)
