"""Tests for the evaluation harness.

The harness decides which model is better, so a bug here is worse than a bug
in any model: it would silently endorse the wrong one. The most important
tests are the ones asserting that no baseline can see the hour it predicts.

No database or network: every test builds its own frame.

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


backtest = _load("forecasting-model/backtest.py", "backtest")


def _hours(n, start="2025-06-01 05:00", rt=None, dam=None):
    """n consecutive UTC hours with optional price series."""
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp_utc": idx,
        "rt_price": np.arange(n, dtype=float) if rt is None else rt,
        "dam_price": np.arange(n, dtype=float) if dam is None else dam,
    })
    local_hour = df["timestamp_utc"].dt.tz_convert(backtest.MARKET_TZ).dt.hour
    df["local_hour"] = local_hour
    df["regime"] = "unclassified"
    for name, hours in backtest.REGIMES.items():
        df.loc[local_hour.isin(list(hours)), "regime"] = name
    return df


class TestBaselinesCannotSeeTheFuture:
    """Each baseline must use only hours strictly before its target."""

    def test_persistence_uses_the_previous_hour(self):
        df = _hours(10)
        pred = backtest.build_baselines(df)["persistence"]

        assert pd.isna(pred.iloc[0]), "first hour has no predecessor to use"
        # Prices are 0,1,2,... so the prediction for hour i must be i-1.
        np.testing.assert_array_equal(pred.iloc[1:].to_numpy(),
                                      df["rt_price"].iloc[:-1].to_numpy())

    def test_seasonal_24h_uses_the_same_hour_yesterday(self):
        df = _hours(50)
        pred = backtest.build_baselines(df)["seasonal_24h"]

        assert pred.iloc[:24].isna().all(), "first day cannot be predicted"
        assert pred.iloc[24] == df["rt_price"].iloc[0]
        assert pred.iloc[49] == df["rt_price"].iloc[25]

    def test_seasonal_168h_uses_the_same_hour_last_week(self):
        df = _hours(200)
        pred = backtest.build_baselines(df)["seasonal_168h"]

        assert pred.iloc[:168].isna().all()
        assert pred.iloc[168] == df["rt_price"].iloc[0]

    def test_no_baseline_ever_equals_its_own_target(self):
        """A baseline returning the actual value would be leakage."""
        rt = np.random.default_rng(0).normal(30, 10, 300)
        df = _hours(300, rt=rt, dam=rt + 5)  # dam deliberately offset

        for name, pred in backtest.build_baselines(df).items():
            overlap = pred.notna() & df["rt_price"].notna()
            identical = (pred[overlap] == df["rt_price"][overlap]).all()
            assert not identical, f"{name} reproduces the target exactly"


class TestScoring:
    def test_rmse_and_mae_on_a_known_case(self):
        actual = pd.Series([10.0, 20.0, 30.0])
        predicted = pd.Series([12.0, 18.0, 33.0])  # errors +2, -2, +3

        result = backtest.score(actual, predicted)

        assert result["mae"] == pytest.approx(7 / 3)
        assert result["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
        assert result["bias"] == pytest.approx(1.0)
        assert result["n"] == 3

    def test_rmse_punishes_one_big_miss_more_than_many_small_ones(self):
        """Why RMSE and MAE can rank predictors differently."""
        actual = pd.Series([10.0] * 10)
        many_small = pd.Series([12.0] * 10)          # 10 errors of 2
        one_huge = pd.Series([10.0] * 9 + [30.0])    # 1 error of 20

        small = backtest.score(actual, many_small)
        huge = backtest.score(actual, one_huge)

        assert small["mae"] == pytest.approx(2.0)
        assert huge["mae"] == pytest.approx(2.0)         # identical MAE
        assert huge["rmse"] > small["rmse"]              # different RMSE

    def test_ignores_hours_where_either_side_is_missing(self):
        actual = pd.Series([10.0, np.nan, 30.0])
        predicted = pd.Series([10.0, 20.0, np.nan])

        result = backtest.score(actual, predicted)

        assert result["n"] == 1
        assert result["rmse"] == pytest.approx(0.0)

    def test_returns_nan_rather_than_raising_on_no_overlap(self):
        result = backtest.score(pd.Series([np.nan]), pd.Series([1.0]))
        assert result["n"] == 0
        assert np.isnan(result["rmse"])


class TestSpikeDetection:
    def test_counts_hits_misses_and_false_alarms(self):
        actual = pd.Series([300.0, 300.0, 50.0, 50.0])
        predicted = pd.Series([300.0, 50.0, 300.0, 50.0])

        s = backtest.spike_recall(actual, predicted, threshold=200.0)

        assert s["spikes"] == 2
        assert s["caught"] == 1
        assert s["missed"] == 1
        assert s["false_alarms"] == 1
        assert s["recall"] == pytest.approx(0.5)
        assert s["precision"] == pytest.approx(0.5)

    def test_threshold_is_inclusive(self):
        s = backtest.spike_recall(
            pd.Series([200.0]), pd.Series([200.0]), threshold=200.0
        )
        assert s["spikes"] == 1 and s["caught"] == 1

    def test_recall_is_nan_when_there_are_no_spikes(self):
        s = backtest.spike_recall(
            pd.Series([10.0, 20.0]), pd.Series([10.0, 20.0]), threshold=200.0
        )
        assert s["spikes"] == 0
        assert np.isnan(s["recall"])


class TestRegimes:
    def test_regimes_partition_the_whole_day(self):
        covered = sorted(h for hours in backtest.REGIMES.values() for h in hours)
        assert covered == list(range(24)), "regimes must tile 0-23 exactly once"

    def test_peak_covers_the_afternoon(self):
        assert 15 in backtest.REGIMES["peak"]
        assert 3 in backtest.REGIMES["overnight"]


class TestResultsTable:
    def test_ranks_predictors_best_first(self):
        df = _hours(300, rt=np.full(300, 50.0))
        predictions = {
            "perfect": pd.Series(np.full(300, 50.0)),
            "poor": pd.Series(np.full(300, 150.0)),
        }

        results = backtest.run(df, predictions, spike_threshold=200.0)

        assert results.iloc[0]["predictor"] == "perfect"
        assert results.iloc[0]["rmse"] < results.iloc[1]["rmse"]
