"""Tests for the dispatch decision layer.

The profit simulation is the number the whole project is judged on, so its
arithmetic is worth pinning down: energy margin only accrues in hours the unit
runs, and a start cost is charged on each off-to-on transition, not per hour
and not once overall.

Run with:  pytest tests/ -v
"""
import importlib

import numpy as np
import pandas as pd
import pytest


dispatch = importlib.import_module('forecasting_model.dispatch')

ECON = {"marginal_cost": 45.0, "start_cost": 5000.0, "capacity_mw": 100.0}


class TestProfitArithmetic:
    def test_running_one_profitable_hour(self):
        # (95 - 45) * 100 MW = 5,000 energy margin, minus one 5,000 start = 0
        out = dispatch.simulate(np.array([True]), np.array([95.0]), **ECON)

        assert out["energy_margin"] == pytest.approx(5000.0)
        assert out["starts"] == 1
        assert out["profit"] == pytest.approx(0.0)

    def test_not_running_earns_and_costs_nothing(self):
        out = dispatch.simulate(np.array([False, False]),
                                np.array([500.0, 500.0]), **ECON)

        assert out["profit"] == 0.0
        assert out["starts"] == 0
        assert out["hours_run"] == 0

    def test_running_below_marginal_cost_loses_money(self):
        out = dispatch.simulate(np.array([True]), np.array([20.0]), **ECON)

        # (20 - 45) * 100 = -2,500, plus a 5,000 start
        assert out["profit"] == pytest.approx(-7500.0)


class TestStartCosts:
    def test_consecutive_hours_are_one_start(self):
        decisions = np.array([True, True, True])
        out = dispatch.simulate(decisions, np.full(3, 145.0), **ECON)

        assert out["starts"] == 1, "a continuous run must not re-start each hour"
        assert out["hours_run"] == 3

    def test_each_restart_is_charged_again(self):
        decisions = np.array([True, False, True, False, True])
        out = dispatch.simulate(decisions, np.full(5, 145.0), **ECON)

        assert out["starts"] == 3

    def test_starting_in_the_first_hour_counts(self):
        """The unit begins the period off, so hour 0 running is a start."""
        out = dispatch.simulate(np.array([True, True]), np.full(2, 145.0), **ECON)
        assert out["starts"] == 1

    def test_intermittent_dispatch_can_be_worse_than_not_running(self):
        """Start costs are what make marginal hours genuinely risky."""
        prices = np.array([46.0, 20.0, 46.0, 20.0, 46.0])
        flapping = np.array([True, False, True, False, True])

        out = dispatch.simulate(flapping, prices, **ECON)

        assert out["starts"] == 3
        assert out["profit"] < 0


class TestCalibration:
    def test_perfectly_calibrated_quantiles_report_no_error(self):
        rng = np.random.default_rng(0)
        price = pd.Series(rng.normal(50, 10, 20000))
        results = pd.DataFrame({
            "price": price,
            "p10": np.quantile(price, 0.10),
            "p50": np.quantile(price, 0.50),
            "p90": np.quantile(price, 0.90),
        })

        cal = dispatch.calibration(results)

        assert cal["error_pp"].abs().max() < 1.0

    def test_detects_an_overconfident_interval(self):
        """A P90 that only covers 60% must be reported as miscalibrated."""
        price = pd.Series(np.arange(100, dtype=float))
        results = pd.DataFrame({
            "price": price, "p10": 5.0, "p50": 50.0, "p90": 60.0,
        })

        cal = dispatch.calibration(results)
        p90 = cal[cal["quantile"] == "p90"].iloc[0]

        assert p90["observed"] == pytest.approx(0.61, abs=0.02)
        assert p90["error_pp"] < -25


class TestExpectedMargin:
    def test_is_zero_when_every_scenario_is_below_cost(self):
        results = pd.DataFrame({"p10": [10.0], "p50": [20.0], "p90": [30.0]})

        assert dispatch.expected_margin(results, 45.0).iloc[0] == 0.0

    def test_upside_tail_contributes_even_when_the_median_is_below_cost(self):
        """The whole point of using the distribution rather than the middle."""
        results = pd.DataFrame({"p10": [10.0], "p50": [40.0], "p90": [345.0]})

        margin = dispatch.expected_margin(results, 45.0).iloc[0]

        assert margin == pytest.approx(300.0 / 3)
        assert margin > 0

    def test_never_goes_negative(self):
        results = pd.DataFrame({"p10": [-200.0], "p50": [-100.0], "p90": [-50.0]})

        assert dispatch.expected_margin(results, 45.0).iloc[0] == 0.0


class TestStrategies:
    def test_perfect_foresight_runs_exactly_the_profitable_hours(self):
        results = pd.DataFrame({
            "price": [10.0, 100.0, 44.0, 46.0],
            "p10": 0.0, "p50": 0.0, "p90": 0.0, "dam": 0.0,
        })

        strategies = dispatch.build_strategies(results, marginal_cost=45.0)

        np.testing.assert_array_equal(
            strategies["perfect foresight"], np.array([False, True, False, True])
        )

    def test_always_and_never_run_are_constant(self):
        results = pd.DataFrame({
            "price": [10.0, 100.0], "p10": 0.0, "p50": 0.0, "p90": 0.0, "dam": 0.0,
        })
        strategies = dispatch.build_strategies(results, marginal_cost=45.0)

        assert strategies["always run"].all()
        assert not strategies["never run"].any()
