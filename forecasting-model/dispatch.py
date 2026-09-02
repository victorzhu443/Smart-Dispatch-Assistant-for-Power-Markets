"""Quantile forecasts turned into dispatch decisions, scored in dollars.

Walk-forward evaluation showed that on scarcity hours every point forecast --
including the market's own day-ahead price -- is wrong by roughly $600-700.
Those are the hours that decide whether a peaker earns its year. A point
forecast is the wrong instrument for that: the operator does not need the price
to the cent, they need to know whether committing is worth the risk.

So this predicts a distribution instead of a number, and converts it into the
decision the plant actually faces:

    run this hour if  E[max(price - marginal_cost, 0)]  exceeds the cost of
    running, allowing for the start cost when the unit is currently off

and scores strategies in dollars rather than RMSE, against a perfect-foresight
upper bound.

Quantiles come from three gradient-boosting models fit with pinball loss at
P10 / P50 / P90, retrained monthly on an expanding window exactly as in
walk_forward.py. Calibration is checked rather than assumed: if only 70% of
actual prices fall below the P90, the "90%" is a lie and every expected value
built on it is wrong.

Economics default to a mid-merit gas peaker. Override at the command line.

Usage:
    python forecasting-model/dispatch.py
    python forecasting-model/dispatch.py --marginal-cost 55 --start-cost 8000
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(relative_path, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wf = _load("forecasting-model/walk_forward.py", "walk_forward")

QUANTILES = (0.10, 0.50, 0.90)
SEED = 42

# A mid-merit gas peaker. Marginal cost is fuel plus variable O&M; start cost
# is the one-off cost of bringing the unit up from cold.
DEFAULT_MARGINAL_COST = 45.0   # $/MWh
DEFAULT_START_COST = 5000.0    # $ per start
DEFAULT_CAPACITY_MW = 100.0


class DispatchError(RuntimeError):
    pass


def fit_quantile_models(X_train, y_train):
    models = {}
    for q in QUANTILES:
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=q,
            max_iter=300, learning_rate=0.06, min_samples_leaf=40,
            l2_regularization=1.0, random_state=SEED,
        )
        model.fit(X_train, y_train)
        models[q] = model
    return models


def walk_forward_quantiles(df: pd.DataFrame, initial_months: int) -> pd.DataFrame:
    """Out-of-sample P10/P50/P90 for every hour, retrained monthly."""
    feat = wf.build_features(df).dropna(subset=wf.FEATURE_COLUMNS + ["price"])
    boundaries = wf.month_starts(feat["timestamp_utc"])
    if len(boundaries) <= initial_months:
        raise DispatchError(
            f"Need more than {initial_months} months, have {len(boundaries)}"
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

        X_train = feat.loc[is_train, wf.FEATURE_COLUMNS].to_numpy()
        y_train = feat.loc[is_train, "price"].to_numpy()
        X_test = feat.loc[is_test, wf.FEATURE_COLUMNS].to_numpy()

        models = fit_quantile_models(X_train, y_train)

        fold = feat.loc[is_test, ["timestamp_utc", "price", "dam"]].copy()
        for q, model in models.items():
            fold[f"p{int(q * 100)}"] = model.predict(X_test)
        collected.append(fold)
        print(f"  fold {i - initial_months + 1:>2}  train {is_train.sum():>6,}h  "
              f"predict {is_test.sum():>4,}h  from {cut:%Y-%m}")

    if not collected:
        raise DispatchError("No usable folds")

    out = pd.concat(collected, ignore_index=True)
    # Quantiles are fit independently and can cross on hard hours; sorting each
    # row restores monotonicity without changing the marginal distributions.
    q_cols = [f"p{int(q * 100)}" for q in QUANTILES]
    out[q_cols] = np.sort(out[q_cols].to_numpy(), axis=1)
    return out


def calibration(results: pd.DataFrame) -> pd.DataFrame:
    """Do the quantiles mean what they claim?"""
    rows = []
    for q in QUANTILES:
        column = f"p{int(q * 100)}"
        covered = float((results["price"] <= results[column]).mean())
        rows.append({
            "quantile": column,
            "nominal": q,
            "observed": covered,
            "error_pp": (covered - q) * 100,
        })
    return pd.DataFrame(rows)


def expected_margin(results: pd.DataFrame, marginal_cost: float) -> pd.Series:
    """Rough E[max(price - cost, 0)] from three quantiles.

    Treats P10/P50/P90 as equally weighted scenarios. Crude, but it uses the
    whole predicted distribution rather than only its middle, which is the
    point: the upside tail is what pays for a start.
    """
    scenarios = [results[f"p{int(q * 100)}"] for q in QUANTILES]
    payoffs = [(s - marginal_cost).clip(lower=0) for s in scenarios]
    return sum(payoffs) / len(payoffs)


def simulate(decisions: np.ndarray, actual: np.ndarray, *, marginal_cost: float,
             start_cost: float, capacity_mw: float) -> dict:
    """Realised profit for a run/don't-run schedule.

    Energy margin accrues in every hour the unit runs. A start cost is charged
    each time it transitions from off to on, which is what makes committing to
    a marginal hour genuinely risky.
    """
    decisions = decisions.astype(bool)
    energy = ((actual - marginal_cost) * capacity_mw)[decisions].sum()

    previous = np.concatenate(([False], decisions[:-1]))
    starts = int((decisions & ~previous).sum())

    return {
        "profit": float(energy - starts * start_cost),
        "hours_run": int(decisions.sum()),
        "starts": starts,
        "energy_margin": float(energy),
        "start_costs": float(starts * start_cost),
    }


def build_strategies(results: pd.DataFrame, marginal_cost: float) -> dict:
    actual = results["price"].to_numpy()
    return {
        "perfect foresight": actual > marginal_cost,
        "model — expected margin": (
            expected_margin(results, marginal_cost).to_numpy() > 0.5
        ),
        "model — P50 only": results["p50"].to_numpy() > marginal_cost,
        "model — P90 (aggressive)": results["p90"].to_numpy() > marginal_cost,
        "day-ahead price": results["dam"].to_numpy() > marginal_cost,
        "always run": np.ones(len(results), dtype=bool),
        "never run": np.zeros(len(results), dtype=bool),
    }


def report(results: pd.DataFrame, *, marginal_cost: float, start_cost: float,
           capacity_mw: float) -> None:
    print(f"\nDispatch simulation — {len(results):,} out-of-sample hours")
    print(f"{capacity_mw:.0f} MW peaker, ${marginal_cost:.0f}/MWh marginal cost, "
          f"${start_cost:,.0f} per start")

    cal = calibration(results)
    print("\nQuantile calibration (share of actual prices at or below each)")
    print(f"{'quantile':<12}{'nominal':>10}{'observed':>11}{'error':>10}")
    print("-" * 43)
    for _, r in cal.iterrows():
        print(f"{r['quantile']:<12}{r['nominal']*100:>9.0f}%"
              f"{r['observed']*100:>10.1f}%{r['error_pp']:>+9.1f}pp")

    worst = cal["error_pp"].abs().max()
    print("Well calibrated." if worst < 5 else
          f"Miscalibrated by up to {worst:.1f}pp — expected values built on "
          f"these are correspondingly off.")

    actual = results["price"].to_numpy()
    outcomes = {
        name: simulate(dec, actual, marginal_cost=marginal_cost,
                       start_cost=start_cost, capacity_mw=capacity_mw)
        for name, dec in build_strategies(results, marginal_cost).items()
    }
    ceiling = outcomes["perfect foresight"]["profit"]

    print(f"\nRealised profit over the period")
    head = (f"{'strategy':<26}{'profit':>14}{'vs perfect':>12}"
            f"{'hours run':>11}{'starts':>8}")
    print(head)
    print("-" * len(head))
    for name, o in sorted(outcomes.items(), key=lambda kv: -kv[1]["profit"]):
        share = f"{o['profit'] / ceiling * 100:.1f}%" if ceiling else "—"
        print(f"{name:<26}{'$'+format(o['profit'], ',.0f'):>14}{share:>12}"
              f"{o['hours_run']:>11,}{o['starts']:>8,}")

    best_real = max(
        (o for n, o in outcomes.items() if n != "perfect foresight"),
        key=lambda o: o["profit"],
    )
    best_name = next(n for n, o in outcomes.items() if o is best_real)
    dam = outcomes["day-ahead price"]["profit"]

    print(f"\nBest realisable strategy: {best_name} at ${best_real['profit']:,.0f}")
    if dam:
        print(f"Against bidding off the day-ahead price (${dam:,.0f}): "
              f"{(best_real['profit'] - dam) / abs(dam) * 100:+.1f}%")
    print(f"Perfect foresight would earn ${ceiling:,.0f}, so the forecast "
          f"captures {best_real['profit'] / ceiling * 100:.1f}% of what is "
          f"theoretically available.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", default="HB_HOUSTON")
    parser.add_argument("--initial-months", type=int, default=6)
    parser.add_argument("--marginal-cost", type=float, default=DEFAULT_MARGINAL_COST)
    parser.add_argument("--start-cost", type=float, default=DEFAULT_START_COST)
    parser.add_argument("--capacity-mw", type=float, default=DEFAULT_CAPACITY_MW)
    args = parser.parse_args()

    engine = wf.setup_database_connection()
    df = wf.load_prices(engine, args.hub)
    print(f"Loaded {len(df):,} hours for {args.hub}; fitting quantile models...")
    results = walk_forward_quantiles(df, args.initial_months)
    report(results, marginal_cost=args.marginal_cost,
           start_cost=args.start_cost, capacity_mw=args.capacity_mw)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (DispatchError, wf.WalkForwardError) as exc:
        print(f"\nDispatch simulation failed: {exc}", file=sys.stderr)
        sys.exit(1)
