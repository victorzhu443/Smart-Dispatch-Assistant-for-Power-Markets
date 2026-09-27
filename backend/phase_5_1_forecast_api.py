"""Forecast API — serves the trained quantile model and a dispatch call.

What this replaced, and why:

The previous version constructed a RandomForest at import time, trained it on
whatever was in the database that moment, and then predicted using hardcoded
feature values -- price_mean fixed at 35.0, trend_slope at 0.0, momentum at 0.0
-- varying only hour-of-day and applying hand-tuned peak multipliers. It was a
time-of-day heuristic wearing a model's clothes, it was never the model that
had been evaluated, and no two processes served the same thing.

This loads the artifact produced by python -m forecasting_model.train_model and serves
that, with enough provenance on every response to tell what answered.

Fallback policy: degrade visibly, never invent. Every degraded response is
machine-detectable by the caller through `provenance.degraded` and
`provenance.model`. The one thing this service will not do is return something
that looks like a full-confidence answer when it is not.

    model artifact missing      serve seasonal-naive, labelled, degraded=true
    database unreachable        503, health check goes red
    hour not computable         422 naming what is missing
    live forecast, stale data   503 stating the age
    interval too wide           forecast returned, recommendation withheld
    model behind the data       served, flagged degraded, retrain named

Endpoints:
    GET /forecast[?timestamp=&marginal_cost=]   quantiles + dispatch call
    GET /health                                 real readiness, not a constant
    GET /                                       this, as JSON
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = Path(os.getenv("MODEL_ARTIFACT", REPO_ROOT / "forecast_model.joblib"))

# Hours of history needed before a target hour to compute its features. The
# longest lag is 168 (a week); the margin absorbs gaps.
HISTORY_HOURS = 200

# A live forecast built on data older than this is not worth serving. Prices
# move far too fast for it to mean anything.
MAX_DATA_AGE_HOURS = float(os.getenv("MAX_DATA_AGE_HOURS", "6"))

# Above this spread the distribution is too wide to act on, so the forecast is
# returned but the recommendation is withheld rather than guessed.
MAX_INTERVAL_WIDTH = float(os.getenv("MAX_INTERVAL_WIDTH", "400"))

# How far the data may run ahead of the model's training cutoff before the
# forecast is flagged.
#
# This service already refuses to serve on data more than a few hours old, but
# it had no equivalent check on the model, so it would serve one trained eight
# months ago against today's prices reporting degraded=false. A model that has
# never seen the current regime is exactly as misleading as stale input, and
# the asymmetry meant only one of the two was ever visible.
#
# Unlike stale data, a stale model is degraded rather than unusable: it is
# still a real prediction, just fitted to an older market. So this flags and
# keeps serving instead of refusing.
MAX_MODEL_LAG_DAYS = float(os.getenv("MAX_MODEL_LAG_DAYS", "30"))

DEFAULT_MARGINAL_COST = 45.0


from forecasting_model import walk_forward as wf


class ForecastService:
    """Holds the model artifact and answers forecast requests."""

    def __init__(self, artifact_path: Path = ARTIFACT_PATH):
        self.artifact_path = artifact_path
        self.artifact = None
        self.load_error = None
        self.engine = None
        self._load_artifact()
        self._connect()

    # ---- startup -------------------------------------------------------

    def _load_artifact(self) -> None:
        if not self.artifact_path.exists():
            self.load_error = f"artifact not found at {self.artifact_path}"
            logger.warning("%s — will serve the seasonal-naive fallback",
                           self.load_error)
            return
        try:
            artifact = joblib.load(self.artifact_path)
            missing = set(artifact["feature_names"]) - set(wf.FEATURE_COLUMNS)
            extra = set(wf.FEATURE_COLUMNS) - set(artifact["feature_names"])
            if missing or extra:
                # A schema drift between the artifact and the feature builder
                # would otherwise produce confident nonsense.
                raise ValueError(
                    f"feature mismatch: artifact-only={sorted(missing)}, "
                    f"builder-only={sorted(extra)}"
                )
            self.artifact = artifact
            logger.info("loaded model %s for %s (cutoff %s)",
                        artifact["version"], artifact["hub"],
                        artifact["data_cutoff"])
        except Exception as exc:
            self.load_error = f"artifact unreadable: {exc}"
            logger.warning("%s — will serve the seasonal-naive fallback",
                           self.load_error)

    def _connect(self) -> None:
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
            self.engine = engine
        except Exception:
            self.engine = create_engine(f"sqlite:///{REPO_ROOT / 'market_data.db'}")

    # ---- data ----------------------------------------------------------

    @property
    def hub(self) -> str:
        return self.artifact["hub"] if self.artifact else "HB_HOUSTON"

    def model_lag_days(self, latest: pd.Timestamp | None) -> float | None:
        """How far the available data runs past the model's training cutoff."""
        if self.artifact is None or latest is None:
            return None
        cutoff = pd.Timestamp(self.artifact["data_cutoff"])
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        return max(0.0, (latest - cutoff).total_seconds() / 86400)

    def latest_data_time(self) -> pd.Timestamp | None:
        try:
            with self.engine.connect() as conn:
                value = conn.execute(
                    text("SELECT MAX(timestamp_utc) FROM market_data_hourly "
                         "WHERE settlement_point = :hub"),
                    {"hub": self.hub},
                ).scalar()
            return pd.to_datetime(value, utc=True, format="mixed") if value else None
        except Exception:
            return None

    def _window(self, target: pd.Timestamp) -> pd.DataFrame:
        start = target - pd.Timedelta(hours=HISTORY_HOURS)
        rt = pd.read_sql(
            text("SELECT timestamp_utc, price FROM market_data_hourly "
                 "WHERE settlement_point = :hub AND timestamp_utc BETWEEN :lo AND :hi "
                 "ORDER BY timestamp_utc"),
            self.engine,
            params={"hub": self.hub,
                    "lo": start.strftime("%Y-%m-%d %H:%M:%S%z"),
                    "hi": target.strftime("%Y-%m-%d %H:%M:%S%z")},
        )
        dam = pd.read_sql(
            text("SELECT timestamp_utc, price AS dam FROM dam_prices_hourly "
                 "WHERE settlement_point = :hub AND timestamp_utc BETWEEN :lo AND :hi "
                 "ORDER BY timestamp_utc"),
            self.engine,
            params={"hub": self.hub,
                    "lo": start.strftime("%Y-%m-%d %H:%M:%S%z"),
                    "hi": target.strftime("%Y-%m-%d %H:%M:%S%z")},
        )
        for frame in (rt, dam):
            if not frame.empty:
                frame["timestamp_utc"] = pd.to_datetime(
                    frame["timestamp_utc"], utc=True, format="mixed"
                )
        if rt.empty:
            return rt
        return rt.merge(dam, on="timestamp_utc", how="left").sort_values(
            "timestamp_utc").reset_index(drop=True)

    # ---- prediction ----------------------------------------------------

    def predict(self, target: pd.Timestamp) -> dict:
        """Quantiles for one hour, or a labelled fallback."""
        window = self._window(target)
        if window.empty:
            raise LookupError(
                f"no price history for {self.hub} at or before "
                f"{target.isoformat()}"
            )

        feat = wf.build_features(window)
        row = feat[feat["timestamp_utc"] == target]
        if row.empty:
            raise LookupError(
                f"{target.isoformat()} is not present in market_data_hourly "
                f"for {self.hub}"
            )

        if self.artifact is None:
            return self._seasonal_naive(feat, target)

        features = row[self.artifact["feature_names"]]
        if features.isna().any(axis=None):
            incomplete = [c for c in features.columns if features[c].isna().any()]
            raise LookupError(
                f"features not computable for {target.isoformat()}: "
                f"{incomplete} — needs {HISTORY_HOURS}h of prior history"
            )

        X = features.to_numpy()
        quantiles = {
            f"p{int(q * 100)}": float(model.predict(X)[0])
            for q, model in self.artifact["models"].items()
        }
        # Independently fit quantiles can cross; sorting restores order.
        ordered = sorted(quantiles.values())
        quantiles = dict(zip(["p10", "p50", "p90"], ordered))

        return {"forecast": quantiles, "model": "quantile_gbm", "degraded": False}

    def predict_range(self, start: pd.Timestamp, hours: int) -> list[dict]:
        """Quantiles for a run of consecutive hours.

        Builds features once across the whole window and predicts in one batch,
        rather than repeating 200 hours of history lookup per hour.
        """
        end = start + pd.Timedelta(hours=hours - 1)
        window = self._window(end)
        if window.empty:
            raise LookupError(f"no price history for {self.hub} near {start}")

        feat = wf.build_features(window)
        targets = feat[
            (feat["timestamp_utc"] >= start) & (feat["timestamp_utc"] <= end)
        ]
        if targets.empty:
            raise LookupError(
                f"no hours between {start.isoformat()} and {end.isoformat()} "
                f"are present for {self.hub}"
            )

        if self.artifact is None:
            out = []
            for ts in targets["timestamp_utc"]:
                try:
                    result = self._seasonal_naive(feat, ts)
                    result["timestamp"] = ts
                    out.append(result)
                except LookupError:
                    continue
            return out

        usable = targets.dropna(subset=self.artifact["feature_names"])
        if usable.empty:
            raise LookupError(
                f"features not computable for any hour in the requested range; "
                f"needs {HISTORY_HOURS}h of prior history"
            )

        X = usable[self.artifact["feature_names"]].to_numpy()
        predictions = {
            f"p{int(q * 100)}": model.predict(X)
            for q, model in self.artifact["models"].items()
        }

        out = []
        for i, ts in enumerate(usable["timestamp_utc"]):
            ordered = sorted(float(predictions[k][i]) for k in ("p10", "p50", "p90"))
            out.append({
                "timestamp": ts,
                "forecast": dict(zip(["p10", "p50", "p90"], ordered)),
                "model": "quantile_gbm",
                "degraded": False,
            })
        return out

    def _seasonal_naive(self, feat: pd.DataFrame, target: pd.Timestamp) -> dict:
        """Fallback when no artifact is loadable.

        Same hour yesterday, with an interval from recent volatility. A real,
        defensible predictor -- just a much weaker one, and labelled as such.
        """
        row = feat[feat["timestamp_utc"] == target]
        centre = row["lag_24"].iloc[0]
        if pd.isna(centre):
            centre = row["lag_1"].iloc[0]
        if pd.isna(centre):
            raise LookupError(f"no basis for a fallback forecast at {target}")

        spread = row["roll_std_24"].iloc[0]
        spread = float(spread) if pd.notna(spread) else 10.0
        centre = float(centre)

        return {
            "forecast": {
                "p10": round(centre - 1.28 * spread, 2),
                "p50": round(centre, 2),
                "p90": round(centre + 1.28 * spread, 2),
            },
            "model": "seasonal_naive_fallback",
            "degraded": True,
        }


def recommend(quantiles: dict, marginal_cost: float) -> dict:
    """Turn a predicted distribution into a dispatch call.

    Uses the P50 threshold, which beat both the expected-margin and aggressive
    P90 rules in backtesting -- the fancier rules over-commit and lose the
    difference to start costs.
    """
    width = quantiles["p90"] - quantiles["p10"]
    if width > MAX_INTERVAL_WIDTH:
        return {
            "action": "no_action",
            "reason": "interval_too_wide",
            "detail": (f"P10-P90 spread of ${width:,.0f}/MWh exceeds the "
                       f"${MAX_INTERVAL_WIDTH:,.0f} threshold; too uncertain "
                       f"to call"),
            "interval_width": round(width, 2),
        }

    margin = quantiles["p50"] - marginal_cost
    return {
        "action": "run" if margin > 0 else "hold",
        "reason": "p50_above_marginal_cost" if margin > 0
                  else "p50_below_marginal_cost",
        "expected_margin_per_mwh": round(margin, 2),
        "marginal_cost": marginal_cost,
        "interval_width": round(width, 2),
    }


service = ForecastService()

app = Flask(__name__)
CORS(app)


def _provenance(result: dict, data_age_hours: float | None,
                latest: "pd.Timestamp | None" = None) -> dict:
    artifact = service.artifact
    lag_days = service.model_lag_days(latest)
    model_stale = lag_days is not None and lag_days > MAX_MODEL_LAG_DAYS

    # A stale model is a degraded answer even though the artifact loaded fine.
    degraded = bool(result["degraded"] or model_stale)

    if result["degraded"]:
        reason = service.load_error
    elif model_stale:
        reason = (f"model trained through "
                  f"{artifact['data_cutoff']:%Y-%m-%d}, "
                  f"{lag_days:,.0f} days behind the data "
                  f"(limit {MAX_MODEL_LAG_DAYS:.0f}); retrain with "
                  f"python -m forecasting_model.train_model")
    else:
        reason = None

    return {
        "model": result["model"],
        "model_version": artifact["version"] if artifact else None,
        "degraded": degraded,
        "fallback_reason": reason,
        "training_cutoff": (artifact["data_cutoff"].isoformat()
                            if artifact else None),
        "model_lag_days": round(lag_days, 1) if lag_days is not None else None,
        "model_stale": model_stale,
        "data_age_hours": (round(data_age_hours, 1)
                           if data_age_hours is not None else None),
        "hub": service.hub,
    }


@app.route("/forecast", methods=["GET"])
def forecast_endpoint():
    if service.engine is None:
        return jsonify({"error": "database unavailable"}), 503

    raw_timestamp = (request.args.get("timestamp") or "").strip()
    try:
        marginal_cost = float(request.args.get("marginal_cost",
                                               DEFAULT_MARGINAL_COST))
    except ValueError:
        return jsonify({
            "error": "marginal_cost must be a number",
            "received": request.args.get("marginal_cost"),
        }), 422

    latest = service.latest_data_time()
    if latest is None:
        return jsonify({
            "error": "no market data available",
            "detail": "run python -m data_ingestion.ingest_ercot_history",
        }), 503

    now = datetime.now(timezone.utc)
    data_age = (now - latest.to_pydatetime()).total_seconds() / 3600

    if raw_timestamp:
        try:
            target = pd.Timestamp(raw_timestamp)
            target = (target.tz_localize("UTC") if target.tzinfo is None
                      else target.tz_convert("UTC"))
        except Exception:
            return jsonify({
                "error": "invalid timestamp",
                "expected": "ISO-8601, e.g. 2025-12-15T18:00:00Z",
                "received": raw_timestamp,
            }), 422
    else:
        # A live forecast is only meaningful on fresh data. A historical one
        # explicitly requested by timestamp is fine regardless of age.
        if data_age > MAX_DATA_AGE_HOURS:
            return jsonify({
                "error": "market data is stale",
                "detail": (f"latest hour is {latest.isoformat()}, "
                           f"{data_age:,.1f}h old; the limit for a live "
                           f"forecast is {MAX_DATA_AGE_HOURS:.0f}h"),
                "remedy": ("re-run python -m data_ingestion.ingest_ercot_history, or "
                           "request a specific historical hour with "
                           "?timestamp="),
                "latest_available": latest.isoformat(),
            }), 503
        target = latest

    try:
        result = service.predict(target)
    except LookupError as exc:
        return jsonify({
            "error": "forecast not available for that hour",
            "detail": str(exc),
            "latest_available": latest.isoformat(),
        }), 422

    return jsonify({
        "hub": service.hub,
        "timestamp": target.isoformat(),
        "forecast": {k: round(v, 2) for k, v in result["forecast"].items()},
        "currency": "USD/MWh",
        "recommendation": recommend(result["forecast"], marginal_cost),
        "provenance": _provenance(result, data_age, latest),
        "generated_at": now.isoformat(),
    })


MAX_RANGE_HOURS = 168


@app.route("/forecast/range", methods=["GET"])
def forecast_range_endpoint():
    """Consecutive hours in one call, for charting a forecast band."""
    if service.engine is None:
        return jsonify({"error": "database unavailable"}), 503

    try:
        hours = int(request.args.get("hours", 24))
        marginal_cost = float(request.args.get("marginal_cost",
                                               DEFAULT_MARGINAL_COST))
    except ValueError:
        return jsonify({
            "error": "hours and marginal_cost must be numbers",
        }), 422

    if not 1 <= hours <= MAX_RANGE_HOURS:
        return jsonify({
            "error": f"hours must be between 1 and {MAX_RANGE_HOURS}",
            "received": hours,
        }), 422

    latest = service.latest_data_time()
    if latest is None:
        return jsonify({
            "error": "no market data available",
            "detail": "run python -m data_ingestion.ingest_ercot_history",
        }), 503

    raw_start = (request.args.get("start") or "").strip()
    if raw_start:
        try:
            start = pd.Timestamp(raw_start)
            start = (start.tz_localize("UTC") if start.tzinfo is None
                     else start.tz_convert("UTC"))
        except Exception:
            return jsonify({
                "error": "invalid start",
                "expected": "ISO-8601, e.g. 2025-12-15T00:00:00Z",
                "received": raw_start,
            }), 422
    else:
        # Default to the most recent full window that the data can support.
        start = latest - pd.Timedelta(hours=hours - 1)

    try:
        predictions = service.predict_range(start, hours)
    except LookupError as exc:
        return jsonify({
            "error": "forecast not available for that range",
            "detail": str(exc),
            "latest_available": latest.isoformat(),
        }), 422

    now = datetime.now(timezone.utc)
    data_age = (now - latest.to_pydatetime()).total_seconds() / 3600

    hours_out = [{
        "timestamp": p["timestamp"].isoformat(),
        "forecast": {k: round(v, 2) for k, v in p["forecast"].items()},
        "recommendation": recommend(p["forecast"], marginal_cost),
    } for p in predictions]

    degraded = any(p["degraded"] for p in predictions)
    model_name = predictions[0]["model"] if predictions else None

    return jsonify({
        "hub": service.hub,
        "start": start.isoformat(),
        "hours": len(hours_out),
        "marginal_cost": marginal_cost,
        "currency": "USD/MWh",
        "series": hours_out,
        "provenance": _provenance(
            {"model": model_name, "degraded": degraded}, data_age, latest
        ),
        "generated_at": now.isoformat(),
    })


@app.route("/health", methods=["GET"])
def health_check():
    """Real readiness. Reports unhealthy when it genuinely cannot serve."""
    latest = service.latest_data_time()
    database_ok = latest is not None
    model_ok = service.artifact is not None

    age_hours = None
    if latest is not None:
        age_hours = (datetime.now(timezone.utc)
                     - latest.to_pydatetime()).total_seconds() / 3600

    # Stale data means live forecasts will 503, so reporting "healthy" would
    # keep an orchestrator routing traffic that cannot be served. Historical
    # queries still work, hence degraded rather than unhealthy.
    data_fresh = age_hours is not None and age_hours <= MAX_DATA_AGE_HOURS

    lag_days = service.model_lag_days(latest)
    model_current = lag_days is None or lag_days <= MAX_MODEL_LAG_DAYS

    if not database_ok:
        status, code = "unhealthy", 503
    elif not model_ok or not data_fresh or not model_current:
        status, code = "degraded", 200
    else:
        status, code = "healthy", 200

    return jsonify({
        "status": status,
        "service": "forecast-api",
        "checks": {
            "database": "ok" if database_ok else "unreachable",
            "model_artifact": "ok" if model_ok else (service.load_error or "missing"),
            "data_freshness": "ok" if data_fresh else (
                f"stale: {age_hours:,.1f}h old, limit {MAX_DATA_AGE_HOURS:.0f}h"
                if age_hours is not None else "unknown"
            ),
            "model_currency": "ok" if model_current else (
                f"stale: {lag_days:,.0f} days behind the data, "
                f"limit {MAX_MODEL_LAG_DAYS:.0f}"
            ),
        },
        "model_lag_days": round(lag_days, 1) if lag_days is not None else None,
        "model_version": service.artifact["version"] if model_ok else None,
        "hub": service.hub,
        "latest_data": latest.isoformat() if latest is not None else None,
        "data_age_hours": round(age_hours, 1) if age_hours is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), code


@app.route("/", methods=["GET"])
def api_info():
    return jsonify({
        "name": "Smart Dispatch Forecast API",
        "serves": "quantile price forecast and a dispatch recommendation",
        "endpoints": {
            "/forecast": {
                "method": "GET",
                "parameters": {
                    "timestamp": "ISO-8601 hour; omit for the latest available",
                    "marginal_cost": f"$/MWh, default {DEFAULT_MARGINAL_COST}",
                },
                "example": "/forecast?timestamp=2025-12-15T18:00:00Z&marginal_cost=45",
            },
            "/health": {"method": "GET", "description": "readiness"},
        },
        "fallback_policy": {
            "principle": "degrade visibly, never invent",
            "detectable_via": "provenance.degraded and provenance.model",
        },
    })


def main() -> None:
    logger.info("Forecast API starting on :5001")
    if service.artifact is None:
        logger.warning("no model artifact — serving the labelled fallback. "
                       "Run: python -m forecasting_model.train_model")
    app.run(host="0.0.0.0", port=5001, debug=False)


if __name__ == "__main__":
    main()
