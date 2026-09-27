"""Tests for model staleness detection.

The service refused to serve on data more than a few hours old, but had no
equivalent check on the model, so it would serve one trained eight months
earlier against today's prices reporting `degraded: false`. A model that has
never seen the current regime is as misleading as stale input; the asymmetry
meant only one of the two was ever visible.

Unlike stale data, a stale model is degraded rather than unusable, so these
assert it keeps serving while flagging.

Run with:  pytest tests/ -v
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("flask", reason="Flask not installed")

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = REPO_ROOT / "forecast_model.joblib"

requires_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="no model artifact; run python -m forecasting_model.train_model",
)


def _load_api():
    spec = importlib.util.spec_from_file_location(
        "forecast_api", REPO_ROOT / "backend" / "phase_5_1_forecast_api.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _age_the_model(api, days: int):
    """Rewind the artifact's training cutoff to simulate drift."""
    latest = api.service.latest_data_time()
    api.service.artifact["data_cutoff"] = latest - pd.Timedelta(days=days)
    return api


class TestLagMeasurement:
    @requires_artifact
    def test_lag_is_zero_for_a_freshly_trained_artifact(self):
        api = _load_api()
        latest = api.service.latest_data_time()

        lag = api.service.model_lag_days(latest)

        assert lag is not None
        assert lag < 1.0

    @requires_artifact
    def test_lag_grows_as_the_cutoff_recedes(self):
        api = _age_the_model(_load_api(), days=100)
        latest = api.service.latest_data_time()

        assert api.service.model_lag_days(latest) == pytest.approx(100.0, abs=0.1)

    @requires_artifact
    def test_lag_never_goes_negative(self):
        """A cutoff ahead of the data is odd but must not read as negative."""
        api = _load_api()
        latest = api.service.latest_data_time()
        api.service.artifact["data_cutoff"] = latest + pd.Timedelta(days=5)

        assert api.service.model_lag_days(latest) == 0.0

    def test_lag_is_unknown_without_an_artifact(self):
        api = _load_api()
        api.service.artifact = None

        assert api.service.model_lag_days(pd.Timestamp.now(tz="UTC")) is None


class TestStaleModelIsFlagged:
    @requires_artifact
    def test_stale_model_marks_the_response_degraded(self):
        api = _age_the_model(_load_api(), days=244)  # the lag actually hit

        body = api.app.test_client().get(
            "/forecast?timestamp=2026-08-15T18:00:00Z"
        ).get_json()
        p = body["provenance"]

        assert p["model_stale"] is True
        assert p["degraded"] is True
        assert p["model_lag_days"] == pytest.approx(244, abs=1)

    @requires_artifact
    def test_the_reason_names_the_remedy(self):
        api = _age_the_model(_load_api(), days=244)

        body = api.app.test_client().get(
            "/forecast?timestamp=2026-08-15T18:00:00Z"
        ).get_json()

        assert "forecasting_model.train_model" in body["provenance"]["fallback_reason"]

    @requires_artifact
    def test_a_stale_model_still_serves(self):
        """Degraded, not unusable: it is a real prediction from an old fit."""
        api = _age_the_model(_load_api(), days=244)

        response = api.app.test_client().get(
            "/forecast?timestamp=2026-08-15T18:00:00Z"
        )

        assert response.status_code == 200
        f = response.get_json()["forecast"]
        assert f["p10"] <= f["p50"] <= f["p90"]

    @requires_artifact
    def test_a_current_model_is_not_flagged(self):
        api = _load_api()

        body = api.app.test_client().get(
            "/forecast?timestamp=2026-08-15T18:00:00Z"
        ).get_json()

        assert body["provenance"]["model_stale"] is False
        assert body["provenance"]["degraded"] is False

    @requires_artifact
    def test_just_inside_the_threshold_is_not_flagged(self):
        api = _age_the_model(_load_api(), days=int(api_max() - 1))

        body = api.app.test_client().get(
            "/forecast?timestamp=2026-08-15T18:00:00Z"
        ).get_json()

        assert body["provenance"]["model_stale"] is False


def api_max():
    return _load_api().MAX_MODEL_LAG_DAYS


class TestHealthReportsModelCurrency:
    @requires_artifact
    def test_health_has_a_model_currency_check(self):
        body = _load_api().app.test_client().get("/health").get_json()

        assert "model_currency" in body["checks"]
        assert "model_lag_days" in body

    @requires_artifact
    def test_stale_model_degrades_health(self):
        api = _age_the_model(_load_api(), days=244)

        body = api.app.test_client().get("/health").get_json()

        assert body["status"] == "degraded"
        assert "stale" in body["checks"]["model_currency"]

    @requires_artifact
    def test_current_model_reports_ok(self):
        body = _load_api().app.test_client().get("/health").get_json()

        assert body["checks"]["model_currency"] == "ok"


class TestRangeEndpointCarriesTheSameFlag:
    @requires_artifact
    def test_range_provenance_reports_staleness(self):
        api = _age_the_model(_load_api(), days=244)

        body = api.app.test_client().get(
            "/forecast/range?start=2026-08-15T00:00:00Z&hours=6"
        ).get_json()

        assert body["provenance"]["model_stale"] is True
        assert body["provenance"]["degraded"] is True
