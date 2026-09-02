"""Tests for the forecast API, including its fallback ladder.

The fallbacks are the point. A service that returns a plausible number when it
should be refusing is worse than one that is simply down, because nothing
downstream can tell. Each degraded path is exercised by fault injection and
asserted to be machine-detectable by the caller.

Skipped when Flask is not installed.

Run with:  pytest tests/ -v
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("flask", reason="Flask not installed")

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = REPO_ROOT / "forecast_model.joblib"


def _load_api(monkeypatch=None):
    spec = importlib.util.spec_from_file_location(
        "forecast_api", REPO_ROOT / "backend" / "phase_5_1_forecast_api.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def api():
    return _load_api()


@pytest.fixture
def client(api):
    return api.app.test_client()


requires_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="no model artifact; run forecasting-model/train_model.py",
)

def _has_market_data() -> bool:
    """True when a populated database is available.

    CI has no database, so integration-shaped tests skip there and the pure
    logic tests still run. That is the honest split: these assertions need
    real rows, and fabricating rows to make them pass would defeat the point.
    """
    db = REPO_ROOT / "market_data.db"
    if not db.exists():
        return False
    import sqlite3
    try:
        con = sqlite3.connect(db)
        n = con.execute("SELECT COUNT(*) FROM market_data_hourly").fetchone()[0]
        con.close()
        return n > 0
    except Exception:
        return False


requires_data = pytest.mark.skipif(
    not _has_market_data(),
    reason="no populated database; run 'make backfill'",
)



class TestRecommendationRule:
    """Pure logic, no data or model needed."""

    def test_runs_when_median_clears_marginal_cost(self, api):
        rec = api.recommend({"p10": 40.0, "p50": 60.0, "p90": 80.0}, 45.0)

        assert rec["action"] == "run"
        assert rec["expected_margin_per_mwh"] == pytest.approx(15.0)

    def test_holds_when_median_is_below_marginal_cost(self, api):
        rec = api.recommend({"p10": 10.0, "p50": 30.0, "p90": 50.0}, 45.0)

        assert rec["action"] == "hold"
        assert rec["expected_margin_per_mwh"] == pytest.approx(-15.0)

    def test_withholds_the_call_when_the_interval_is_too_wide(self, api):
        """Too uncertain to act on is a valid answer; guessing is not."""
        rec = api.recommend({"p10": 10.0, "p50": 60.0, "p90": 5000.0}, 45.0)

        assert rec["action"] == "no_action"
        assert rec["reason"] == "interval_too_wide"
        assert "interval_width" in rec

    def test_marginal_cost_moves_the_decision(self, api):
        quantiles = {"p10": 40.0, "p50": 60.0, "p90": 80.0}

        assert api.recommend(quantiles, 45.0)["action"] == "run"
        assert api.recommend(quantiles, 75.0)["action"] == "hold"


class TestForecastResponse:
    @requires_artifact
    def test_returns_ordered_quantiles_with_provenance(self, client):
        response = client.get("/forecast?timestamp=2025-12-15T18:00:00Z")

        assert response.status_code == 200
        body = response.get_json()

        f = body["forecast"]
        assert f["p10"] <= f["p50"] <= f["p90"], "quantiles must not cross"

        p = body["provenance"]
        assert p["degraded"] is False
        assert p["model"] == "quantile_gbm"
        assert p["model_version"]
        assert p["training_cutoff"]

    @requires_artifact
    def test_marginal_cost_is_honoured(self, client):
        cheap = client.get(
            "/forecast?timestamp=2025-12-15T18:00:00Z&marginal_cost=1"
        ).get_json()
        dear = client.get(
            "/forecast?timestamp=2025-12-15T18:00:00Z&marginal_cost=9000"
        ).get_json()

        assert cheap["recommendation"]["action"] == "run"
        assert dear["recommendation"]["action"] == "hold"


class TestInputValidation:
    @requires_data
    def test_rejects_an_unparseable_timestamp(self, client):
        response = client.get("/forecast?timestamp=not-a-date")

        assert response.status_code == 422
        assert "expected" in response.get_json()

    @requires_data

    def test_rejects_a_non_numeric_marginal_cost(self, client):
        response = client.get(
            "/forecast?timestamp=2025-12-15T18:00:00Z&marginal_cost=cheap"
        )

        assert response.status_code == 422

    @requires_data

    def test_refuses_an_hour_outside_the_data(self, client):
        """Better to say the hour is unavailable than to extrapolate."""
        response = client.get("/forecast?timestamp=1999-01-01T00:00:00Z")

        assert response.status_code == 422
        body = response.get_json()
        assert "latest_available" in body


class TestFallbackLadder:
    """Fault injection. Each degraded path must be visible to the caller."""

    @requires_data

    def test_missing_artifact_serves_a_labelled_fallback(self, monkeypatch):
        api = _load_api()
        # Simulate the artifact having been unreadable at startup.
        api.service.artifact = None
        api.service.load_error = "artifact not found (injected)"

        response = api.app.test_client().get(
            "/forecast?timestamp=2025-12-15T18:00:00Z"
        )

        assert response.status_code == 200, "should degrade, not fail outright"
        body = response.get_json()

        assert body["provenance"]["degraded"] is True
        assert body["provenance"]["model"] == "seasonal_naive_fallback"
        assert body["provenance"]["fallback_reason"]
        assert body["forecast"]["p10"] <= body["forecast"]["p90"]

    @requires_data

    def test_fallback_is_distinguishable_from_a_real_forecast(self, monkeypatch):
        """A caller must be able to tell the difference programmatically."""
        api = _load_api()
        api.service.artifact = None
        api.service.load_error = "injected"

        body = api.app.test_client().get(
            "/forecast?timestamp=2025-12-15T18:00:00Z"
        ).get_json()

        assert body["provenance"]["model"] != "quantile_gbm"
        assert body["provenance"]["model_version"] is None

    @requires_data

    def test_live_forecast_refuses_on_stale_data(self, client, api):
        """No timestamp means "now", which stale data cannot answer."""
        response = client.get("/forecast")

        # The committed dataset ends 2026-01-01, so a live request must refuse.
        if response.status_code == 503:
            body = response.get_json()
            assert "stale" in body["error"]
            assert "latest_available" in body
            assert "remedy" in body
        else:
            # Only valid if the data really is fresh.
            assert response.get_json()["provenance"]["data_age_hours"] <= \
                api.MAX_DATA_AGE_HOURS

    @requires_data

    def test_historical_request_still_works_when_data_is_stale(self, client):
        """Staleness blocks live forecasts only, not explicit historical ones."""
        response = client.get("/forecast?timestamp=2025-12-15T18:00:00Z")

        assert response.status_code == 200

    def test_unreachable_database_reports_unhealthy(self, monkeypatch):
        api = _load_api()
        monkeypatch.setattr(api.service, "latest_data_time", lambda: None)

        response = api.app.test_client().get("/health")

        assert response.status_code == 503
        assert response.get_json()["status"] == "unhealthy"
        assert response.get_json()["checks"]["database"] == "unreachable"


class TestHealth:
    @requires_data
    def test_reports_the_real_component_state(self, client):
        body = client.get("/health").get_json()

        assert body["status"] in {"healthy", "degraded", "unhealthy"}
        for check in ("database", "model_artifact", "data_freshness"):
            assert check in body["checks"]

    @requires_data

    def test_stale_data_is_not_reported_as_healthy(self, client, api):
        """Otherwise an orchestrator keeps routing traffic that will 503."""
        body = client.get("/health").get_json()
        age = body["data_age_hours"]

        if age is not None and age > api.MAX_DATA_AGE_HOURS:
            assert body["status"] == "degraded"
            assert "stale" in body["checks"]["data_freshness"]

    @requires_data

    def test_missing_model_degrades_rather_than_fails(self, monkeypatch):
        api = _load_api()
        api.service.artifact = None
        api.service.load_error = "injected"

        response = api.app.test_client().get("/health")

        assert response.status_code == 200
        assert response.get_json()["status"] == "degraded"
        assert response.get_json()["model_version"] is None
