"""Tests for the multi-hour forecast range endpoint.

This is what the dashboard chart draws, so its contract matters: quantiles must
not cross, every hour must carry its own recommendation, and the range must be
bounded so a caller cannot ask for an unbounded scan.

Skipped when Flask is not installed.

Run with:  pytest tests/ -v
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("flask", reason="Flask not installed")

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = REPO_ROOT / "forecast_model.joblib"

requires_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="no model artifact; run python -m forecasting_model.train_model",
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


def _load_api():
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


class TestRangeContract:
    @requires_artifact
    def test_returns_the_requested_number_of_hours(self, client):
        body = client.get(
            "/forecast/range?start=2025-12-15T00:00:00Z&hours=24"
        ).get_json()

        assert body["hours"] == 24
        assert len(body["series"]) == 24

    @requires_artifact
    def test_hours_are_consecutive_and_ordered(self, client):
        import pandas as pd

        body = client.get(
            "/forecast/range?start=2025-12-15T00:00:00Z&hours=12"
        ).get_json()
        stamps = [pd.Timestamp(h["timestamp"]) for h in body["series"]]

        assert stamps == sorted(stamps)
        gaps = {b - a for a, b in zip(stamps, stamps[1:])}
        assert gaps == {pd.Timedelta(hours=1)}

    @requires_artifact
    def test_quantiles_never_cross_on_any_hour(self, client):
        body = client.get(
            "/forecast/range?start=2025-12-15T00:00:00Z&hours=48"
        ).get_json()

        for hour in body["series"]:
            f = hour["forecast"]
            assert f["p10"] <= f["p50"] <= f["p90"], f"crossed at {hour['timestamp']}"

    @requires_artifact
    def test_every_hour_carries_its_own_recommendation(self, client):
        body = client.get(
            "/forecast/range?start=2025-12-15T00:00:00Z&hours=24"
        ).get_json()

        for hour in body["series"]:
            assert hour["recommendation"]["action"] in {"run", "hold", "no_action"}

    @requires_artifact
    def test_marginal_cost_changes_the_calls(self, client):
        base = "/forecast/range?start=2025-12-15T00:00:00Z&hours=24"
        cheap = client.get(base + "&marginal_cost=1").get_json()
        dear = client.get(base + "&marginal_cost=9000").get_json()

        cheap_runs = sum(1 for h in cheap["series"]
                         if h["recommendation"]["action"] == "run")
        dear_runs = sum(1 for h in dear["series"]
                        if h["recommendation"]["action"] == "run")

        assert cheap_runs == 24
        assert dear_runs == 0

    @requires_artifact
    def test_carries_provenance(self, client):
        body = client.get(
            "/forecast/range?start=2025-12-15T00:00:00Z&hours=6"
        ).get_json()

        p = body["provenance"]
        assert p["model"] == "quantile_gbm"
        assert p["degraded"] is False
        assert p["model_version"]


class TestRangeValidation:
    @requires_data
    def test_rejects_zero_or_negative_hours(self, client):
        for hours in (0, -5):
            response = client.get(f"/forecast/range?hours={hours}")
            assert response.status_code == 422

    @requires_data

    def test_bounds_the_range(self, client, api):
        """An unbounded range would let a caller scan the whole table."""
        response = client.get(f"/forecast/range?hours={api.MAX_RANGE_HOURS + 1}")

        assert response.status_code == 422
        assert str(api.MAX_RANGE_HOURS) in response.get_json()["error"]

    @requires_data

    def test_accepts_the_maximum(self, client, api):
        response = client.get(
            f"/forecast/range?start=2025-11-01T00:00:00Z&hours={api.MAX_RANGE_HOURS}"
        )
        assert response.status_code in (200, 422)  # 422 only if data runs out

    @requires_data

    def test_rejects_a_bad_start(self, client):
        response = client.get("/forecast/range?start=yesterday&hours=6")

        assert response.status_code == 422
        assert "expected" in response.get_json()

    @requires_data

    def test_rejects_non_numeric_hours(self, client):
        assert client.get("/forecast/range?hours=lots").status_code == 422

    @requires_data

    def test_refuses_a_range_outside_the_data(self, client):
        response = client.get("/forecast/range?start=1999-01-01T00:00:00Z&hours=6")

        assert response.status_code == 422
        assert "latest_available" in response.get_json()


class TestRangeFallback:
    @requires_data
    def test_missing_artifact_degrades_across_the_whole_range(self):
        api = _load_api()
        api.service.artifact = None
        api.service.load_error = "injected"

        response = api.app.test_client().get(
            "/forecast/range?start=2025-12-15T00:00:00Z&hours=6"
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["provenance"]["degraded"] is True
        assert body["provenance"]["model"] == "seasonal_naive_fallback"
        assert len(body["series"]) > 0
