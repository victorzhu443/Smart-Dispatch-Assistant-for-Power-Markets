"""Input-validation hardening for the forecast API (Semgrep `nan-injection`).

`float(request.args.get("marginal_cost"))` accepts the strings "nan", "inf" and
"-inf" (and overflowing literals like "1e999" -> inf) without raising. An
unvalidated non-finite value then flows into `recommend()` as the margin
reference, yielding a NaN `expected_margin_per_mwh` and, worse, a response body
with a bare `NaN` token — which is not valid JSON and breaks strict consumers.

These tests assert the endpoints reject a non-finite `marginal_cost` with a
clean 422 *before* any forecast is computed, while leaving ordinary numeric
input (including negatives and the default) untouched. They run against the real
Flask app via its test client, so they also prove the reject happens ahead of
the data-staleness / prediction path.
"""
import importlib.util
import warnings
from pathlib import Path

import pytest

pytest.importorskip("flask", reason="Flask not installed")

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def client():
    warnings.filterwarnings("ignore")
    spec = importlib.util.spec_from_file_location(
        "forecast_api", REPO_ROOT / "backend" / "phase_5_1_forecast_api.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = module.app.test_client()
    # /forecast* is bearer-gated (RC-6); present the dev default like Django.
    c.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {module.FORECAST_API_KEY}"
    return c


NON_FINITE = ["nan", "NaN", "inf", "Inf", "-inf", "infinity", "1e999", "-1e999"]


@pytest.mark.parametrize("value", NON_FINITE)
def test_forecast_rejects_non_finite_marginal_cost(client, value):
    resp = client.get(f"/forecast?marginal_cost={value}")
    assert resp.status_code == 422, (value, resp.status_code)
    # get_json() parsing at all proves the body is strict JSON with no bare NaN
    # token (the pre-fix recommendation path emitted `"expected_margin": NaN`).
    body = resp.get_json()
    assert "finite" in body["error"]


@pytest.mark.parametrize("value", NON_FINITE)
def test_forecast_range_rejects_non_finite_marginal_cost(client, value):
    resp = client.get(f"/forecast/range?hours=6&marginal_cost={value}")
    assert resp.status_code == 422, (value, resp.status_code)
    assert "finite" in resp.get_json()["error"]


@pytest.mark.parametrize("value", ["abc", "4,5", "", "$45"])
def test_forecast_still_rejects_non_numeric_marginal_cost(client, value):
    # Pre-existing behaviour must be preserved: a non-number is still a 422.
    resp = client.get(f"/forecast?marginal_cost={value}")
    assert resp.status_code == 422


@pytest.mark.parametrize("value", ["45", "0", "-10.5", "45.0"])
def test_forecast_accepts_ordinary_marginal_cost(client, value):
    # A finite number must NOT be rejected by the new guard. The endpoint may
    # still 503/422 downstream (stale data, missing hour), but never on the
    # finite-check path — so the one thing we assert is: not the finite error.
    resp = client.get(f"/forecast?marginal_cost={value}")
    body = resp.get_json() or {}
    assert body.get("error") != "marginal_cost must be a finite number"
