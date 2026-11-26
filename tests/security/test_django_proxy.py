"""The Django proxy is the forecast API's only client (SECURITY.md RC-1/RC-6).

- A malformed numeric parameter is a 400 from Django and never leaves the
  process (the upstream call is not made).
- Every upstream call carries the bearer token.
- Only whitelisted parameters are forwarded.
"""
import sys
from pathlib import Path
from unittest import mock

import pytest

django = pytest.importorskip("django", reason="Django not installed")

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


@pytest.fixture(scope="module")
def client():
    import os
    if str(FRONTEND) not in sys.path:
        sys.path.insert(0, str(FRONTEND))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "smart_ui.settings")
    from django.apps import apps
    if not apps.ready:
        django.setup()
    from django.test import Client
    return Client()


class _Upstream:
    status_code = 200

    def json(self):
        return {"ok": True}


@pytest.mark.parametrize("path", [
    "/api/forecast?marginal_cost=nan",
    "/api/forecast?marginal_cost=inf",
    "/api/forecast?marginal_cost=1e999",
    "/api/forecast?marginal_cost=abc",
    "/api/forecast/range?hours=0",
    "/api/forecast/range?hours=999",
    "/api/forecast/range?hours=lots",
    "/api/forecast/range?hours=6&marginal_cost=-inf",
])
def test_malformed_parameter_is_400_and_never_forwarded(client, path):
    with mock.patch("dashboard.views.requests.get") as get:
        resp = client.get(path)
    assert resp.status_code == 400
    assert "error" in resp.json()
    get.assert_not_called()


def test_valid_request_is_forwarded_with_the_bearer(client):
    from django.conf import settings
    with mock.patch("dashboard.views.requests.get", return_value=_Upstream()) as get:
        resp = client.get("/api/forecast/range?start=2025-12-15T00:00:00Z&hours=6"
                          "&marginal_cost=45&evil=1")
    assert resp.status_code == 200
    kwargs = get.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == f"Bearer {settings.FORECAST_API_KEY}"
    assert set(kwargs["params"]) == {"start", "hours", "marginal_cost"}  # whitelist


def test_upstream_status_passes_through(client):
    class Upstream(_Upstream):
        status_code = 422

        def json(self):
            return {"error": "hours: above maximum 168"}

    with mock.patch("dashboard.views.requests.get", return_value=Upstream()):
        resp = client.get("/api/forecast?timestamp=2025-12-15T18:00:00Z")
    assert resp.status_code == 422
