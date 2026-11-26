"""Service-edge access control on the forecast API (SECURITY.md RC-6).

Runs against the real Flask app through its test client with no environment
set, i.e. exactly the zero-configuration developer flow:

- /forecast* without a bearer token, or with the wrong one, is a 401 before
  any query parameter is read;
- the dev-default key (what the Django proxy sends) is accepted;
- a burst past the per-key limit is a 429, and so is a flood of *rejected*
  requests from one address (the limiter counts before the gate answers);
- a non-ASCII token is a 401, not a 500;
- /health and / stay open for probes and exempt from the limit;
- a CORS preflight from an origin other than the dashboard gets no
  Access-Control-Allow-Origin header, while the dashboard origin does.
"""
import pytest

pytest.importorskip("flask", reason="Flask not installed")
pytest.importorskip("flask_limiter", reason="Flask-Limiter not installed")

from _api import load_api  # noqa: E402

DEV_KEY = "dev-only-forecast-key"


@pytest.fixture(scope="module")
def api():
    return load_api("forecast_api_access")


@pytest.fixture
def client(api):
    return api.app.test_client()


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("path", ["/forecast", "/forecast/range?hours=6",
                                  "/forecast?marginal_cost=45"])
def test_no_key_is_401(client, path):
    resp = client.get(path)
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "unauthorized"


@pytest.mark.parametrize("header", [
    {"Authorization": "Bearer wrong-key"},
    {"Authorization": "Bearer "},
    {"Authorization": "Basic ZGV2OmRldg=="},
    {"Authorization": DEV_KEY},              # scheme missing
    {"X-Api-Key": DEV_KEY},                  # wrong header
])
def test_wrong_or_malformed_credential_is_401(client, header):
    assert client.get("/forecast", headers=header).status_code == 401


def test_dev_default_key_is_accepted(api, client):
    assert api.FORECAST_API_KEY == DEV_KEY  # APP_ENV unset -> dev default
    resp = client.get("/forecast?marginal_cost=45", headers=_bearer(DEV_KEY))
    # Past the gate. Downstream may still 503 (no DB/stale data) or 422, but
    # never 401/429 here.
    assert resp.status_code not in (401, 403, 429)


def test_auth_runs_before_parameter_validation(client):
    # A malformed value with no key is still a 401, not a 422: nothing about
    # the request is interpreted before the caller is identified.
    assert client.get("/forecast?marginal_cost=nan").status_code == 401


@pytest.mark.parametrize("path", ["/health", "/"])
def test_probe_routes_stay_open(client, path):
    resp = client.get(path)
    assert resp.status_code in (200, 503)   # 503 = DB genuinely unreachable
    assert resp.status_code != 401


def test_preflight_from_foreign_origin_is_not_allowed(client):
    resp = client.options("/forecast", headers={
        "Origin": "https://attacker.example",
        "Access-Control-Request-Method": "GET",
    })
    assert resp.headers.get("Access-Control-Allow-Origin") is None


def test_preflight_from_dashboard_origin_is_allowed(api, client):
    resp = client.options("/forecast", headers={
        "Origin": api.DASHBOARD_ORIGIN,
        "Access-Control-Request-Method": "GET",
    })
    assert resp.headers.get("Access-Control-Allow-Origin") == api.DASHBOARD_ORIGIN


def test_wildcard_cors_is_gone(client):
    resp = client.get("/health", headers={"Origin": "https://attacker.example"})
    assert resp.headers.get("Access-Control-Allow-Origin") != "*"


def test_burst_past_limit_is_429():
    # Fresh app so this module's other tests do not share the counter.
    api = load_api("forecast_api_burst")
    client = api.app.test_client()
    limit = int(api.RATE_LIMIT.split()[0])
    statuses = [client.get("/forecast/range?hours=1", headers=_bearer(DEV_KEY)).status_code
                for _ in range(limit + 5)]
    assert 429 not in statuses[:limit], "limit tripped early"
    assert statuses[-1] == 429
    body = client.get("/forecast/range?hours=1", headers=_bearer(DEV_KEY)).get_json()
    assert "rate limit" in body["error"]


def test_limit_is_per_key_not_global():
    api = load_api("forecast_api_perkey")
    client = api.app.test_client()
    limit = int(api.RATE_LIMIT.split()[0])
    for _ in range(limit + 1):
        client.get("/forecast/range?hours=1", headers=_bearer(DEV_KEY))
    assert client.get("/forecast/range?hours=1", headers=_bearer(DEV_KEY)).status_code == 429
    # A caller without a valid key is keyed by address: a separate bucket,
    # so the exhausted key's budget does not apply to it (it is still a 401).
    assert client.get("/forecast/range?hours=1").status_code == 401
    # Probe routes are exempt from the limit altogether.
    assert client.get("/health").status_code != 429


def test_rejected_requests_are_counted_too():
    # Rotating wrong tokens from one address must not escape the limit: an
    # unverified token never gets its own bucket, and the limiter runs before
    # the gate answers, so a flood of 401s becomes a 429.
    api = load_api("forecast_api_flood")
    client = api.app.test_client()
    limit = int(api.RATE_LIMIT.split()[0])
    statuses = [client.get("/forecast", headers=_bearer(f"wrong-{i}")).status_code
                for i in range(limit + 5)]
    assert set(statuses[:limit]) == {401}
    assert 429 in statuses[limit:]


def test_non_ascii_token_is_401_not_500(client):
    resp = client.get("/forecast", headers={"Authorization": "Bearer caf\u00e9"})
    assert resp.status_code == 401
    resp = client.get("/forecast", headers={"Authorization": "Bearer \u00fcber-\u2603"})
    assert resp.status_code == 401


def test_probe_routes_are_exempt_from_the_limit():
    api = load_api("forecast_api_exempt")
    client = api.app.test_client()
    limit = int(api.RATE_LIMIT.split()[0])
    statuses = {client.get("/health").status_code for _ in range(limit + 5)}
    assert 429 not in statuses
