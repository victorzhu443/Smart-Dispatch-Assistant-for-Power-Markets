import requests
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET
from secprim import MalformedInput, finite_float

MAX_RANGE_HOURS = 168
NUMERIC_PARAMS = {
    # name -> (lo, hi); None means unbounded. Checked here so a malformed
    # value is a 400 from Django and never leaves this process (RC-1).
    "marginal_cost": (None, None),
    "hours": (1, MAX_RANGE_HOURS),
}


def _auth_headers() -> dict:
    """Django is the forecast API's only client; it presents the key (RC-6)."""
    return {"Authorization": f"Bearer {settings.FORECAST_API_KEY}"}


def _forward_params(request, keys) -> dict:
    """Whitelist and validate the query parameters that are proxied upstream."""
    params = {key: request.GET[key] for key in keys if request.GET.get(key)}
    for key, (lo, hi) in NUMERIC_PARAMS.items():
        if key in params:
            finite_float(params[key], key, lo=lo, hi=hi)
    return params


def _proxy(url: str, params: dict, timeout: int) -> JsonResponse:
    try:
        r = requests.get(url, params=params, headers=_auth_headers(),
                         timeout=timeout)
        return JsonResponse(r.json(), status=r.status_code, safe=False)
    except requests.RequestException as e:
        return JsonResponse({
            "error": "forecast service unreachable",
            "detail": str(e),
            "remedy": "start it with: python backend/phase_5_1_forecast_api.py",
        }, status=503)


def dashboard(request):
    return render(request, "dashboard.html")


@require_GET
def api_forecast(request):
    """Proxy a single-hour forecast."""
    try:
        params = _forward_params(request, ("timestamp", "marginal_cost"))
    except MalformedInput as e:
        return JsonResponse({"error": str(e)}, status=400)
    return _proxy(settings.FORECAST_API_URL, params, timeout=20)


@require_GET
def api_forecast_range(request):
    """Proxy a multi-hour forecast band, which is what the chart draws."""
    try:
        params = _forward_params(request, ("start", "hours", "marginal_cost"))
    except MalformedInput as e:
        return JsonResponse({"error": str(e)}, status=400)
    return _proxy(settings.FORECAST_RANGE_URL, params, timeout=30)
