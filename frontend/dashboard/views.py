import json
import requests
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt

def dashboard(request):
    return render(request, "dashboard.html")

def chat(request):
    return render(request, "chat.html")

@require_GET
def api_forecast(request):
    """Proxy a single-hour forecast."""
    params = {
        key: request.GET[key]
        for key in ("timestamp", "marginal_cost")
        if request.GET.get(key)
    }
    try:
        r = requests.get(settings.FORECAST_API_URL, params=params, timeout=20)
        return JsonResponse(r.json(), status=r.status_code, safe=False)
    except requests.RequestException as e:
        return JsonResponse({
            "error": "forecast service unreachable",
            "detail": str(e),
            "remedy": "start it with: python backend/phase_5_1_forecast_api.py",
        }, status=503)


@require_GET
def api_forecast_range(request):
    """Proxy a multi-hour forecast band, which is what the chart draws."""
    params = {
        key: request.GET[key]
        for key in ("start", "hours", "marginal_cost")
        if request.GET.get(key)
    }
    try:
        r = requests.get(settings.FORECAST_RANGE_URL, params=params, timeout=30)
        return JsonResponse(r.json(), status=r.status_code, safe=False)
    except requests.RequestException as e:
        return JsonResponse({
            "error": "forecast service unreachable",
            "detail": str(e),
            "remedy": "start it with: python backend/phase_5_1_forecast_api.py",
        }, status=503)

@csrf_exempt
@require_POST
def api_query(request):
    try:
        data = json.loads(request.body.decode("utf-8") or "{}")
        payload = {"question": data.get("question", "")}
        r = requests.post(settings.QUERY_API_URL, json=payload, timeout=40)
        return JsonResponse(r.json(), status=r.status_code, safe=False)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
