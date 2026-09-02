from django.urls import path
from dashboard import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    # server-side proxies (avoid CORS)
    path("api/forecast", views.api_forecast, name="api_forecast"),
    path("api/forecast/range", views.api_forecast_range, name="api_forecast_range"),
]
