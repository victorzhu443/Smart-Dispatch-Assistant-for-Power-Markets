from django.urls import path
from dashboard import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("chat/", views.chat, name="chat"),
    # server-side proxies (avoid CORS)
    path("api/forecast", views.api_forecast, name="api_forecast"),
    path("api/query", views.api_query, name="api_query"),
]
