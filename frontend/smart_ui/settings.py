import os
from pathlib import Path

from secprim import require_secret

try:  # secprim >= 0.1.1: fail-closed label set (staging/qa/typos are not dev)
    from secprim.secrets import is_dev
except ImportError:  # 0.1.0 shim; remove when the pin is bumped
    def is_dev():
        return (os.environ.get("APP_ENV") or "dev").strip().lower() in {
            "dev", "development", "test", "testing", "local"}

BASE_DIR = Path(__file__).resolve().parent.parent

# Secrets come from the environment (SECURITY.md RC-5). With APP_ENV unset,
# "dev" or "test", the placeholder below is used so `runserver` behaves exactly
# as before with zero configuration. With APP_ENV=production a missing
# DJANGO_SECRET_KEY raises secprim.MissingSecret at import, so a deployment can
# never boot on the dev key by accident.
SECRET_KEY = require_secret("DJANGO_SECRET_KEY", dev_default="dev-only-smart-dispatch")

# DEBUG follows the environment: on only for an explicit dev/test label (or
# none), off for production AND for anything unrecognised such as "staging",
# unless DJANGO_DEBUG=1/0 says otherwise explicitly.
DEBUG = ((os.environ.get("DJANGO_DEBUG") == "1") if "DJANGO_DEBUG" in os.environ
         else is_dev())
# "testserver" is the host Django's own test client sends; without it
# every request in a test returns 400 before reaching a view.
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "dashboard",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "smart_ui.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "dashboard" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "smart_ui.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "frontend.db",
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# --- Static files ---
STATIC_URL = "/static/"                           # fixed leading slash
STATICFILES_DIRS = [BASE_DIR / "static"]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Backend API endpoints (override via env vars if needed) ---
FORECAST_API_URL = os.getenv("FORECAST_API_URL", "http://127.0.0.1:5001/forecast")
FORECAST_RANGE_URL = os.getenv("FORECAST_RANGE_URL",
                               "http://127.0.0.1:5001/forecast/range")

# The forecast API requires a bearer token on /forecast* (RC-6). Django is its
# only client and holds the key; the browser only ever talks to Django. The dev
# default matches the API's own, so the local flow needs no configuration.
FORECAST_API_KEY = require_secret("FORECAST_API_KEY",
                                  dev_default="dev-only-forecast-key")
