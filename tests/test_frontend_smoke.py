"""Smoke tests for the Django UI.

PRD test case 6.1 is "localhost:8000 loads basic UI". It did not pass: the
project referenced a "smartui" module while the package directory is
"smart_ui", and frontend/dashboard/ had no __init__.py so Django treated it as
a namespace package and refused to start. These tests fail if either returns.

Skipped when Django is not installed, so the rest of the suite still runs.

Run with:  pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest

django = pytest.importorskip("django", reason="Django not installed")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@pytest.fixture(scope="module")
def client():
    """Configure Django once for this module and hand back a test client."""
    import os

    if str(FRONTEND) not in sys.path:
        sys.path.insert(0, str(FRONTEND))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "smart_ui.settings")

    from django.apps import apps

    if not apps.ready:
        django.setup()

    from django.test import Client

    return Client()


def test_django_configuration_is_valid():
    """The check the naming bug used to fail with ModuleNotFoundError."""
    import os

    if str(FRONTEND) not in sys.path:
        sys.path.insert(0, str(FRONTEND))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "smart_ui.settings")

    from django.apps import apps

    if not apps.ready:
        django.setup()

    from django.core.management import call_command

    call_command("check")  # raises SystemCheckError on any issue


def test_dashboard_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    body = response.content.decode()
    assert "Smart Dispatch Assistant" in body
    assert "chart" in body.lower(), "dashboard should render its forecast chart"


def test_chat_page_loads(client):
    response = client.get("/chat/")

    assert response.status_code == 200
    assert "Smart Dispatch Assistant" in response.content.decode()


def test_dashboard_app_is_a_real_package():
    """A namespace package makes Django raise ImproperlyConfigured."""
    assert (FRONTEND / "dashboard" / "__init__.py").exists()


def test_settings_reference_the_actual_package_name():
    """Guard against the smartui/smart_ui mismatch returning."""
    settings_text = (FRONTEND / "smart_ui" / "settings.py").read_text()

    assert "smart_ui.urls" in settings_text
    assert "smart_ui.wsgi" in settings_text
    assert "smartui." not in settings_text.replace("smart_ui.", "")


def test_static_files_directory_exists():
    """STATICFILES_DIRS pointed at a directory that was never created."""
    import os

    if str(FRONTEND) not in sys.path:
        sys.path.insert(0, str(FRONTEND))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "smart_ui.settings")

    from django.apps import apps

    if not apps.ready:
        django.setup()

    from django.conf import settings

    for directory in settings.STATICFILES_DIRS:
        assert Path(directory).is_dir(), f"{directory} does not exist"
