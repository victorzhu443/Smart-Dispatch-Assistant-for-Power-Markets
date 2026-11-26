"""Secrets from the environment, never a production default (SECURITY.md RC-5).

The Django settings module is imported in a subprocess with a controlled
environment so the assertions hold regardless of what this shell exports, and
so a MissingSecret at import does not poison the test process.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "frontend"

PROBE = """
import json, sys
try:
    import smart_ui.settings as s
    print(json.dumps({"ok": True, "secret": s.SECRET_KEY, "debug": s.DEBUG,
                      "api_key": s.FORECAST_API_KEY}))
except Exception as e:
    print(json.dumps({"ok": False, "type": type(e).__name__, "msg": str(e)}))
"""


def _import_settings(env_overrides: dict) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("APP_ENV", "DJANGO_SECRET_KEY", "DJANGO_DEBUG", "FORECAST_API_KEY")}
    env.update(env_overrides)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(FRONTEND), str(REPO_ROOT),
                                                     env.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True,
                         text=True, check=True, cwd=str(REPO_ROOT))
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_dev_gets_the_documented_defaults():
    r = _import_settings({})
    assert r["ok"], r
    assert r["secret"] == "dev-only-smart-dispatch"
    assert r["api_key"] == "dev-only-forecast-key"
    assert r["debug"] is True


@pytest.mark.parametrize("app_env", ["production", "prod", "PRODUCTION"])
def test_production_without_secret_refuses_to_import(app_env):
    r = _import_settings({"APP_ENV": app_env})
    assert not r["ok"]
    assert r["type"] == "MissingSecret"
    assert "DJANGO_SECRET_KEY" in r["msg"]


def test_production_with_django_secret_still_needs_the_api_key():
    r = _import_settings({"APP_ENV": "production", "DJANGO_SECRET_KEY": "x" * 50})
    assert not r["ok"]
    assert r["type"] == "MissingSecret"
    assert "FORECAST_API_KEY" in r["msg"]


def test_production_with_both_secrets_boots_with_debug_off():
    r = _import_settings({"APP_ENV": "production", "DJANGO_SECRET_KEY": "x" * 50,
                          "FORECAST_API_KEY": "k" * 32})
    assert r["ok"], r
    assert r["secret"] == "x" * 50
    assert r["debug"] is False


def test_explicit_debug_override_wins():
    assert _import_settings({"DJANGO_DEBUG": "0"})["debug"] is False
    r = _import_settings({"APP_ENV": "production", "DJANGO_SECRET_KEY": "x" * 50,
                          "FORECAST_API_KEY": "k" * 32, "DJANGO_DEBUG": "1"})
    assert r["debug"] is True


@pytest.mark.parametrize("app_env", ["staging", "qa", "prodution", "live"])
def test_unrecognised_label_is_not_dev(app_env):
    """An unrecognised APP_ENV must never inherit development behaviour."""
    import importlib
    r = _import_settings({"APP_ENV": app_env})
    if r["ok"]:
        # secprim 0.1.0: require_secret still hands out the dev default for
        # non-prod labels (fixed upstream in 0.1.1, fail-closed). DEBUG is
        # derived from is_dev() locally, so it is already off.
        assert r["debug"] is False, r
        assert not hasattr(importlib.import_module("secprim.secrets"), "is_dev"), (
            "toolkit has is_dev(): require_secret should now fail closed on "
            f"APP_ENV={app_env!r}; drop this branch")
    else:
        assert r["type"] == "MissingSecret"


def test_dev_default_is_never_used_as_a_production_secret():
    r = _import_settings({"APP_ENV": "production", "DJANGO_SECRET_KEY": "x" * 50,
                          "FORECAST_API_KEY": "k" * 32})
    assert r["secret"] != "dev-only-smart-dispatch"
