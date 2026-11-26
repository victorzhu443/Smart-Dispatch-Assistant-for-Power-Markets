"""Load the Flask forecast API as a fresh module (fresh app, fresh limiter)."""
import importlib.util
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
API_PATH = REPO_ROOT / "backend" / "phase_5_1_forecast_api.py"


def load_api(name: str = "forecast_api_under_test"):
    warnings.filterwarnings("ignore")
    spec = importlib.util.spec_from_file_location(name, API_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
