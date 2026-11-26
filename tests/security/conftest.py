"""Make the project root importable for the security suite.

The security tests run under a dedicated venv (`.venv-sec`) that is not an
editable install of the project, so — unlike the main suite invoked from the
repo root — `import data_ingestion...` is not guaranteed to resolve. Prepend the
project root so the fuzz targets import the real parsers.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
