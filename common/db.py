"""Shared database connection.

This function was copy-pasted verbatim into twelve files. Every one of them
tried PostgreSQL from the same five environment variables and fell back to the
same SQLite file, which meant a change to connection handling was twelve edits
and a chance to miss one.

It stayed duplicated because it could not be shared: the packages were named
`data-ingestion`, `feature-engineering` and `forecasting-model`, and a hyphen
is not valid in a Python identifier, so none of them was importable. Every
consumer either duplicated the code or loaded modules by file path through
importlib. Renaming the directories to underscores is what makes this module
possible.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parent.parent
SQLITE_PATH = REPO_ROOT / "market_data.db"


def postgres_url() -> str:
    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'password')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DATABASE', 'smart_dispatch')}"
    )


def setup_database_connection(*, quiet: bool = False):
    """Connect to PostgreSQL if it is reachable, else the local SQLite file.

    The SQLite path is resolved from the repository root rather than the
    working directory, so a script behaves the same whether it is run from the
    root or from its own package.
    """
    try:
        engine = create_engine(postgres_url())
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        if not quiet:
            print("PostgreSQL connection successful")
        return engine
    except Exception:
        if not quiet:
            print(f"PostgreSQL not available, using SQLite ({SQLITE_PATH.name})")
        return create_engine(f"sqlite:///{SQLITE_PATH}")
