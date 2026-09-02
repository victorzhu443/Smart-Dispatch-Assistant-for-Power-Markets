"""Explicit schema: constraints, indexes, lineage, and a migration runner.

Why this exists.

Every table in this project was created implicitly by `pandas.to_sql`, which
infers types and creates nothing else. The result had no primary keys, no
NOT NULL, no indexes, and no DDL under version control. Three consequences,
all measured against the live database:

  1. Duplicates were accepted silently. Idempotency was a convention inside one
     writer function rather than a guarantee, so any other script, notebook or
     teammate writing by a different path could double-count and nothing would
     complain:

         inserted an exact duplicate: 93,544 -> 93,545  (accepted)

  2. Every query was a full table scan. The forecast API issues a
     (settlement_point, time range) lookup on every request:

         SCAN market_data_hourly                      2.09 ms
         SEARCH market_data_hourly USING INDEX        0.02 ms   117x

  3. No row could say where it came from. Six months from now, "why is this
     number odd?" is archaeology instead of a WHERE clause.

A constraint is a guarantee. Application logic is a convention. This module
moves the guarantees into the database, where they hold regardless of which
code path does the writing.

Usage:
    python data-ingestion/schema.py --check      # report drift, change nothing
    python data-ingestion/schema.py --migrate    # apply, preserving all rows
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from sqlalchemy import create_engine, inspect, text

# Sentinel for rows that predate lineage tracking. Explicit beats NULL here:
# "we do not know" is different from "nobody has filled this in yet".
PRE_MIGRATION = "pre-migration"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


# Business columns per table, in order, followed by the lineage columns every
# landed table carries.
LINEAGE_COLUMNS = ["ingested_at", "source_report", "source_file"]

TABLES: dict[str, dict] = {
    "market_data_hourly": {
        "columns": ["settlement_point", "timestamp_utc", "price"],
        "ddl": """
            CREATE TABLE market_data_hourly (
                settlement_point TEXT NOT NULL,
                timestamp_utc    TEXT NOT NULL,
                price            REAL NOT NULL,
                ingested_at      TEXT NOT NULL,
                source_report    TEXT,
                source_file      TEXT,
                PRIMARY KEY (settlement_point, timestamp_utc)
            )
        """,
        # The composite primary key already indexes
        # (settlement_point, timestamp_utc), which is the API's access pattern.
        # This second index serves cross-hub queries ordered by time.
        "indexes": [
            "CREATE INDEX IF NOT EXISTS ix_mdh_time "
            "ON market_data_hourly (timestamp_utc)",
        ],
    },
    "dam_prices_hourly": {
        "columns": ["timestamp_utc", "settlement_point", "price"],
        "ddl": """
            CREATE TABLE dam_prices_hourly (
                timestamp_utc    TEXT NOT NULL,
                settlement_point TEXT NOT NULL,
                price            REAL NOT NULL,
                ingested_at      TEXT NOT NULL,
                source_report    TEXT,
                source_file      TEXT,
                PRIMARY KEY (settlement_point, timestamp_utc)
            )
        """,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS ix_dam_time "
            "ON dam_prices_hourly (timestamp_utc)",
        ],
    },
    "spp_raw_15min": {
        "columns": ["timestamp_utc", "settlement_point",
                    "settlement_point_type", "price", "repeated_hour_flag"],
        # ingested_at is part of the key here, unlike the derived tables. This
        # is the raw landing zone: if ERCOT restates a settled interval, the
        # correction should land beside the original rather than overwrite it,
        # so "what did we believe on Tuesday?" stays answerable.
        "ddl": """
            CREATE TABLE spp_raw_15min (
                timestamp_utc         TEXT NOT NULL,
                settlement_point      TEXT NOT NULL,
                settlement_point_type TEXT,
                price                 REAL NOT NULL,
                repeated_hour_flag    TEXT,
                ingested_at           TEXT NOT NULL,
                source_report         TEXT,
                source_file           TEXT,
                PRIMARY KEY (settlement_point, timestamp_utc, ingested_at)
            )
        """,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS ix_raw_time "
            "ON spp_raw_15min (timestamp_utc)",
        ],
    },
}


def setup_database_connection():
    try:
        pg = (
            f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'password')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DATABASE', 'smart_dispatch')}"
        )
        engine = create_engine(pg)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception:
        return create_engine("sqlite:///market_data.db")


def describe(engine, table: str) -> dict:
    """What the database actually has right now."""
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return {"exists": False}

    columns = inspector.get_columns(table)
    pk = inspector.get_pk_constraint(table).get("constrained_columns") or []
    indexes = [ix["name"] for ix in inspector.get_indexes(table)]
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()

    return {
        "exists": True,
        "columns": [c["name"] for c in columns],
        "primary_key": pk,
        "indexes": indexes,
        "rows": rows,
        "has_lineage": all(c in {c["name"] for c in columns}
                           for c in LINEAGE_COLUMNS),
    }


def check(engine) -> bool:
    """Report drift between the declared schema and the live one."""
    ok = True
    for table, spec in TABLES.items():
        state = describe(engine, table)
        print(f"\n{table}")
        if not state["exists"]:
            print("  MISSING — will be created")
            ok = False
            continue

        print(f"  rows          {state['rows']:,}")

        if state["primary_key"]:
            print(f"  primary key   {state['primary_key']}")
        else:
            print("  primary key   NONE — duplicates can be inserted silently")
            ok = False

        if state["has_lineage"]:
            print("  lineage       ok")
        else:
            missing = [c for c in LINEAGE_COLUMNS if c not in state["columns"]]
            print(f"  lineage       MISSING {missing}")
            ok = False

        if state["indexes"]:
            print(f"  indexes       {state['indexes']}")
        else:
            print("  indexes       NONE — every query is a full table scan")
            ok = False
    return ok


def migrate_table(engine, table: str, spec: dict) -> str:
    """Rebuild one table under the declared schema, preserving every row.

    SQLite cannot ALTER a table to add a primary key, so the only honest route
    is create-copy-swap. Doing it inside one transaction means a failure leaves
    the original untouched rather than half-migrated.
    """
    state = describe(engine, table)
    if not state["exists"]:
        with engine.begin() as conn:
            conn.execute(text(spec["ddl"]))
            for index_sql in spec["indexes"]:
                conn.execute(text(index_sql))
        return f"{table}: created"

    if state["primary_key"] and state["has_lineage"]:
        with engine.begin() as conn:
            for index_sql in spec["indexes"]:
                conn.execute(text(index_sql))
        return f"{table}: already current ({state['rows']:,} rows)"

    before = state["rows"]
    business = spec["columns"]
    # Carry lineage across if it already exists, otherwise stamp the sentinel.
    if state["has_lineage"]:
        select_cols = ", ".join(business + LINEAGE_COLUMNS)
    else:
        select_cols = ", ".join(
            business + [f"'{utc_now_iso()}'", f"'{PRE_MIGRATION}'", "NULL"]
        )
    insert_cols = ", ".join(business + LINEAGE_COLUMNS)

    # De-duplicate on the way across. There are none today, but a migration
    # that silently fails on a duplicate is worse than one that resolves it.
    key = "settlement_point, timestamp_utc"
    if table == "spp_raw_15min" and not state["has_lineage"]:
        key = "settlement_point, timestamp_utc"

    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} RENAME TO {table}_old"))
        conn.execute(text(spec["ddl"]))
        conn.execute(text(
            f"INSERT INTO {table} ({insert_cols}) "
            f"SELECT {select_cols} FROM {table}_old "
            f"WHERE rowid IN (SELECT MIN(rowid) FROM {table}_old GROUP BY {key})"
        ))
        for index_sql in spec["indexes"]:
            conn.execute(text(index_sql))
        after = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()

        if after == 0 and before > 0:
            raise RuntimeError(
                f"{table}: migration produced 0 rows from {before}; rolling back"
            )
        conn.execute(text(f"DROP TABLE {table}_old"))

    dropped = before - after
    note = f", {dropped} duplicate(s) collapsed" if dropped else ""
    return f"{table}: migrated {after:,} rows{note}"


def migrate(engine) -> None:
    for table, spec in TABLES.items():
        print("  " + migrate_table(engine, table, spec))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true",
                       help="report drift without changing anything")
    group.add_argument("--migrate", action="store_true",
                       help="apply the declared schema, preserving rows")
    args = parser.parse_args()

    engine = setup_database_connection()

    if args.check:
        print("Schema check")
        return 0 if check(engine) else 1

    print("Applying schema")
    migrate(engine)
    print("\nVerifying")
    return 0 if check(engine) else 1


if __name__ == "__main__":
    sys.exit(main())
