"""Data quality: record every check, and detect derived tables going stale.

Two gaps this closes.

Validation results were discarded. `validate()` either passed silently or
aborted the run, so nobody could ask "has the null rate been creeping up for
three weeks?" A threshold catches a cliff; only a trend catches a drift, and a
trend needs history. Every check now writes one row to `data_quality_runs`.

Derived tables went stale invisibly. The `features` table covered
2024-01-02 to 2026-01-01 while the raw data ran to 2026-09-02 -- eight months
behind, with nothing to say so. That is the same failure as the model staleness
bug and the data staleness bug before it: a downstream artifact silently
describing a world that has moved on. The pattern repeats at every layer of a
pipeline, so it is worth having one place that checks all of them.

Usage:
    python data-ingestion/quality.py --report      # freshness + recent history
    python data-ingestion/quality.py --check       # exit 1 if anything is stale
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

import schema as schema_module

RUNS_TABLE = "data_quality_runs"

RUNS_DDL = """
CREATE TABLE IF NOT EXISTS data_quality_runs (
    ran_at        TEXT    NOT NULL,
    stage         TEXT    NOT NULL,
    table_name    TEXT,
    rows_checked  INTEGER,
    passed        INTEGER NOT NULL,
    failure       TEXT,
    duration_ms   REAL
)
"""

# How far behind the raw data each derived table may fall before it is stale.
# Generous, because these rebuild on a schedule rather than continuously.
STALENESS_BUDGET_HOURS = {
    "spp_raw_15min": 6,
    "market_data_hourly": 6,
    "dam_prices_hourly": 30,      # published once a day, ahead of delivery
    "features": 48,
}

# Which column carries the event time in each table.
TIME_COLUMN = {
    "spp_raw_15min": "timestamp_utc",
    "market_data_hourly": "timestamp_utc",
    "dam_prices_hourly": "timestamp_utc",
    "features": "target_time",
}


def ensure_runs_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(RUNS_DDL))


def record(engine, *, stage: str, table: str | None, rows: int | None,
           passed: bool, failure: str | None = None,
           duration_ms: float | None = None) -> None:
    """Append one row describing a check that just ran.

    Deliberately never raises: a failure to record must not take down an
    ingestion that is otherwise healthy.
    """
    try:
        ensure_runs_table(engine)
        with engine.begin() as conn:
            conn.execute(text(
                f"INSERT INTO {RUNS_TABLE} "
                "(ran_at, stage, table_name, rows_checked, passed, failure, "
                " duration_ms) "
                "VALUES (:ran_at, :stage, :table, :rows, :passed, :failure, :ms)"
            ), {
                "ran_at": datetime.now(timezone.utc)
                          .strftime("%Y-%m-%d %H:%M:%S%z"),
                "stage": stage, "table": table, "rows": rows,
                "passed": 1 if passed else 0, "failure": failure,
                "ms": duration_ms,
            })
    except Exception as exc:  # pragma: no cover - best effort by design
        print(f"  (could not record quality run: {exc})")


def timed_validate(engine, validate_fn, df, stage: str,
                   table: str | None = None):
    """Run a validation, record the outcome either way, then re-raise on failure.

    Recording only successes would make the history useless: the interesting
    rows are the failures.
    """
    started = time.perf_counter()
    try:
        validate_fn(df, stage)
    except Exception as exc:
        record(engine, stage=stage, table=table, rows=len(df), passed=False,
               failure=str(exc)[:500],
               duration_ms=(time.perf_counter() - started) * 1000)
        raise
    record(engine, stage=stage, table=table, rows=len(df), passed=True,
           duration_ms=(time.perf_counter() - started) * 1000)


def table_freshness(engine) -> list[dict]:
    """How far behind now each table's newest event time is."""
    now = datetime.now(timezone.utc)
    rows = []
    for table, column in TIME_COLUMN.items():
        try:
            with engine.connect() as conn:
                latest = conn.execute(
                    text(f"SELECT MAX({column}) FROM {table}")
                ).scalar()
                count = conn.execute(
                    text(f"SELECT COUNT(*) FROM {table}")
                ).scalar()
        except Exception:
            rows.append({"table": table, "exists": False})
            continue

        if latest is None:
            rows.append({"table": table, "exists": True, "rows": 0,
                         "latest": None, "age_hours": None, "stale": True})
            continue

        latest_ts = pd.to_datetime(latest, utc=True, format="mixed")
        age = (now - latest_ts.to_pydatetime()).total_seconds() / 3600
        budget = STALENESS_BUDGET_HOURS[table]
        # Day-ahead prices are published before delivery, so the newest event
        # time is legitimately in the future. Negative age means "ahead of
        # schedule", which is the healthiest state this table has.
        rows.append({
            "table": table, "exists": True, "rows": count,
            "latest": latest_ts, "age_hours": age,
            "budget_hours": budget, "stale": age > budget,
        })
    return rows


def recent_runs(engine, limit: int = 10) -> pd.DataFrame:
    try:
        return pd.read_sql(
            text(f"SELECT ran_at, stage, table_name, rows_checked, passed, "
                 f"failure FROM {RUNS_TABLE} ORDER BY ran_at DESC LIMIT :n"),
            engine, params={"n": limit},
        )
    except Exception:
        return pd.DataFrame()


def report(engine) -> bool:
    """Print freshness and recent history. Returns False if anything is stale."""
    print("Table freshness\n")
    header = (f"{'table':<22}{'rows':>10}{'latest event':>22}"
              f"{'age':>13}{'budget':>9}  state")
    print(header)
    print("-" * len(header))

    all_fresh = True
    for row in table_freshness(engine):
        if not row["exists"]:
            print(f"{row['table']:<22}{'—':>10}{'missing':>22}")
            all_fresh = False
            continue
        if row["latest"] is None:
            print(f"{row['table']:<22}{0:>10}{'empty':>22}")
            all_fresh = False
            continue

        if row["stale"]:
            all_fresh = False
            state = "STALE"
        elif row["age_hours"] < 0:
            state = "ok (ahead)"
        else:
            state = "ok"

        age = row["age_hours"]
        age_text = f"{abs(age):.1f}h {'ahead' if age < 0 else 'old'}"
        print(f"{row['table']:<22}{row['rows']:>10,}"
              f"{row['latest'].strftime('%Y-%m-%d %H:%M'):>22}"
              f"{age_text:>13}{row['budget_hours']:>8.0f}h  {state}")

    runs = recent_runs(engine)
    print("\nRecent quality checks")
    if runs.empty:
        print("  none recorded yet — they accumulate as ingestion runs")
    else:
        for _, r in runs.iterrows():
            mark = "pass" if r["passed"] else "FAIL"
            note = f"  {r['failure'][:60]}" if r["failure"] else ""
            print(f"  {r['ran_at'][:16]}  {mark:<5}{r['stage']:<22}"
                  f"{r['rows_checked'] or 0:>8,} rows{note}")

        total = len(runs)
        failed = int((runs["passed"] == 0).sum())
        print(f"\n  last {total} checks: {total - failed} passed, {failed} failed")

    return all_fresh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--report", action="store_true",
                       help="print freshness and recent check history")
    group.add_argument("--check", action="store_true",
                       help="exit 1 if any table is stale (for cron/alerting)")
    args = parser.parse_args()

    engine = schema_module.setup_database_connection()
    fresh = report(engine)

    if args.check and not fresh:
        print("\nAt least one table is stale. Run 'make ingest' and "
              "'make features'.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
