"""Alerting: notice when the pipeline stops, before the source forgets.

The failure this guards against is specific and permanent. ERCOT's interval
feed keeps a rolling window of about seven days. If the incremental job breaks
on a Friday and nobody notices, the data it needed has aged out of the source
by the following week. It is not recoverable at any price.

Nothing was watching. The API reported `degraded` if you asked it, but nobody
asks at 3am, and a cron job that fails silently looks exactly like one that
succeeded.

Three distinct signals, because they fail differently:

    data staleness    the newest row is too old -> ingestion is not landing
    job silence       no quality check has been recorded recently -> the job
                      is not running at all, which stale data alone cannot
                      distinguish from a source outage
    check failures    recent validations are failing -> the job runs but the
                      data is wrong

Exit code is 1 when anything is wrong, so cron mails the output. That is real
alerting for a single-machine pipeline; ALERT_WEBHOOK_URL adds a push when you
outgrow it.

Usage:
    python -m data_ingestion.monitor              # human-readable, exit 1 if bad
    python -m data_ingestion.monitor --quiet      # only print when something is wrong

Cron:
    0 * * * * cd /path/to/repo && python -m data_ingestion.monitor --quiet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import text

from common.db import setup_database_connection
from data_ingestion import quality

# If no quality check has been recorded in this long, the job is not running.
# Distinct from data staleness: a source outage leaves data stale while the job
# still runs and records passes.
MAX_SILENCE_HOURS = float(os.getenv("MAX_JOB_SILENCE_HOURS", "3"))

# How far back to look when judging whether checks are failing.
FAILURE_WINDOW_HOURS = 24

WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")


def last_successful_run(engine) -> pd.Timestamp | None:
    try:
        with engine.connect() as conn:
            value = conn.execute(text(
                f"SELECT MAX(ran_at) FROM {quality.RUNS_TABLE} WHERE passed = 1"
            )).scalar()
    except Exception:
        return None
    return pd.to_datetime(value, utc=True, format="mixed") if value else None


def recent_failures(engine, hours: float = FAILURE_WINDOW_HOURS) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)) \
        .strftime("%Y-%m-%d %H:%M:%S%z")
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                f"SELECT ran_at, stage, failure FROM {quality.RUNS_TABLE} "
                f"WHERE passed = 0 AND ran_at >= :cutoff ORDER BY ran_at DESC"
            ), {"cutoff": cutoff}).fetchall()
    except Exception:
        return []
    return [{"ran_at": r[0], "stage": r[1], "failure": r[2]} for r in rows]


def collect(engine) -> list[dict]:
    """Every problem worth waking someone for, as structured alerts."""
    alerts = []

    for row in quality.table_freshness(engine):
        if not row.get("exists"):
            alerts.append({
                "signal": "table_missing", "table": row["table"],
                "detail": f"{row['table']} does not exist",
            })
        elif row.get("stale"):
            age = row.get("age_hours")
            alerts.append({
                "signal": "data_stale", "table": row["table"],
                "detail": (f"{row['table']} newest row is "
                           f"{age:,.1f}h old, budget {row['budget_hours']:.0f}h"
                           if age is not None else f"{row['table']} is empty"),
            })

    last = last_successful_run(engine)
    if last is None:
        alerts.append({
            "signal": "job_silent", "table": None,
            "detail": ("no successful quality check has ever been recorded; "
                       "has ingestion run?"),
        })
    else:
        silence = (datetime.now(timezone.utc)
                   - last.to_pydatetime()).total_seconds() / 3600
        if silence > MAX_SILENCE_HOURS:
            alerts.append({
                "signal": "job_silent", "table": None,
                "detail": (f"no successful check in {silence:,.1f}h "
                           f"(limit {MAX_SILENCE_HOURS:.0f}h); the job may not "
                           f"be running"),
            })

    for failure in recent_failures(engine):
        alerts.append({
            "signal": "check_failed", "table": None,
            "detail": (f"{failure['stage']} failed at {failure['ran_at'][:16]}: "
                       f"{(failure['failure'] or '')[:120]}"),
        })

    return alerts


def notify(alerts: list[dict]) -> None:
    """Push to a webhook when one is configured. Never fatal."""
    if not WEBHOOK_URL or not alerts:
        return
    try:
        import requests
        requests.post(WEBHOOK_URL, json={
            "source": "smart-dispatch-pipeline",
            "at": datetime.now(timezone.utc).isoformat(),
            "alert_count": len(alerts),
            "alerts": alerts,
        }, timeout=10)
    except Exception as exc:
        # An alerting failure must not mask the alert itself, which is already
        # on stdout and in the exit code.
        print(f"  (webhook delivery failed: {exc})", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true",
                        help="print nothing when healthy (for cron)")
    parser.add_argument("--json", action="store_true",
                        help="emit alerts as JSON")
    args = parser.parse_args()

    engine = setup_database_connection(quiet=True)
    alerts = collect(engine)

    if args.json:
        print(json.dumps({"healthy": not alerts, "alerts": alerts}, indent=2))
    elif alerts:
        print(f"PIPELINE ALERT — {len(alerts)} problem(s)\n", file=sys.stderr)
        for alert in alerts:
            print(f"  [{alert['signal']}] {alert['detail']}", file=sys.stderr)
        print("\nRemedy: make ingest && make features", file=sys.stderr)
    elif not args.quiet:
        print("Pipeline healthy: data fresh, job running, checks passing.")

    notify(alerts)
    return 1 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
