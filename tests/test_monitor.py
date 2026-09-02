"""Tests for pipeline alerting.

The three signals are deliberately separate, and the tests assert they stay
separate. Stale data alone cannot distinguish "the source is down" from "our
job is not running" — the first is somebody else's outage, the second is ours,
and they need different responses.

Run with:  pytest tests/ -v
"""
import importlib
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

schema = importlib.import_module("data_ingestion.schema")
quality = importlib.import_module("data_ingestion.quality")
monitor = importlib.import_module("data_ingestion.monitor")


def _stamp(hours_ago: float) -> str:
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return ts.strftime("%Y-%m-%d %H:%M:%S%z")


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    schema.migrate(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE features (window_id INTEGER, target_time TEXT)"
        ))
    quality.ensure_runs_table(engine)
    return engine


def _make_healthy(engine):
    """Fresh rows in every table plus a recent passing check."""
    with engine.begin() as conn:
        for table in ("market_data_hourly", "dam_prices_hourly"):
            conn.execute(text(
                f"INSERT INTO {table} (settlement_point, timestamp_utc, price, "
                f"ingested_at) VALUES ('HB_HOUSTON', :ts, 30.0, :ts)"
            ), {"ts": _stamp(1)})
        conn.execute(text(
            "INSERT INTO spp_raw_15min (settlement_point, timestamp_utc, price, "
            "ingested_at) VALUES ('HB_HOUSTON', :ts, 30.0, :ts)"
        ), {"ts": _stamp(1)})
        conn.execute(text(
            "INSERT INTO features (window_id, target_time) VALUES (1, :ts)"
        ), {"ts": _stamp(1)})
    quality.record(engine, stage="recent hourly", table="market_data_hourly",
                   rows=10, passed=True)


class TestHealthy:
    def test_no_alerts_when_everything_is_current(self, db):
        _make_healthy(db)

        assert monitor.collect(db) == []

    def test_exit_zero_when_healthy(self, db, monkeypatch, capsys):
        _make_healthy(db)
        monkeypatch.setattr(monitor, "setup_database_connection",
                            lambda **kw: db)
        monkeypatch.setattr("sys.argv", ["monitor"])

        assert monitor.main() == 0
        assert "healthy" in capsys.readouterr().out


class TestDataStaleness:
    def test_flags_a_table_that_has_stopped_updating(self, db):
        _make_healthy(db)
        with db.begin() as conn:
            conn.execute(text("DELETE FROM market_data_hourly"))
            conn.execute(text(
                "INSERT INTO market_data_hourly (settlement_point, "
                "timestamp_utc, price, ingested_at) "
                "VALUES ('HB_HOUSTON', :ts, 30.0, :ts)"
            ), {"ts": _stamp(500)})

        signals = {a["signal"] for a in monitor.collect(db)}

        assert "data_stale" in signals

    def test_names_the_table_and_the_budget(self, db):
        _make_healthy(db)
        with db.begin() as conn:
            conn.execute(text("DELETE FROM features"))

        alert = next(a for a in monitor.collect(db)
                     if a.get("table") == "features")

        assert "features" in alert["detail"]


class TestJobSilence:
    """Distinct from staleness: the job not running is our problem, a source
    outage is somebody else's, and they need different responses."""

    def test_flags_silence_even_when_data_is_fresh(self, db, monkeypatch):
        _make_healthy(db)
        monkeypatch.setattr(monitor, "MAX_SILENCE_HOURS", 0.0)

        signals = {a["signal"] for a in monitor.collect(db)}

        assert "job_silent" in signals
        assert "data_stale" not in signals, "data is fresh; only the job is quiet"

    def test_flags_when_no_check_has_ever_run(self, db):
        with db.begin() as conn:
            conn.execute(text(f"DELETE FROM {quality.RUNS_TABLE}"))
        _insert = None

        alerts = monitor.collect(db)

        assert any(a["signal"] == "job_silent" for a in alerts)

    def test_recent_success_clears_silence(self, db):
        _make_healthy(db)

        assert not any(a["signal"] == "job_silent" for a in monitor.collect(db))


class TestCheckFailures:
    def test_surfaces_a_recent_validation_failure(self, db):
        _make_healthy(db)
        quality.record(db, stage="recent raw", table="spp_raw_15min", rows=5,
                       passed=False, failure="price outside ERCOT bounds")

        alert = next(a for a in monitor.collect(db)
                     if a["signal"] == "check_failed")

        assert "bounds" in alert["detail"]

    def test_old_failures_are_not_alerted_forever(self, db):
        _make_healthy(db)
        with db.begin() as conn:
            conn.execute(text(
                f"INSERT INTO {quality.RUNS_TABLE} "
                "(ran_at, stage, table_name, rows_checked, passed, failure) "
                "VALUES (:ts, 'old', 't', 1, 0, 'ancient')"
            ), {"ts": _stamp(90 * 24)})

        assert not any(a["signal"] == "check_failed"
                       for a in monitor.collect(db))


class TestExitCodeAndDelivery:
    def test_exit_one_when_anything_is_wrong(self, db, monkeypatch, capsys):
        monkeypatch.setattr(monitor, "setup_database_connection",
                            lambda **kw: db)
        monkeypatch.setattr("sys.argv", ["monitor", "--quiet"])

        assert monitor.main() == 1
        assert "PIPELINE ALERT" in capsys.readouterr().err

    def test_quiet_prints_nothing_when_healthy(self, db, monkeypatch, capsys):
        _make_healthy(db)
        monkeypatch.setattr(monitor, "setup_database_connection",
                            lambda **kw: db)
        monkeypatch.setattr("sys.argv", ["monitor", "--quiet"])

        assert monitor.main() == 0
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_webhook_failure_does_not_mask_the_alert(self, monkeypatch, capsys):
        """The alert is already on stderr and in the exit code; delivery is
        best effort."""
        monkeypatch.setattr(monitor, "WEBHOOK_URL", "http://127.0.0.1:1/nope")

        monitor.notify([{"signal": "data_stale", "table": "t", "detail": "x"}])

        assert "webhook delivery failed" in capsys.readouterr().err
