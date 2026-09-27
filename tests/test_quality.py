"""Tests for data quality recording and freshness detection.

Two behaviours worth pinning. Recording must never take down an ingestion that
is otherwise healthy — an observability failure is not a data failure. And
freshness must treat day-ahead prices correctly: they are published before
delivery, so their newest event time is legitimately in the future, and naive
"age > budget" logic would either flag that as broken or, worse, mask a real
staleness elsewhere.

Run with:  pytest tests/ -v
"""
import importlib
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text


schema = importlib.import_module('data_ingestion.schema')
quality = importlib.import_module('data_ingestion.quality')


def _stamp(hours_ago: float) -> str:
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return ts.strftime("%Y-%m-%d %H:%M:%S%z")


@pytest.fixture
def db(tmp_path):
    """An empty database with the declared schema applied."""
    engine = create_engine(f"sqlite:///{tmp_path / 'q.db'}")
    schema.migrate(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE features (window_id INTEGER, target_time TEXT)"
        ))
    return engine


def _insert_hourly(engine, hours_ago: float):
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO market_data_hourly (settlement_point, timestamp_utc, "
            "price, ingested_at) VALUES ('HB_HOUSTON', :ts, 30.0, :ts)"
        ), {"ts": _stamp(hours_ago)})


class TestRecording:
    def test_records_a_passing_check(self, db):
        quality.record(db, stage="test", table="market_data_hourly",
                       rows=10, passed=True)

        runs = quality.recent_runs(db)
        assert len(runs) == 1
        assert runs.iloc[0]["passed"] == 1
        assert runs.iloc[0]["rows_checked"] == 10

    def test_records_a_failure_with_its_reason(self, db):
        quality.record(db, stage="test", table="t", rows=5, passed=False,
                       failure="price out of bounds")

        runs = quality.recent_runs(db)
        assert runs.iloc[0]["passed"] == 0
        assert "bounds" in runs.iloc[0]["failure"]

    def test_recording_never_raises(self, tmp_path):
        """An observability failure must not break a healthy ingestion."""
        broken = create_engine(f"sqlite:////nonexistent/path/{tmp_path.name}.db")

        quality.record(broken, stage="test", table="t", rows=1, passed=True)

    def test_history_accumulates_newest_first(self, db):
        for i in range(3):
            quality.record(db, stage=f"stage-{i}", table="t", rows=i,
                           passed=True)

        runs = quality.recent_runs(db)
        assert len(runs) == 3


class TestTimedValidate:
    def test_passes_through_and_records_success(self, db):
        called = {}

        def validate(df, stage):
            called["stage"] = stage

        quality.timed_validate(db, validate, [1, 2, 3], "my-stage", table="t")

        assert called["stage"] == "my-stage"
        assert quality.recent_runs(db).iloc[0]["passed"] == 1

    def test_records_then_re_raises_on_failure(self, db):
        def validate(df, stage):
            raise ValueError("bad rows")

        with pytest.raises(ValueError, match="bad rows"):
            quality.timed_validate(db, validate, [1], "my-stage", table="t")

        # The failure must be recorded, not swallowed — the interesting rows in
        # the history are the failures.
        runs = quality.recent_runs(db)
        assert runs.iloc[0]["passed"] == 0
        assert "bad rows" in runs.iloc[0]["failure"]

    def test_records_the_row_count_it_checked(self, db):
        quality.timed_validate(db, lambda df, s: None, [1] * 42, "s", table="t")

        assert quality.recent_runs(db).iloc[0]["rows_checked"] == 42


class TestFreshness:
    def test_fresh_table_is_not_stale(self, db):
        _insert_hourly(db, hours_ago=1)

        row = next(r for r in quality.table_freshness(db)
                   if r["table"] == "market_data_hourly")

        assert row["stale"] is False
        assert row["age_hours"] == pytest.approx(1, abs=0.2)

    def test_old_table_is_stale(self, db):
        _insert_hourly(db, hours_ago=100)

        row = next(r for r in quality.table_freshness(db)
                   if r["table"] == "market_data_hourly")

        assert row["stale"] is True

    def test_empty_table_counts_as_stale(self, db):
        row = next(r for r in quality.table_freshness(db)
                   if r["table"] == "market_data_hourly")

        assert row["stale"] is True
        assert row["rows"] == 0

    def test_day_ahead_published_ahead_of_delivery_is_not_stale(self, db):
        """DAM clears the day before, so a future event time is healthy."""
        with db.begin() as conn:
            conn.execute(text(
                "INSERT INTO dam_prices_hourly (settlement_point, "
                "timestamp_utc, price, ingested_at) "
                "VALUES ('HB_HOUSTON', :ts, 30.0, :now)"
            ), {"ts": _stamp(-12), "now": _stamp(0)})

        row = next(r for r in quality.table_freshness(db)
                   if r["table"] == "dam_prices_hourly")

        assert row["age_hours"] < 0, "future event time should read as negative"
        assert row["stale"] is False

    def test_every_tracked_table_has_a_budget(self):
        for table in quality.TIME_COLUMN:
            assert table in quality.STALENESS_BUDGET_HOURS

    def test_report_returns_false_when_something_is_stale(self, db, capsys):
        _insert_hourly(db, hours_ago=500)

        assert quality.report(db) is False
        assert "STALE" in capsys.readouterr().out
