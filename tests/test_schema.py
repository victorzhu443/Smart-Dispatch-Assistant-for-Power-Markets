"""Tests for the explicit schema and its migration.

A migration is the most dangerous code in a data project: it runs once, against
production, and a mistake is measured in lost rows. These build a throwaway
database in the old (implicit) shape, migrate it, and assert the properties the
migration exists to establish.

Run with:  pytest tests/ -v
"""
import importlib
import sqlite3

import pytest
from sqlalchemy import create_engine, text


schema = importlib.import_module('data_ingestion.schema')


@pytest.fixture
def legacy_db(tmp_path):
    """A database in the shape pandas.to_sql used to leave behind."""
    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE market_data_hourly "
                "(settlement_point TEXT, timestamp_utc TEXT, price FLOAT)")
    con.execute("CREATE TABLE dam_prices_hourly "
                "(timestamp_utc TEXT, settlement_point TEXT, price FLOAT)")
    con.execute("CREATE TABLE spp_raw_15min "
                "(timestamp_utc TEXT, settlement_point TEXT, "
                " settlement_point_type TEXT, price FLOAT, "
                " repeated_hour_flag TEXT)")
    for hour in range(24):
        ts = f"2026-01-01 {hour:02d}:00:00+0000"
        con.execute("INSERT INTO market_data_hourly VALUES (?,?,?)",
                    ("HB_HOUSTON", ts, 30.0 + hour))
        con.execute("INSERT INTO dam_prices_hourly VALUES (?,?,?)",
                    (ts, "HB_HOUSTON", 29.0 + hour))
        con.execute("INSERT INTO spp_raw_15min VALUES (?,?,?,?,?)",
                    (ts, "HB_HOUSTON", "HU", 31.0 + hour, "N"))
    con.commit()
    con.close()
    return create_engine(f"sqlite:///{path}")


class TestSpecIsWellFormed:
    def test_every_table_declares_what_the_migration_needs(self):
        for name, spec in schema.TABLES.items():
            assert spec["columns"], f"{name} declares no business columns"
            assert "PRIMARY KEY" in spec["ddl"], f"{name} has no primary key"
            assert "NOT NULL" in spec["ddl"], f"{name} has no NOT NULL columns"
            assert spec["indexes"], f"{name} declares no indexes"

    def test_every_table_carries_lineage(self):
        for name, spec in schema.TABLES.items():
            for column in schema.LINEAGE_COLUMNS:
                assert column in spec["ddl"], f"{name} is missing {column}"

    def test_raw_landing_keys_on_ingestion_time(self):
        """Restatements must land beside the original, not overwrite it."""
        ddl = schema.TABLES["spp_raw_15min"]["ddl"]
        assert "PRIMARY KEY (settlement_point, timestamp_utc, ingested_at)" in ddl

    def test_derived_tables_key_only_on_the_business_key(self):
        for name in ("market_data_hourly", "dam_prices_hourly"):
            ddl = schema.TABLES[name]["ddl"]
            assert "PRIMARY KEY (settlement_point, timestamp_utc)" in ddl


class TestMigration:
    def test_preserves_every_row(self, legacy_db):
        before = {
            table: schema.describe(legacy_db, table)["rows"]
            for table in schema.TABLES
        }

        schema.migrate(legacy_db)

        for table, count in before.items():
            assert schema.describe(legacy_db, table)["rows"] == count

    def test_establishes_primary_keys(self, legacy_db):
        assert schema.describe(legacy_db, "market_data_hourly")["primary_key"] == []

        schema.migrate(legacy_db)

        pk = schema.describe(legacy_db, "market_data_hourly")["primary_key"]
        assert pk == ["settlement_point", "timestamp_utc"]

    def test_establishes_lineage_with_a_sentinel(self, legacy_db):
        schema.migrate(legacy_db)

        with legacy_db.connect() as conn:
            report = conn.execute(text(
                "SELECT DISTINCT source_report FROM market_data_hourly"
            )).scalar()
        assert report == schema.PRE_MIGRATION

    def test_establishes_indexes(self, legacy_db):
        assert schema.describe(legacy_db, "market_data_hourly")["indexes"] == []

        schema.migrate(legacy_db)

        assert schema.describe(legacy_db, "market_data_hourly")["indexes"]

    def test_is_idempotent(self, legacy_db):
        schema.migrate(legacy_db)
        first = schema.describe(legacy_db, "market_data_hourly")

        schema.migrate(legacy_db)
        second = schema.describe(legacy_db, "market_data_hourly")

        assert first == second

    def test_leaves_no_temporary_table_behind(self, legacy_db):
        schema.migrate(legacy_db)

        with legacy_db.connect() as conn:
            leftovers = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE name LIKE '%_old'"
            )).fetchall()
        assert leftovers == []

    def test_collapses_pre_existing_duplicates(self, tmp_path):
        """The old schema allowed duplicates; the new one cannot hold them."""
        path = tmp_path / "dupes.db"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE market_data_hourly "
                    "(settlement_point TEXT, timestamp_utc TEXT, price FLOAT)")
        for _ in range(3):
            con.execute("INSERT INTO market_data_hourly VALUES (?,?,?)",
                        ("HB_HOUSTON", "2026-01-01 00:00:00+0000", 30.0))
        con.commit(); con.close()
        engine = create_engine(f"sqlite:///{path}")

        result = schema.migrate_table(
            engine, "market_data_hourly", schema.TABLES["market_data_hourly"]
        )

        assert schema.describe(engine, "market_data_hourly")["rows"] == 1
        assert "duplicate" in result

    def test_creates_a_missing_table(self, tmp_path):
        engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")

        schema.migrate(engine)

        for table in schema.TABLES:
            state = schema.describe(engine, table)
            assert state["exists"] and state["primary_key"]


class TestConstraintsActuallyBite:
    """The point of the migration: guarantees in the database, not in a
    function that callers may bypass."""

    def test_duplicate_key_is_rejected(self, legacy_db):
        schema.migrate(legacy_db)
        row = ("HB_HOUSTON", "2026-01-01 00:00:00+0000", 30.0,
               "2026-01-01 00:00:00+0000", "test", None)

        with pytest.raises(Exception) as excinfo:
            with legacy_db.begin() as conn:
                conn.execute(text(
                    "INSERT INTO market_data_hourly (settlement_point, "
                    "timestamp_utc, price, ingested_at, source_report, "
                    "source_file) VALUES (:a,:b,:c,:d,:e,:f)"
                ), dict(zip("abcdef", row)))
        assert "UNIQUE" in str(excinfo.value).upper()

    def test_null_price_is_rejected(self, legacy_db):
        schema.migrate(legacy_db)

        with pytest.raises(Exception) as excinfo:
            with legacy_db.begin() as conn:
                conn.execute(text(
                    "INSERT INTO market_data_hourly (settlement_point, "
                    "timestamp_utc, price, ingested_at) "
                    "VALUES ('X','2099-01-01 00:00:00+0000',NULL,'now')"
                ))
        assert "NOT NULL" in str(excinfo.value).upper()

    def test_a_restatement_can_land_beside_the_original(self, legacy_db):
        """Raw landing keys on ingested_at, so corrections do not overwrite."""
        schema.migrate(legacy_db)

        with legacy_db.begin() as conn:
            conn.execute(text(
                "INSERT INTO spp_raw_15min (timestamp_utc, settlement_point, "
                "price, ingested_at) VALUES "
                "('2026-01-01 00:00:00+0000','HB_HOUSTON', 99.0, 'later')"
            ))
            n = conn.execute(text(
                "SELECT COUNT(*) FROM spp_raw_15min WHERE "
                "settlement_point='HB_HOUSTON' AND "
                "timestamp_utc='2026-01-01 00:00:00+0000'"
            )).scalar()
        assert n == 2


class TestCheckReporting:
    def test_check_fails_on_a_legacy_database(self, legacy_db, capsys):
        assert schema.check(legacy_db) is False
        assert "NONE" in capsys.readouterr().out

    def test_check_passes_after_migrating(self, legacy_db):
        schema.migrate(legacy_db)
        assert schema.check(legacy_db) is True
