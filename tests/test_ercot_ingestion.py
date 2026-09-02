"""Tests for ERCOT settlement point price ingestion.

The timestamp conversion is the part worth guarding. ERCOT publishes in Central
Prevailing Time using hour-ending numbering, and repeats an hour at DST
fall-back. Getting either wrong shifts prices by an hour without raising
anything, which is the kind of bug that only shows up as unexplained model
error months later.

No network access: every test builds its own small frame.

Run with:  pytest tests/ -v
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(relative_path, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingest = _load("data-ingestion/ingest_ercot_history.py", "ingest_ercot_history")


def _rows(records):
    """Build a frame shaped like a parsed ERCOT archive."""
    return pd.DataFrame(
        records,
        columns=[
            "delivery_date",
            "delivery_hour",
            "delivery_interval",
            "repeated_hour_flag",
            "settlement_point",
            "price",
        ],
    )


class TestHourEndingConversion:
    """Delivery Hour is hour-ENDING: hour 1 is the interval starting 00:00."""

    def test_hour_one_interval_one_is_midnight_local(self):
        df = _rows([("06/15/2025", 1, 1, "N", "HB_HOUSTON", 25.0)])
        out = ingest.to_utc_timestamps(df)

        local = out["timestamp_utc"].dt.tz_convert(ingest.MARKET_TZ)
        assert local.iloc[0].hour == 0
        assert local.iloc[0].strftime("%Y-%m-%d") == "2025-06-15"

    def test_hour_twentyfour_is_11pm_not_midnight(self):
        """Hour 24 ends at midnight, so it starts at 23:00 the same day."""
        df = _rows([("06/15/2025", 24, 1, "N", "HB_HOUSTON", 25.0)])
        out = ingest.to_utc_timestamps(df)

        local = out["timestamp_utc"].dt.tz_convert(ingest.MARKET_TZ)
        assert local.iloc[0].hour == 23
        assert local.iloc[0].strftime("%Y-%m-%d") == "2025-06-15"

    def test_intervals_map_to_quarter_hours(self):
        df = _rows(
            [("06/15/2025", 5, i, "N", "HB_HOUSTON", 25.0) for i in (1, 2, 3, 4)]
        )
        out = ingest.to_utc_timestamps(df)

        local = out["timestamp_utc"].dt.tz_convert(ingest.MARKET_TZ)
        assert list(local.dt.minute) == [0, 15, 30, 45]
        assert set(local.dt.hour) == {4}  # hour-ending 5 starts at 04:00


class TestDaylightSaving:
    def test_repeated_hour_flag_disambiguates_fall_back(self):
        """Nov 2 2025: 01:00-02:00 local happens twice, once CDT once CST."""
        df = _rows(
            [
                ("11/02/2025", 2, 1, "N", "HB_HOUSTON", 20.0),  # first pass, CDT
                ("11/02/2025", 2, 1, "Y", "HB_HOUSTON", 30.0),  # repeat, CST
            ]
        )
        out = ingest.to_utc_timestamps(df)

        first, repeat = out["timestamp_utc"].iloc[0], out["timestamp_utc"].iloc[1]
        assert first != repeat, "repeated hour collapsed to a single timestamp"
        assert (repeat - first) == pd.Timedelta(hours=1)

        # The repeat is on standard time, so its UTC offset is one hour later.
        local = out["timestamp_utc"].dt.tz_convert(ingest.MARKET_TZ)
        assert local.iloc[0].utcoffset() == pd.Timedelta(hours=-5)  # CDT
        assert local.iloc[1].utcoffset() == pd.Timedelta(hours=-6)  # CST

    def test_spring_forward_gap_is_rejected_not_guessed(self):
        """02:00-03:00 does not exist on Mar 9 2025; it must not be invented."""
        df = _rows([("03/09/2025", 3, 1, "N", "HB_HOUSTON", 20.0)])

        with pytest.raises(Exception):
            ingest.to_utc_timestamps(df)


class TestValidation:
    def _valid(self):
        df = _rows(
            [("06/15/2025", h, 1, "N", "HB_HOUSTON", 25.0) for h in range(1, 5)]
        )
        return ingest.to_utc_timestamps(df)

    def test_accepts_clean_data(self):
        ingest.validate(self._valid(), "test")  # must not raise

    def test_rejects_price_above_offer_cap(self):
        df = self._valid()
        df.loc[0, "price"] = ingest.PRICE_CAP + 1

        with pytest.raises(ingest.IngestionError, match="bounds"):
            ingest.validate(df, "test")

    def test_rejects_price_below_floor(self):
        df = self._valid()
        df.loc[0, "price"] = ingest.PRICE_FLOOR - 1

        with pytest.raises(ingest.IngestionError, match="bounds"):
            ingest.validate(df, "test")

    def test_accepts_legitimate_negative_prices(self):
        """Negative prices are real in ERCOT when wind overproduces."""
        df = self._valid()
        df.loc[0, "price"] = -50.0
        ingest.validate(df, "test")  # must not raise

    def test_rejects_nulls(self):
        df = self._valid()
        df.loc[0, "price"] = None

        with pytest.raises(ingest.IngestionError, match="null"):
            ingest.validate(df, "test")

    def test_rejects_duplicate_keys(self):
        df = self._valid()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        with pytest.raises(ingest.IngestionError, match="duplicate"):
            ingest.validate(df, "test")

    def test_rejects_empty(self):
        with pytest.raises(ingest.IngestionError):
            ingest.validate(self._valid().iloc[0:0], "test")


class TestHourlyAggregation:
    def test_averages_the_four_intervals(self):
        df = _rows(
            [
                ("06/15/2025", 5, 1, "N", "HB_HOUSTON", 10.0),
                ("06/15/2025", 5, 2, "N", "HB_HOUSTON", 20.0),
                ("06/15/2025", 5, 3, "N", "HB_HOUSTON", 30.0),
                ("06/15/2025", 5, 4, "N", "HB_HOUSTON", 40.0),
            ]
        )
        hourly = ingest.to_hourly(ingest.to_utc_timestamps(df))

        assert len(hourly) == 1
        assert hourly["price"].iloc[0] == pytest.approx(25.0)

    def test_keeps_hubs_separate(self):
        df = _rows(
            [
                ("06/15/2025", 5, 1, "N", "HB_HOUSTON", 10.0),
                ("06/15/2025", 5, 1, "N", "HB_NORTH", 90.0),
            ]
        )
        hourly = ingest.to_hourly(ingest.to_utc_timestamps(df))

        assert len(hourly) == 2
        by_hub = hourly.set_index("settlement_point")["price"]
        assert by_hub["HB_HOUSTON"] == pytest.approx(10.0)
        assert by_hub["HB_NORTH"] == pytest.approx(90.0)


class TestTimestampFormat:
    """The bug this guards: bounds formatted differently from stored values
    compare wrong as strings, so a range delete silently misses rows."""

    def test_format_is_lexicographically_ordered(self):
        earlier = ingest._format_timestamp(pd.Timestamp("2024-01-01 06:00", tz="UTC"))
        later = ingest._format_timestamp(pd.Timestamp("2024-01-02 06:00", tz="UTC"))
        same_day_later = ingest._format_timestamp(
            pd.Timestamp("2024-01-01 07:00", tz="UTC")
        )

        assert earlier < later
        assert earlier < same_day_later

    def test_format_matches_what_is_stored(self):
        """A stored value and a bound built from it must be byte-identical."""
        ts = pd.Timestamp("2024-01-01 06:00", tz="UTC")
        bound = ingest._format_timestamp(ts)
        stored = pd.Series([ts]).dt.strftime(ingest.TIMESTAMP_FORMAT).iloc[0]

        assert bound == stored, "range bounds would not match stored rows"
