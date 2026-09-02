"""Tests for incremental ingestion.

The file-selection logic is the part worth pinning. Choosing too few files
silently loses hours -- the failure mode that leaves a forecast uncomputable
without any error -- while choosing too many only costs bandwidth, because the
writer is idempotent. These tests assert the bias runs in that direction.

No network: every test builds its own file listing.

Run with:  pytest tests/ -v
"""
import importlib
from datetime import datetime

import pandas as pd
import pytest


base = importlib.import_module('data_ingestion.ingest_ercot_history')
recent = importlib.import_module('data_ingestion.ingest_recent')


def _files(*stamps):
    """Listing entries shaped like list_recent_files() returns."""
    return sorted(
        [{"name": f"SPPHLZNP6905_{s:%Y%m%d_%H%M}_csv",
          "doc_id": i, "stamp": s} for i, s in enumerate(stamps)],
        key=lambda f: f["stamp"], reverse=True,
    )


class TestFilenameParsing:
    def test_matches_a_real_csv_name(self):
        match = recent.FILENAME_PATTERN.search("SPPHLZNP6905_20260902_0015_csv")

        assert match is not None
        assert match.groups() == ("20260902", "0015")

    def test_ignores_the_xml_twin(self):
        """Each interval is published as both csv and xml; only csv is parsed."""
        assert recent.FILENAME_PATTERN.search("SPPHLZNP6905_20260902_0015_xml") is None

    def test_ignores_unrelated_names(self):
        assert recent.FILENAME_PATTERN.search("RTMLZHBSPP_2025") is None


class TestFileSelection:
    def test_first_run_takes_the_newest_files_up_to_the_cap(self):
        files = _files(*[datetime(2026, 9, 1, h) for h in range(10)])

        chosen = recent.select_new_files(files, since=None, max_files=3)

        assert len(chosen) == 3
        assert chosen[0]["stamp"] == datetime(2026, 9, 1, 9)  # newest first

    def test_skips_files_older_than_what_is_stored(self):
        files = _files(
            datetime(2026, 9, 1, 0), datetime(2026, 9, 1, 6),
            datetime(2026, 9, 1, 12), datetime(2026, 9, 1, 18),
        )
        # 09-01 12:00 local Central is 17:00Z (CDT, UTC-5).
        since = pd.Timestamp("2026-09-01 17:00", tz="UTC")

        chosen = recent.select_new_files(files, since=since, max_files=50)
        stamps = {f["stamp"] for f in chosen}

        assert datetime(2026, 9, 1, 0) not in stamps
        assert datetime(2026, 9, 1, 18) in stamps

    def test_re_ingests_the_boundary_rather_than_skipping_it(self):
        """An hour of slack: re-ingesting is free, skipping loses data."""
        boundary = datetime(2026, 9, 1, 12)
        files = _files(boundary, datetime(2026, 9, 1, 13))
        since = pd.Timestamp("2026-09-01 17:00", tz="UTC")  # == 12:00 local

        chosen = recent.select_new_files(files, since=since, max_files=50)

        assert boundary in {f["stamp"] for f in chosen}

    def test_respects_the_cap_even_when_more_are_new(self):
        files = _files(*[datetime(2026, 9, 1, 0, m) for m in range(0, 60, 5)])

        chosen = recent.select_new_files(files, since=None, max_files=4)

        assert len(chosen) == 4

    def test_returns_nothing_when_already_current(self):
        files = _files(datetime(2026, 9, 1, 0))
        since = pd.Timestamp("2026-09-05 00:00", tz="UTC")

        assert recent.select_new_files(files, since=since, max_files=50) == []


class TestColumnNormalisation:
    """The interval files use compact names; the shared timestamp conversion
    expects the annual archive's names. A rename drift here would surface as a
    KeyError at ingest, but only on a live run."""

    def test_renamed_frame_is_accepted_by_the_shared_converter(self):
        raw = pd.DataFrame({
            "DeliveryDate": ["09/02/2026"],
            "DeliveryHour": [1],
            "DeliveryInterval": [1],
            "SettlementPointName": ["HB_HOUSTON"],
            "SettlementPointType": ["HU"],
            "SettlementPointPrice": [29.16],
            "DSTFlag": ["N"],
        })
        renamed = raw.rename(columns={
            "DeliveryDate": "delivery_date",
            "DeliveryHour": "delivery_hour",
            "DeliveryInterval": "delivery_interval",
            "SettlementPointName": "settlement_point",
            "SettlementPointType": "settlement_point_type",
            "SettlementPointPrice": "price",
            "DSTFlag": "repeated_hour_flag",
        })

        out = base.to_utc_timestamps(renamed)

        assert "timestamp_utc" in out.columns
        # Hour-ending 1 on 09/02 local is 05:00Z during CDT.
        assert out["timestamp_utc"].iloc[0] == pd.Timestamp("2026-09-02 05:00", tz="UTC")

    def test_day_ahead_rename_is_accepted_by_its_converter(self):
        raw = pd.DataFrame({
            "DeliveryDate": ["09/02/2026"],
            "HourEnding": ["01:00"],
            "SettlementPoint": ["HB_HOUSTON"],
            "SettlementPointPrice": [27.20],
            "DSTFlag": ["N"],
        })
        renamed = raw.rename(columns={
            "DeliveryDate": "delivery_date",
            "HourEnding": "hour_ending",
            "SettlementPoint": "settlement_point",
            "SettlementPointPrice": "price",
            "DSTFlag": "repeated_hour_flag",
        })

        out = base.dam_to_utc_timestamps(renamed)

        assert out["timestamp_utc"].iloc[0] == pd.Timestamp("2026-09-02 05:00", tz="UTC")


class TestReportIdentifiers:
    def test_both_feeds_are_configured(self):
        """Real-time alone leaves every forecast uncomputable, because the
        day-ahead price is a required feature."""
        assert recent.RECENT_REPORT_TYPE_ID == 12301
        assert recent.RECENT_DAM_REPORT_TYPE_ID == 12331
