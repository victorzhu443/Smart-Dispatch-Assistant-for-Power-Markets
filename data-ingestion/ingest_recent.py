"""Incremental ingestion: pull settlement point prices published since last run.

ingest_ercot_history.py backfills from annual archives, which ERCOT publishes
once a year. That is the wrong instrument for keeping current: it leaves the
database as stale as the last archive, which is why the forecast API's
staleness fallback fires on every live request.

ERCOT also publishes each 15-minute settlement interval as its own file
(report 12301), on a rolling window of roughly the last seven days. This reads
those, ingests only what is new, and is safe to run from cron every 15 minutes.

    backfill    ingest_ercot_history.py   annual archives, 2010 onwards
    top up      this module               rolling 7-day window, every 15 min

Both write the same tables through the same idempotent writer, so overlapping
runs converge rather than duplicating.

Usage:
    python data-ingestion/ingest_recent.py
    python data-ingestion/ingest_recent.py --max-files 50 --hubs HB_NORTH

Cron (every 15 minutes, log to a file):
    */15 * * * * cd /path/to/repo && python data-ingestion/ingest_recent.py \\
                 >> logs/ingest.log 2>&1

Exits non-zero on failure so cron reports it, and prints nothing surprising on
a no-op run.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# Sibling module: data-ingestion/ is on sys.path when this runs as a script.
import ingest_ercot_history as base

# Report 12301: settlement point prices for one 15-minute interval, published
# as it settles. Rolling window of about seven days.
RECENT_REPORT_TYPE_ID = 12301

# Report 12331: the day-ahead clearing prices for a whole delivery day,
# published around 12:30 the day before. Rolling window of about a month.
#
# This is not optional. The forecaster uses the day-ahead price for the target
# hour as a feature, so without it no forecast is computable for any hour --
# which is exactly what a live request hit before this was added:
#
#     422  features not computable: ['dam', 'dam_minus_lag1']
#
# Because DAM clears the day before delivery, these files also carry prices for
# hours that have not happened yet, which is what makes a forward-looking
# forecast possible at all.
RECENT_DAM_REPORT_TYPE_ID = 12331

# Filenames look like SPPHLZNP6905_20260902_0015_csv.
FILENAME_PATTERN = re.compile(r"_(\d{8})_(\d{4})_csv$")

# A guard against a first run with an empty table trying to fetch everything.
DEFAULT_MAX_FILES = 200


class RecentIngestError(RuntimeError):
    pass


def list_recent_files() -> list[dict]:
    """Every CSV interval file ERCOT currently offers, newest first."""
    response = requests.get(
        base.MIS_LIST_URL,
        params={"reportTypeId": RECENT_REPORT_TYPE_ID},
        timeout=60,
    )
    response.raise_for_status()

    documents = response.json()["ListDocsByRptTypeRes"]["DocumentList"]
    files = []
    for entry in documents:
        doc = entry["Document"]
        name = doc.get("FriendlyName", "")
        match = FILENAME_PATTERN.search(name)
        if not match:
            continue  # skip the xml twin of each csv
        date_part, time_part = match.groups()
        # The filename stamp is Central local time, matching the delivery data.
        stamp = datetime.strptime(date_part + time_part, "%Y%m%d%H%M")
        files.append({"name": name, "doc_id": int(doc["DocID"]), "stamp": stamp})

    if not files:
        raise RecentIngestError(
            f"No CSV files under report {RECENT_REPORT_TYPE_ID}; ERCOT may have "
            f"changed the report layout."
        )
    return sorted(files, key=lambda f: f["stamp"], reverse=True)


def latest_ingested(engine, hub: str) -> pd.Timestamp | None:
    """Most recent hour already stored, or None if the table is empty."""
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            value = conn.execute(
                text("SELECT MAX(timestamp_utc) FROM market_data_hourly "
                     "WHERE settlement_point = :hub"),
                {"hub": hub},
            ).scalar()
    except Exception:
        return None
    return pd.to_datetime(value, utc=True, format="mixed") if value else None


def select_new_files(files: list[dict], since: pd.Timestamp | None,
                     max_files: int) -> list[dict]:
    """Files covering intervals at or after `since`.

    The filename stamp is local Central; `since` is UTC. Converting the cutoff
    rather than every filename keeps this cheap, and an hour of slack on either
    side means a boundary file is re-ingested rather than skipped -- harmless,
    because the writer is idempotent, whereas skipping loses data.
    """
    if since is None:
        return files[:max_files]

    cutoff_local = (
        since.tz_convert(base.MARKET_TZ).tz_localize(None) - timedelta(hours=1)
    )
    fresh = [f for f in files if f["stamp"] >= cutoff_local]
    return fresh[:max_files]


def download_and_parse(doc_id: int, hubs: list[str]) -> pd.DataFrame:
    """One interval file, filtered to the hubs of interest."""
    response = requests.get(
        base.MIS_DOWNLOAD_URL, params={"doclookupId": doc_id}, timeout=120
    )
    response.raise_for_status()

    if not response.content.startswith(b"PK"):
        raise RecentIngestError(
            f"doc {doc_id}: expected a zip, got {response.content[:16]!r}"
        )

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    frame = pd.read_csv(io.BytesIO(archive.read(archive.namelist()[0])))
    frame = frame[frame["SettlementPointName"].isin(hubs)]

    # The interval files use compact column names and a DSTFlag where the
    # annual archives use spaced names and a Repeated Hour Flag. Normalise to
    # the annual shape so the shared timestamp conversion applies unchanged.
    return frame.rename(columns={
        "DeliveryDate": "delivery_date",
        "DeliveryHour": "delivery_hour",
        "DeliveryInterval": "delivery_interval",
        "SettlementPointName": "settlement_point",
        "SettlementPointType": "settlement_point_type",
        "SettlementPointPrice": "price",
        "DSTFlag": "repeated_hour_flag",
    })


def ingest_day_ahead(engine, hubs: list[str], days: int) -> int:
    """Top up day-ahead prices from the most recent daily files.

    Each file covers one delivery day, so `days` files bridge `days` days.
    Returns the number of hourly rows written.
    """
    response = requests.get(
        base.MIS_LIST_URL,
        params={"reportTypeId": RECENT_DAM_REPORT_TYPE_ID},
        timeout=60,
    )
    response.raise_for_status()

    documents = [
        entry["Document"]
        for entry in response.json()["ListDocsByRptTypeRes"]["DocumentList"]
        if entry["Document"].get("FriendlyName", "").endswith("_csv")
    ]
    if not documents:
        raise RecentIngestError(
            f"No CSV files under report {RECENT_DAM_REPORT_TYPE_ID}"
        )

    # All these files share a name, so publish date is the only ordering.
    documents.sort(key=lambda d: d["PublishDate"], reverse=True)
    wanted = documents[:max(1, days)]

    frames = []
    for doc in wanted:
        try:
            raw = requests.get(
                base.MIS_DOWNLOAD_URL,
                params={"doclookupId": int(doc["DocID"])},
                timeout=120,
            )
            raw.raise_for_status()
            archive = zipfile.ZipFile(io.BytesIO(raw.content))
            frame = pd.read_csv(io.BytesIO(archive.read(archive.namelist()[0])))
            frame = frame[frame["SettlementPoint"].isin(hubs)]
            if not frame.empty:
                frames.append(frame.rename(columns={
                    "DeliveryDate": "delivery_date",
                    "HourEnding": "hour_ending",
                    "SettlementPoint": "settlement_point",
                    "SettlementPointPrice": "price",
                    "DSTFlag": "repeated_hour_flag",
                }))
        except Exception as exc:
            print(f"  skipped a day-ahead file: {exc}")
            continue

    if not frames:
        raise RecentIngestError(f"No day-ahead rows for {hubs}")

    dam = base.dam_to_utc_timestamps(pd.concat(frames, ignore_index=True))
    dam = dam.drop_duplicates(subset=["timestamp_utc", "settlement_point"])
    base.validate(dam, "recent day-ahead")

    base.write_table(
        dam[["timestamp_utc", "settlement_point", "price"]],
        engine=engine, table="dam_prices_hourly",
    )
    return len(dam)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hubs", nargs="+", default=base.DEFAULT_HUBS)
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    args = parser.parse_args()

    engine = base.setup_database_connection()
    since = latest_ingested(engine, args.hubs[0])

    if since is None:
        print("market_data_hourly is empty for these hubs. Run "
              "ingest_ercot_history.py first to backfill, then top up here.")
    else:
        age = (datetime.now(timezone.utc) - since.to_pydatetime())
        print(f"Latest stored hour: {since} ({age.total_seconds()/3600:,.1f}h old)")

    files = list_recent_files()
    new_files = select_new_files(files, since, args.max_files)

    if not new_files:
        print("Already current; nothing to ingest.")
        return 0

    print(f"{len(files)} files offered, {len(new_files)} to ingest "
          f"({new_files[-1]['stamp']:%Y-%m-%d %H:%M} .. "
          f"{new_files[0]['stamp']:%Y-%m-%d %H:%M} local)")

    frames = []
    for i, meta in enumerate(reversed(new_files), 1):
        try:
            frames.append(download_and_parse(meta["doc_id"], args.hubs))
        except Exception as exc:
            # One bad file should not lose the rest of the batch.
            print(f"  skipped {meta['name']}: {exc}")
            continue
        if i % 25 == 0:
            print(f"  {i}/{len(new_files)} fetched")

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RecentIngestError(
            f"No rows for {args.hubs} in {len(new_files)} files"
        )

    raw = base.to_utc_timestamps(pd.concat(frames, ignore_index=True))
    raw = raw.drop_duplicates(subset=["timestamp_utc", "settlement_point"])
    base.validate(raw, "recent raw")

    raw_columns = ["timestamp_utc", "settlement_point", "settlement_point_type",
                   "price", "repeated_hour_flag"]
    base.write_table(raw[raw_columns], engine=engine, table="spp_raw_15min")

    hourly = base.to_hourly(raw)
    base.validate(hourly, "recent hourly")
    base.write_table(hourly, engine=engine, table="market_data_hourly")

    print(f"Ingested {len(raw):,} intervals -> {len(hourly):,} hourly rows")

    # Day-ahead prices are a required feature, so a real-time top-up alone
    # leaves every forecast uncomputable. Bridge at least the gap just filled.
    gap_days = 2
    if since is not None:
        gap_days = max(2, int((datetime.now(timezone.utc)
                               - since.to_pydatetime()).days) + 2)
    dam_rows = ingest_day_ahead(engine, args.hubs, days=min(gap_days, 31))
    print(f"Ingested {dam_rows:,} day-ahead hourly rows")

    now_latest = latest_ingested(engine, args.hubs[0])
    added = "unknown"
    if since is not None and now_latest is not None:
        added = f"{(now_latest - since).total_seconds() / 3600:,.0f}"
    print(f"Latest stored hour is now {now_latest} (+{added}h)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RecentIngestError, base.IngestionError) as exc:
        print(f"\nIncremental ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as exc:
        print(f"\nERCOT unreachable: {exc}", file=sys.stderr)
        sys.exit(1)
