"""Ingest ERCOT Real-Time Market Settlement Point Prices from the public MIS.

No credentials required. ERCOT publishes annual archives of settlement point
prices for load zones and hubs as report type 13061, downloadable anonymously.
This replaces the previous approach of calling the authenticated api.ercot.com
endpoint, which returned a snapshot across ~1,000 nodes at a single instant --
breadth across locations where the model needs depth across time.

Three tables are written, following a raw/clean split:

    spp_raw_15min       As-delivered real-time 15-minute records. Append-only
                        landing zone; never edited, so any downstream number
                        can be traced back to what ERCOT actually published.
    market_data_hourly  Hourly means derived from the raw table. What the
                        feature pipeline consumes.
    dam_prices_hourly   Day-ahead clearing prices. The day-ahead price is the
                        market's own forecast of real-time, so it is the
                        benchmark any model has to beat to be worth running.

Timestamps are stored in UTC. ERCOT publishes in Central Prevailing Time using
hour-ending convention (Delivery Hour 1-24, where hour 1 covers 00:00-01:00),
and marks the repeated hour at DST fall-back with a flag. Both are handled
explicitly in to_utc_timestamps(); getting this wrong silently shifts every
price by an hour twice a year.

Usage:
    python data-ingestion/ingest_ercot_history.py                # last 2 years
    python data-ingestion/ingest_ercot_history.py --years 2023 2024 2025
    python data-ingestion/ingest_ercot_history.py --hubs HB_HOUSTON HB_NORTH

Re-running is idempotent: rows for an ingested (year, settlement point) range
are replaced, not duplicated.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import create_engine, text

import quality

# ERCOT public MIS endpoints. Anonymous access; no API key.
MIS_LIST_URL = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
MIS_DOWNLOAD_URL = "https://www.ercot.com/misdownload/servlets/mirDownload"

# Annual archives, one per year, published back to 2010.
#   13061  Real-time market settlement point prices (15-minute)
#   13060  Day-ahead market settlement point prices (hourly)
#
# The day-ahead price matters as more than another feature: it is the market's
# own published forecast of the real-time price, so it is the benchmark a
# forecaster has to beat to be worth anything. Naive baselines are a floor;
# day-ahead is the bar.
SPP_REPORT_TYPE_ID = 13061
DAM_REPORT_TYPE_ID = 13060

# ERCOT publishes in Central Prevailing Time.
MARKET_TZ = "America/Chicago"

# The four trading hubs worth modelling. HB_BUSAVG/HB_HUBAVG are averages of
# other points, not independent locations, so they are excluded by default.
DEFAULT_HUBS = ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST"]

# ERCOT systemwide offer cap and floor. A value outside this range is a parsing
# bug, not a market event.
PRICE_FLOOR = -251.0
PRICE_CAP = 5000.0

CACHE_DIR = Path("data-ingestion/.ercot_cache")


class IngestionError(RuntimeError):
    """Ingestion could not produce trustworthy data."""


def setup_database_connection():
    """Connect to PostgreSQL if configured, else the local SQLite file."""
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
        print("PostgreSQL connection successful")
        return engine
    except Exception:
        print("PostgreSQL not available, using SQLite (market_data.db)")
        return create_engine("sqlite:///market_data.db")


def list_annual_archives(report_type_id: int = SPP_REPORT_TYPE_ID) -> dict[int, int]:
    """Map year -> ERCOT document id for every published annual archive."""
    response = requests.get(
        MIS_LIST_URL, params={"reportTypeId": report_type_id}, timeout=60
    )
    response.raise_for_status()

    documents = response.json()["ListDocsByRptTypeRes"]["DocumentList"]
    archives = {}
    for entry in documents:
        doc = entry["Document"]
        name = doc.get("FriendlyName", "")
        # Names look like "RTMLZHBSPP_2025".
        if "_" in name:
            suffix = name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                archives[int(suffix)] = int(doc["DocID"])

    if not archives:
        raise IngestionError(
            f"No annual archives found under report type {SPP_REPORT_TYPE_ID}. "
            f"ERCOT may have changed the report layout."
        )
    return archives


def download_archive(year: int, doc_id: int, prefix: str = "RTMLZHBSPP",
                     cache_dir: Path = CACHE_DIR) -> Path:
    """Download one annual archive, caching it so re-runs do not refetch."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{prefix}_{year}.zip"

    if destination.exists() and destination.stat().st_size > 0:
        print(f"  {year}: using cached {destination.name}")
        return destination

    print(f"  {year}: downloading...")
    response = requests.get(
        MIS_DOWNLOAD_URL, params={"doclookupId": doc_id}, timeout=300
    )
    response.raise_for_status()

    if not response.content.startswith(b"PK"):
        raise IngestionError(
            f"{year}: expected a zip archive, got {len(response.content)} bytes "
            f"starting {response.content[:16]!r}"
        )

    destination.write_bytes(response.content)
    print(f"  {year}: {len(response.content):,} bytes cached")
    return destination


def parse_archive(archive_path: Path, hubs: list[str]) -> pd.DataFrame:
    """Read one annual archive into 15-minute records for the given hubs.

    Each archive holds a single workbook with one sheet per month.
    """
    with zipfile.ZipFile(archive_path) as zf:
        workbook_name = zf.namelist()[0]
        workbook = pd.ExcelFile(io.BytesIO(zf.read(workbook_name)))

        monthly_frames = []
        for sheet in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet)
            frame = frame[frame["Settlement Point Name"].isin(hubs)]
            if not frame.empty:
                monthly_frames.append(frame)

    if not monthly_frames:
        raise IngestionError(
            f"{archive_path.name}: no rows for hubs {hubs}. Check the names "
            f"against the archive's 'Settlement Point Name' column."
        )

    df = pd.concat(monthly_frames, ignore_index=True)
    return df.rename(
        columns={
            "Delivery Date": "delivery_date",
            "Delivery Hour": "delivery_hour",
            "Delivery Interval": "delivery_interval",
            "Repeated Hour Flag": "repeated_hour_flag",
            "Settlement Point Name": "settlement_point",
            "Settlement Point Type": "settlement_point_type",
            "Settlement Point Price": "price",
        }
    )


def to_utc_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Turn ERCOT's date/hour-ending/interval columns into UTC timestamps.

    ERCOT uses hour-ending numbering: Delivery Hour 1 is the interval starting
    at 00:00. Delivery Interval 1-4 selects the 15-minute block within it, so
    interval 1 starts on the hour and interval 4 starts at :45.

    At DST fall-back the local clock repeats an hour, and ERCOT marks the second
    pass with Repeated Hour Flag 'Y'. pandas needs that disambiguation to
    localize: the first pass is still on daylight time, the repeat is on
    standard time.
    """
    local_naive = (
        pd.to_datetime(df["delivery_date"], format="%m/%d/%Y")
        + pd.to_timedelta(df["delivery_hour"].astype(int) - 1, unit="h")
        + pd.to_timedelta((df["delivery_interval"].astype(int) - 1) * 15, unit="m")
    )

    is_first_pass = df["repeated_hour_flag"].astype(str).str.upper().str.strip() != "Y"

    local_aware = local_naive.dt.tz_localize(
        MARKET_TZ,
        ambiguous=is_first_pass.to_numpy(),
        nonexistent="raise",  # spring-forward hours should simply be absent
    )

    out = df.copy()
    out["timestamp_utc"] = local_aware.dt.tz_convert("UTC")
    return out


def parse_dam_archive(archive_path: Path, hubs: list[str]) -> pd.DataFrame:
    """Read one annual day-ahead archive into hourly records for the hubs.

    The day-ahead layout differs from real-time: it is already hourly, so there
    is no interval column, "Hour Ending" is a string like "01:00" rather than
    an integer, and the settlement point column is named differently.
    """
    with zipfile.ZipFile(archive_path) as zf:
        workbook_name = zf.namelist()[0]
        workbook = pd.ExcelFile(io.BytesIO(zf.read(workbook_name)))

        monthly_frames = []
        for sheet in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet)
            frame = frame[frame["Settlement Point"].isin(hubs)]
            if not frame.empty:
                monthly_frames.append(frame)

    if not monthly_frames:
        raise IngestionError(
            f"{archive_path.name}: no day-ahead rows for hubs {hubs}"
        )

    df = pd.concat(monthly_frames, ignore_index=True)
    return df.rename(
        columns={
            "Delivery Date": "delivery_date",
            "Hour Ending": "hour_ending",
            "Repeated Hour Flag": "repeated_hour_flag",
            "Settlement Point": "settlement_point",
            "Settlement Point Price": "price",
        }
    )


def dam_to_utc_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert day-ahead date + "HH:00" hour-ending strings to UTC timestamps.

    Hour ending "01:00" is the interval starting at 00:00, so hour ending
    "24:00" starts at 23:00 on the same day -- it is not midnight of the next.
    """
    hour_ending = (
        df["hour_ending"].astype(str).str.strip().str.split(":").str[0].astype(int)
    )

    local_naive = pd.to_datetime(
        df["delivery_date"], format="%m/%d/%Y"
    ) + pd.to_timedelta(hour_ending - 1, unit="h")

    is_first_pass = df["repeated_hour_flag"].astype(str).str.upper().str.strip() != "Y"

    local_aware = local_naive.dt.tz_localize(
        MARKET_TZ, ambiguous=is_first_pass.to_numpy(), nonexistent="raise"
    )

    out = df.copy()
    out["timestamp_utc"] = local_aware.dt.tz_convert("UTC")
    return out


def validate(df: pd.DataFrame, stage: str) -> None:
    """Fail loudly on data that cannot be trusted downstream."""
    if df.empty:
        raise IngestionError(f"{stage}: no rows")

    nulls = df["price"].isna().sum()
    if nulls:
        raise IngestionError(f"{stage}: {nulls} null prices")

    out_of_range = df[(df["price"] < PRICE_FLOOR) | (df["price"] > PRICE_CAP)]
    if not out_of_range.empty:
        raise IngestionError(
            f"{stage}: {len(out_of_range)} prices outside ERCOT's "
            f"[{PRICE_FLOOR}, {PRICE_CAP}] $/MWh bounds -- "
            f"min {out_of_range['price'].min()}, max {out_of_range['price'].max()}. "
            f"This is a parsing bug, not a market event."
        )

    duplicates = df.duplicated(subset=["timestamp_utc", "settlement_point"]).sum()
    if duplicates:
        raise IngestionError(
            f"{stage}: {duplicates} duplicate (timestamp, settlement_point) rows"
        )


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Average the four 15-minute intervals into hour-beginning records."""
    hourly = (
        df.set_index("timestamp_utc")
        .groupby("settlement_point")["price"]
        .resample("1h")
        .mean()
        .reset_index()
    )

    incomplete = hourly["price"].isna().sum()
    if incomplete:
        # A gap means ERCOT did not publish that hour; drop rather than fill.
        print(f"  dropping {incomplete} hours with no published price")
        hourly = hourly.dropna(subset=["price"])

    return hourly


# Canonical on-disk timestamp format. Fixed width and lexicographically
# ordered, so string comparison in SQL matches chronological comparison.
# Both the stored values and any range bounds MUST use this: formatting the
# bounds differently (e.g. with .isoformat(), which uses 'T' and '+00:00')
# makes the boundary date compare wrong and silently survive a delete.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S%z"


def _format_timestamp(ts) -> str:
    return pd.Timestamp(ts).strftime(TIMESTAMP_FORMAT)


def write_table(df: pd.DataFrame, engine, table: str, *,
                source_report: str | None = None,
                source_file: str | None = None) -> int:
    """Replace this ingest's (settlement point, time range) rows, then append.

    Delete-then-append rather than plain append, so re-running the same years
    does not duplicate rows.

    Since the schema gained primary keys, a delete that fails to match no
    longer produces silent duplicates -- the append raises instead. That is the
    intended behaviour: a loud failure beats a quiet double-count.

    Lineage columns are stamped here so every row can name the feed and file it
    came from. See data-ingestion/schema.py.
    """
    lo = _format_timestamp(df["timestamp_utc"].min())
    hi = _format_timestamp(df["timestamp_utc"].max())
    points = sorted(df["settlement_point"].unique())

    with engine.begin() as conn:
        exists = True
        try:
            conn.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
        except Exception:
            exists = False

        if exists:
            placeholders = ", ".join(f":p{i}" for i in range(len(points)))
            conn.execute(
                text(
                    f"DELETE FROM {table} WHERE timestamp_utc BETWEEN :lo AND :hi "
                    f"AND settlement_point IN ({placeholders})"
                ),
                {"lo": lo, "hi": hi, **{f"p{i}": p for i, p in enumerate(points)}},
            )

    # SQLite has no native tz-aware type; store UTC strings so the value
    # round-trips identically on both backends.
    out = df.copy()
    out["timestamp_utc"] = out["timestamp_utc"].dt.strftime(TIMESTAMP_FORMAT)
    out["ingested_at"] = pd.Timestamp.now(tz="UTC").strftime(TIMESTAMP_FORMAT)
    out["source_report"] = source_report
    out["source_file"] = source_file
    out.to_sql(table, engine, if_exists="append", index=False, method="multi",
               chunksize=5000)

    with engine.connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        # Self-check: the delete-then-append above is only idempotent if the
        # range delete matched. Verify rather than assume.
        duplicates = conn.execute(
            text(
                f"SELECT COUNT(*) FROM (SELECT timestamp_utc, settlement_point "
                f"FROM {table} GROUP BY timestamp_utc, settlement_point "
                f"HAVING COUNT(*) > 1)"
            )
        ).scalar()

    if duplicates:
        raise IngestionError(
            f"{table}: {duplicates} duplicate (timestamp, settlement_point) "
            f"pairs after write -- the range delete did not match. This "
            f"usually means stored timestamps and the delete bounds were "
            f"formatted differently."
        )

    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Years to ingest (default: the two most recent complete)")
    parser.add_argument("--hubs", nargs="+", default=DEFAULT_HUBS,
                        help=f"Settlement points (default: {' '.join(DEFAULT_HUBS)})")
    args = parser.parse_args()

    print("ERCOT settlement point price ingestion (public MIS, no credentials)")

    archives = list_annual_archives()
    print(f"{len(archives)} annual archives available: "
          f"{min(archives)}-{max(archives)}")

    years = args.years or sorted(archives)[-3:-1]
    missing = [y for y in years if y not in archives]
    if missing:
        raise IngestionError(f"No archive published for {missing}")

    print(f"Ingesting {years} for {args.hubs}")

    engine = setup_database_connection()

    frames = []
    for year in years:
        path = download_archive(year, archives[year])
        raw = parse_archive(path, args.hubs)
        raw = to_utc_timestamps(raw)
        quality.timed_validate(engine, validate, raw, f"{year} raw",
                               table="spp_raw_15min")
        print(f"  {year}: {len(raw):,} 15-minute records")
        frames.append(raw)

    raw_all = pd.concat(frames, ignore_index=True)

    raw_columns = [
        "timestamp_utc", "settlement_point", "settlement_point_type",
        "price", "repeated_hour_flag",
    ]
    raw_rows = write_table(raw_all[raw_columns], engine=engine,
                           table="spp_raw_15min",
                           source_report=f"ercot-mis-{SPP_REPORT_TYPE_ID}")
    print(f"spp_raw_15min: {raw_rows:,} rows total")

    hourly = to_hourly(raw_all)
    quality.timed_validate(engine, validate, hourly, "hourly",
                           table="market_data_hourly")
    hourly_rows = write_table(hourly, engine=engine, table="market_data_hourly",
                              source_report=f"derived:spp_raw_15min")
    print(f"market_data_hourly: {hourly_rows:,} rows total")

    # Day-ahead prices: the benchmark the forecaster has to beat.
    print("\nDay-ahead market prices")
    dam_archives = list_annual_archives(DAM_REPORT_TYPE_ID)
    dam_frames = []
    for year in years:
        if year not in dam_archives:
            print(f"  {year}: no day-ahead archive published, skipping")
            continue
        path = download_archive(year, dam_archives[year], prefix="DAMLZHBSPP")
        dam = dam_to_utc_timestamps(parse_dam_archive(path, args.hubs))
        quality.timed_validate(engine, validate, dam, f"{year} day-ahead",
                               table="dam_prices_hourly")
        print(f"  {year}: {len(dam):,} hourly records")
        dam_frames.append(dam)

    if dam_frames:
        dam_all = pd.concat(dam_frames, ignore_index=True)[
            ["timestamp_utc", "settlement_point", "price"]
        ]
        dam_rows = write_table(dam_all, engine=engine, table="dam_prices_hourly",
                               source_report=f"ercot-mis-{DAM_REPORT_TYPE_ID}")
        print(f"dam_prices_hourly: {dam_rows:,} rows total")

    print("\nPer-hub hourly coverage:")
    for point, group in hourly.groupby("settlement_point"):
        print(f"  {point:<12} {len(group):>6,} hours  "
              f"{group['timestamp_utc'].min()} .. {group['timestamp_utc'].max()}  "
              f"${group['price'].min():.2f}..${group['price'].max():.2f}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except IngestionError as exc:
        print(f"\nIngestion failed: {exc}", file=sys.stderr)
        sys.exit(1)
