# phase_2_2_sliding_windows.py - Generate Sliding Windows (24h to predict next hour)
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from common.db import setup_database_connection
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

# ERCOT trading hub to model. HB_HOUSTON is the most liquid.
DEFAULT_HUB = "HB_HOUSTON"

# Minimum real hourly observations needed to build 24-hour windows.
MIN_HOURLY_POINTS = 75


class InsufficientDataError(RuntimeError):
    """Not enough real market data to continue.

    This used to be handled by generating prices to fill the gap, which meant
    every downstream figure described a random walk instead of the market.
    Failing here is the point: ingest more history, never fabricate it.
    """


def load_data_from_sql(engine, table_name="market_data_hourly", hub=DEFAULT_HUB):
    """Load the hourly price series for one settlement point.

    Reads market_data_hourly, written by
    python -m data_ingestion.ingest_ercot_history. The older market_data table held
    5-minute snapshots across ~1,000 nodes at a single instant, which is a
    cross-section rather than a time series and cannot be modelled.
    """
    print(f"Loading hourly prices for {hub} from '{table_name}'...")

    query = text(
        f"SELECT timestamp_utc AS timestamp, settlement_point, price "
        f"FROM {table_name} WHERE settlement_point = :hub "
        f"ORDER BY timestamp_utc"
    )
    df = pd.read_sql(query, engine, params={"hub": hub})

    if df.empty:
        raise InsufficientDataError(
            f"No rows in {table_name} for {hub!r}. Run "
            f"'python -m data_ingestion.ingest_ercot_history' first, or pass a "
            f"different --hub."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    print(f"Loaded {len(df):,} hourly records for {hub}")
    return df


def prepare_hourly_data(df):
    """Validate the already-hourly series. No resampling: the ingestion
    already averaged 15-minute intervals into hours."""
    hub = df["settlement_point"].iloc[0]
    df_hourly = (
        df[["timestamp", "price"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    gaps = df_hourly["timestamp"].diff().dropna()
    irregular = gaps[gaps != pd.Timedelta(hours=1)]
    if len(irregular):
        print(f"  note: {len(irregular)} breaks in hourly continuity")

    if len(df_hourly) < MIN_HOURLY_POINTS:
        raise InsufficientDataError(
            f"Need {MIN_HOURLY_POINTS} hourly observations to build 24-hour "
            f"windows, have {len(df_hourly)} for {hub!r}. Ingest more history; "
            f"do not pad the series."
        )

    print(f"{len(df_hourly):,} hourly price points for {hub}")
    return df_hourly

def generate_sliding_windows(df_hourly, window_size=24):
    """
    Phase 2.2: Generate Sliding Windows (24h to predict next hour)
    Use rolling() + lag shift
    Test Case: First row contains exactly 24 timestamps
    """
    print(f"🔄 Phase 2.2: Generating sliding windows...")
    print(f"📏 Window size: {window_size} hours")
    
    # Ensure data is sorted by timestamp
    df_hourly = df_hourly.sort_values('timestamp').reset_index(drop=True)
    
    # Create sliding windows using rolling() + lag shift
    sliding_windows = []
    
    # We need at least window_size + 1 data points (24 for input + 1 for target)
    min_required = window_size + 1
    
    if len(df_hourly) < min_required:
        raise InsufficientDataError(
            f"Need {min_required} hourly observations to build a "
            f"{window_size}-hour window plus a target, have {len(df_hourly)}. "
            f"Ingest more history -- see the README section 'Generating the "
            f"data and models'. Do not pad the series."
        )

    print(f"📊 Creating sliding windows from {len(df_hourly)} hourly data points...")
    
    for i in range(len(df_hourly) - window_size):
        # Extract 24-hour window (input features)
        window_start = i
        window_end = i + window_size
        target_idx = i + window_size
        
        # Get the 24-hour price sequence
        price_window = df_hourly.iloc[window_start:window_end]['price'].values
        timestamp_window = df_hourly.iloc[window_start:window_end]['timestamp'].values
        
        # Get the target (next hour price)
        target_price = df_hourly.iloc[target_idx]['price']
        target_timestamp = df_hourly.iloc[target_idx]['timestamp']
        
        # Create window record
        window_record = {
            'window_id': i,
            'window_start_time': timestamp_window[0],
            'window_end_time': timestamp_window[-1],
            'target_time': target_timestamp,
            'price_sequence': price_window.tolist(),  # 24 hourly prices
            'timestamp_sequence': timestamp_window.tolist(),  # 24 timestamps
            'target_price': target_price
        }
        
        sliding_windows.append(window_record)
    
    print(f"✅ Generated {len(sliding_windows)} sliding windows")
    
    # Test Case: First row contains exactly 24 timestamps
    if sliding_windows:
        first_window = sliding_windows[0]
        timestamp_count = len(first_window['timestamp_sequence'])
        test_passed = timestamp_count == window_size
        
        print(f"\n🧪 Test Case - First row contains exactly {window_size} timestamps:")
        print(f"   Actual timestamps: {timestamp_count}")
        print(f"   Required: {window_size}")
        print(f"   Result: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        # Show details of first window
        print(f"\n📋 First sliding window details:")
        print(f"   Window ID: {first_window['window_id']}")
        print(f"   Time range: {first_window['window_start_time']} to {first_window['window_end_time']}")
        print(f"   Target time: {first_window['target_time']}")
        print(f"   Price sequence length: {len(first_window['price_sequence'])}")
        print(f"   Price range in window: ${min(first_window['price_sequence']):.2f} - ${max(first_window['price_sequence']):.2f}")
        print(f"   Target price: ${first_window['target_price']:.2f}")
        
        # Show first few timestamps and prices
        print(f"\n📊 Sample from first window:")
        for j in range(min(5, len(first_window['timestamp_sequence']))):
            ts = first_window['timestamp_sequence'][j]
            price = first_window['price_sequence'][j]
            print(f"   {pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M')} | ${price:.2f}")
        print(f"   ... ({len(first_window['timestamp_sequence'])-5} more timestamps)")
    else:
        print(f"❌ No sliding windows generated")
        return None
    
    return sliding_windows

def main():
    """Execute Phase 2.2 workflow"""
    print("🚀 Phase 2.2: Generate Sliding Windows (24h to predict next hour)")
    
    try:
        # Step 1: Setup database connection
        engine, db_type = setup_database_connection()
        
        # Step 2: Load data from SQL
        df_raw = load_data_from_sql(engine)
        
        # Step 3: Prepare hourly data
        df_hourly = prepare_hourly_data(df_raw)
        
        # Step 4: Generate sliding windows (Phase 2.2)
        windows = generate_sliding_windows(df_hourly, window_size=24)
        
        if windows and len(windows) > 0:
            print(f"\n✅ Phase 2.2 COMPLETE: Successfully generated sliding windows")
            print(f"📊 Generated {len(windows)} training samples")
            print(f"🎯 Each sample: 24 hours → predict next hour")
            print(f"🔄 Next: Phase 2.3 - Compute Technical Features")
            
            return windows, df_hourly, engine
        else:
            print(f"\n❌ Phase 2.2 failed: Could not generate sliding windows")
            return None, None, None
            
    except Exception as e:
        print(f"❌ Phase 2.2 failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    windows, df_hourly, engine = main()
