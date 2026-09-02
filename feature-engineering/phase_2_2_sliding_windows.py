# phase_2_2_sliding_windows.py - Generate Sliding Windows (24h to predict next hour)
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()


class InsufficientDataError(RuntimeError):
    """Not enough real market data to continue.

    This used to be handled by generating prices to fill the gap, which meant
    every downstream figure described a random walk instead of the market.
    Failing here is the point: ingest more history, never fabricate it.
    """


def setup_database_connection():
    """Setup database connection"""
    try:
        # Try PostgreSQL first
        pg_user = os.getenv('POSTGRES_USER', 'postgres')
        pg_password = os.getenv('POSTGRES_PASSWORD', 'password')
        pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        pg_port = os.getenv('POSTGRES_PORT', '5432')
        pg_database = os.getenv('POSTGRES_DATABASE', 'smart_dispatch')
        
        pg_connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
        engine = create_engine(pg_connection_string)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        print("✅ PostgreSQL connection successful")
        return engine, "postgresql"
        
    except Exception as e:
        print(f"⚠️ PostgreSQL not available, using SQLite")
        sqlite_path = "market_data.db"
        sqlite_connection_string = f"sqlite:///{sqlite_path}"
        engine = create_engine(sqlite_connection_string)
        print(f"✅ SQLite connection successful: {sqlite_path}")
        return engine, "sqlite"

def load_data_from_sql(engine, table_name="market_data"):
    """Load data from SQL table"""
    print(f"📊 Loading data from SQL table '{table_name}'...")
    
    query = f"""
    SELECT timestamp, settlement_point, price
    FROM {table_name}
    ORDER BY timestamp, settlement_point
    """
    
    df = pd.read_sql(query, engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Loaded {len(df)} records from database")
    return df

def prepare_hourly_data(df):
    """Convert 5-minute data to hourly averages for sliding windows"""
    print(f"🔄 Converting 5-minute data to hourly averages...")
    
    # Focus on a major trading hub for simplicity
    # Find the most common settlement point (likely a major hub)
    top_settlement = df['settlement_point'].value_counts().index[0]
    print(f"📍 Using settlement point: {top_settlement}")
    
    # Filter to single settlement point and create hourly data
    df_hub = df[df['settlement_point'] == top_settlement].copy()
    
    # Set timestamp as index and resample to hourly
    df_hub.set_index('timestamp', inplace=True)
    df_hourly = df_hub.resample('1H')['price'].mean().reset_index()
    
    # Fill any missing hours
    df_hourly['price'] = df_hourly['price'].fillna(method='ffill').fillna(method='bfill')
    
    print(f"✅ Created {len(df_hourly)} hourly price points")
    print(f"⏱️ Time range: {df_hourly['timestamp'].min()} to {df_hourly['timestamp'].max()}")
    print(f"💰 Price range: ${df_hourly['price'].min():.2f} - ${df_hourly['price'].max():.2f}")
    
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
