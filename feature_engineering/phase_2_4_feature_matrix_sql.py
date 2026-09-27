# phase_2_4_feature_matrix_sql.py - Write Feature Matrix to SQL Table features
import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from common.db import setup_database_connection
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

# ERCOT trading hub to model. HB_HOUSTON is the most liquid.
DEFAULT_HUB = "HB_HOUSTON"

# Minimum real hourly observations needed to build 24-hour windows plus a
# target. Below this the pipeline stops rather than padding the series.
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
    """Generate sliding windows from hourly data"""
    print(f"🔄 Generating sliding windows (window size: {window_size})...")
    
    df_hourly = df_hourly.sort_values('timestamp').reset_index(drop=True)
    sliding_windows = []
    
    for i in range(len(df_hourly) - window_size):
        window_start = i
        window_end = i + window_size
        target_idx = i + window_size
        
        price_window = df_hourly.iloc[window_start:window_end]['price'].values
        timestamp_window = df_hourly.iloc[window_start:window_end]['timestamp'].values
        
        target_price = df_hourly.iloc[target_idx]['price']
        target_timestamp = df_hourly.iloc[target_idx]['timestamp']
        
        window_record = {
            'window_id': i,
            'window_start_time': timestamp_window[0],
            'window_end_time': timestamp_window[-1],
            'target_time': target_timestamp,
            'price_sequence': price_window.tolist(),
            'timestamp_sequence': timestamp_window.tolist(),
            'target_price': target_price
        }
        
        sliding_windows.append(window_record)
    
    print(f"✅ Generated {len(sliding_windows)} sliding windows")
    return sliding_windows

def compute_technical_features(windows):
    """Compute technical features for each sliding window"""
    print(f"🔄 Computing technical features over {len(windows)} windows...")
    
    feature_records = []
    
    for window in windows:
        window_id = window['window_id']
        price_sequence = np.array(window['price_sequence'])
        target_price = window['target_price']
        target_time = pd.to_datetime(window['target_time'])
        
        # Basic Statistical Features
        price_mean = np.mean(price_sequence)
        price_std = np.std(price_sequence)
        price_min = np.min(price_sequence)
        price_max = np.max(price_sequence)
        price_median = np.median(price_sequence)
        
        # Trend Analysis
        x = np.arange(len(price_sequence))
        trend_slope = np.polyfit(x, price_sequence, 1)[0]
        
        # Price momentum and changes
        price_first = price_sequence[0]
        price_last = price_sequence[-1]
        price_change = price_last - price_first
        price_change_pct = (price_change / price_first) * 100 if price_first != 0 else 0
        
        # Volatility measures
        price_range = price_max - price_min
        price_volatility = price_std / price_mean if price_mean != 0 else 0
        
        # Moving averages
        if len(price_sequence) >= 12:
            price_ma_12 = np.mean(price_sequence[-12:])
            price_ma_6 = np.mean(price_sequence[-6:])
        else:
            price_ma_12 = price_mean
            price_ma_6 = price_mean
        
        # Time-based features
        hour_of_day = target_time.hour
        day_of_week = target_time.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        is_peak_hour = 1 if 14 <= hour_of_day <= 18 else 0
        
        # Momentum indicators
        momentum_1h = price_sequence[-1] - price_sequence[-2] if len(price_sequence) >= 2 else 0
        momentum_3h = price_sequence[-1] - price_sequence[-4] if len(price_sequence) >= 4 else 0
        
        # Relative position
        recent_min = np.min(price_sequence[-6:]) if len(price_sequence) >= 6 else price_min
        recent_max = np.max(price_sequence[-6:]) if len(price_sequence) >= 6 else price_max
        relative_position = ((price_last - recent_min) / (recent_max - recent_min) 
                           if recent_max != recent_min else 0.5)
        
        # Create feature record
        feature_record = {
            'window_id': window_id,
            'target_time': target_time,
            'target_price': target_price,
            'price_mean': round(price_mean, 4),
            'price_std': round(price_std, 4),
            'price_min': round(price_min, 4),
            'price_max': round(price_max, 4),
            'price_median': round(price_median, 4),
            'trend_slope': round(trend_slope, 6),
            'price_change': round(price_change, 4),
            'price_change_pct': round(price_change_pct, 4),
            'price_range': round(price_range, 4),
            'price_volatility': round(price_volatility, 6),
            'price_ma_12': round(price_ma_12, 4),
            'price_ma_6': round(price_ma_6, 4),
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_peak_hour': is_peak_hour,
            'momentum_1h': round(momentum_1h, 4),
            'momentum_3h': round(momentum_3h, 4),
            'relative_position': round(relative_position, 4),
            'price_sequence_json': json.dumps(price_sequence.tolist())  # Store as JSON string for SQL
        }
        
        feature_records.append(feature_record)
    
    # Convert to DataFrame
    df_features = pd.DataFrame(feature_records)
    print(f"✅ Computed technical features for {len(df_features)} windows")
    
    return df_features

def write_feature_matrix_to_sql(df_features, engine, table_name="features"):
    """
    Phase 2.4: Write Feature Matrix to SQL Table features
    Test Case: Verify number of rows = number of sliding windows
    """
    print(f"🔄 Phase 2.4: Writing feature matrix to SQL table '{table_name}'...")
    
    try:
        # Store original sliding window count
        sliding_window_count = len(df_features)
        
        # Write to a staging table first, then swap. `if_exists="replace"`
        # drops the live table before inserting, so a failure mid-insert leaves
        # it empty -- which is exactly what happened the first time this ran
        # against real data instead of 51 rows.
        staging = f"{table_name}_staging"

        # method="multi" batches rows into one INSERT, and SQLite caps a
        # statement at 32,766 bound variables. With 23 columns that is ~1,424
        # rows, so an unbounded batch raises "too many SQL variables". The bug
        # was invisible while the table held 51 rows.
        max_variables = 30000
        chunk = max(1, max_variables // max(1, len(df_features.columns)))

        df_features.to_sql(
            name=staging,
            con=engine,
            if_exists='replace',
            index=False,
            method='multi',
            chunksize=chunk,
        )

        # Swap only once the staging table is known good.
        with engine.begin() as conn:
            staged = conn.execute(
                text(f"SELECT COUNT(*) FROM {staging}")
            ).scalar()
            if staged != sliding_window_count:
                raise SQLAlchemyError(
                    f"staged {staged} rows but expected {sliding_window_count}; "
                    f"leaving {table_name} untouched"
                )
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            conn.execute(text(f"ALTER TABLE {staging} RENAME TO {table_name}"))
        
        print(f"✅ Feature matrix saved to table '{table_name}'")
        print(f"   Features saved: {len(df_features)} records")
        print(f"   Columns saved: {len(df_features.columns)} features")
        
        # Test Case: Verify number of rows = number of sliding windows
        with engine.connect() as conn:
            # Query row count from database
            result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table_name}"))
            db_row_count = result.fetchone()[0]
            
            # Compare with original sliding window count
            counts_match = db_row_count == sliding_window_count
            
            print(f"\n🧪 Test Case - Verify number of rows = number of sliding windows:")
            print(f"   Sliding windows: {sliding_window_count}")
            print(f"   Database rows: {db_row_count}")
            print(f"   Match: {'✅ PASSED' if counts_match else '❌ FAILED'}")
            
            # Additional validation: verify data integrity
            sample_query = f"""
                SELECT window_id, target_time, target_price, price_mean, price_std, trend_slope
                FROM {table_name} 
                ORDER BY window_id 
                LIMIT 5
            """
            sample_result = conn.execute(text(sample_query))
            sample_rows = sample_result.fetchall()
            
            print(f"\n📊 Sample feature records from database:")
            print(f"   {'ID':<4} {'Target Time':<20} {'Target$':<8} {'Mean$':<8} {'Std$':<8} {'Trend':<10}")
            print(f"   {'-'*4} {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
            
            for row in sample_rows:
                target_time_str = pd.to_datetime(row.target_time).strftime('%m-%d %H:%M')
                print(f"   {row.window_id:<4} {target_time_str:<20} ${row.target_price:<7.2f} ${row.price_mean:<7.2f} ${row.price_std:<7.2f} {row.trend_slope:<10.6f}")
            
            # Verify feature completeness
            feature_stats_query = f"""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT window_id) as unique_windows,
                    MIN(target_price) as min_target_price,
                    MAX(target_price) as max_target_price,
                    AVG(price_mean) as avg_price_mean,
                    AVG(price_std) as avg_price_std,
                    AVG(trend_slope) as avg_trend_slope
                FROM {table_name}
            """
            stats_result = conn.execute(text(feature_stats_query))
            stats = stats_result.fetchone()
            
            print(f"\n📈 Feature Matrix Statistics:")
            print(f"   Total records: {stats.total_records}")
            print(f"   Unique windows: {stats.unique_windows}")
            print(f"   Target price range: ${stats.min_target_price:.2f} - ${stats.max_target_price:.2f}")
            print(f"   Average price mean: ${stats.avg_price_mean:.2f}")
            print(f"   Average price std: ${stats.avg_price_std:.2f}")
            print(f"   Average trend slope: {stats.avg_trend_slope:.6f}")
            
            # Check for any null values in critical features
            null_check_query = f"""
                SELECT 
                    SUM(CASE WHEN price_mean IS NULL THEN 1 ELSE 0 END) as null_price_mean,
                    SUM(CASE WHEN price_std IS NULL THEN 1 ELSE 0 END) as null_price_std,
                    SUM(CASE WHEN trend_slope IS NULL THEN 1 ELSE 0 END) as null_trend_slope,
                    SUM(CASE WHEN target_price IS NULL THEN 1 ELSE 0 END) as null_target_price
                FROM {table_name}
            """
            null_result = conn.execute(text(null_check_query))
            null_stats = null_result.fetchone()
            
            total_nulls = (null_stats.null_price_mean + null_stats.null_price_std + 
                          null_stats.null_trend_slope + null_stats.null_target_price)
            
            print(f"\n🔍 Data Quality Check:")
            print(f"   Null values in critical features: {total_nulls}")
            print(f"   Data quality: {'✅ EXCELLENT' if total_nulls == 0 else '⚠️ NEEDS ATTENTION'}")
            
            return counts_match and total_nulls == 0
            
    except SQLAlchemyError as e:
        print(f"❌ SQL operation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Execute Phase 2.4 workflow"""
    print("🚀 Phase 2.4: Write Feature Matrix to SQL Table features")
    
    try:
        # Step 1: Setup database connection
        engine, db_type = setup_database_connection()
        
        # Step 2: Load and prepare data (rerun full pipeline)
        df_raw = load_data_from_sql(engine)
        df_hourly = prepare_hourly_data(df_raw)
        
        # Step 3: Generate sliding windows
        windows = generate_sliding_windows(df_hourly, window_size=24)
        
        # Step 4: Compute technical features
        df_features = compute_technical_features(windows)
        
        # Step 5: Write feature matrix to SQL (Phase 2.4)
        success = write_feature_matrix_to_sql(df_features, engine, table_name="features")
        
        if success:
            print(f"\n✅ Phase 2.4 COMPLETE: Feature matrix successfully stored in {db_type.upper()} database")
            print(f"✅ ALL PHASE 2 STEPS COMPLETE!")
            print(f"\n🎯 ETL Pipeline Summary:")
            print(f"   ✅ 2.1: Load Data from SQL - {len(df_raw)} raw records")
            print(f"   ✅ 2.2: Generate Sliding Windows - {len(windows)} windows")
            print(f"   ✅ 2.3: Compute Technical Features - {len(df_features.columns)} features")
            print(f"   ✅ 2.4: Write Feature Matrix to SQL - features table created")
            print(f"\n🚀 Ready for Phase 3: Forecasting Model (LSTM Training)")
        else:
            print(f"\n❌ Phase 2.4 failed: Database operations unsuccessful")
            
        return df_features, engine
        
    except Exception as e:
        print(f"❌ Phase 2.4 failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    df_features, engine = main()
