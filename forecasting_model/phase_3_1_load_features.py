# phase_3_1_load_features.py - Load Feature Matrix into ML Training Script
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from common.db import setup_database_connection
from sklearn.preprocessing import StandardScaler

load_dotenv()

def load_feature_matrix_from_sql(engine, table_name="features"):
    """
    Phase 3.1: Load Feature Matrix into ML Training Script
    Test Case: Script reads rows, shape is (N, M)
    """
    print(f"🔄 Phase 3.1: Loading feature matrix from SQL table '{table_name}'...")
    
    try:
        # Load all features from the database
        query = f"""
        SELECT * FROM {table_name}
        ORDER BY window_id
        """
        
        df_features = pd.read_sql(query, engine)
        
        print(f"✅ Loaded feature matrix from database")
        print(f"   Records: {len(df_features)}")
        print(f"   Columns: {len(df_features.columns)}")
        
        # Convert target_time to datetime
        df_features['target_time'] = pd.to_datetime(df_features['target_time'])
        
        # Test Case: Script reads rows, shape is (N, M)
        N, M = df_features.shape
        test_passed = N > 0 and M > 0
        
        print(f"\n🧪 Test Case - Script reads rows, shape is (N, M):")
        print(f"   Shape: ({N}, {M})")
        print(f"   N (rows): {N}")
        print(f"   M (columns): {M}")
        print(f"   Result: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        # Show feature matrix summary
        print(f"\n📊 Feature Matrix Summary:")
        print(f"   Time range: {df_features['target_time'].min()} to {df_features['target_time'].max()}")
        print(f"   Target price range: ${df_features['target_price'].min():.2f} - ${df_features['target_price'].max():.2f}")
        print(f"   Available features: {list(df_features.columns)}")
        
        # Show data types
        print(f"\n📋 Data Types:")
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        print(f"   Numeric columns: {len(numeric_cols)}")
        print(f"   First 10 numeric: {numeric_cols[:10]}")
        
        return df_features
        
    except Exception as e:
        print(f"❌ Failed to load feature matrix: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_ml_data(df_features):
    """Prepare data for ML training by extracting features and targets"""
    print(f"🔄 Preparing data for ML training...")
    
    # Define feature columns (exclude metadata and target)
    exclude_cols = ['window_id', 'target_time', 'target_price', 'price_sequence_json']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    print(f"📊 Feature columns for ML ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols):
        if i < 10:  # Show first 10
            print(f"   {i+1:2d}. {col}")
        elif i == 10:
            print(f"   ... and {len(feature_cols)-10} more features")
    
    # Extract feature matrix (X) and target vector (y)
    X = df_features[feature_cols].values
    y = df_features['target_price'].values
    
    # Extract price sequences for LSTM (if available)
    price_sequences = []
    if 'price_sequence_json' in df_features.columns:
        print(f"🔄 Extracting price sequences for LSTM...")
        for seq_json in df_features['price_sequence_json']:
            try:
                seq = json.loads(seq_json)
                price_sequences.append(seq)
            except:
                # Fallback: create dummy sequence
                price_sequences.append([df_features.loc[len(price_sequences), 'target_price']] * 24)
        
        price_sequences = np.array(price_sequences)
        print(f"✅ Extracted {len(price_sequences)} price sequences of length {price_sequences.shape[1]}")
    else:
        print(f"⚠️ No price sequences found, using feature matrix only")
        price_sequences = None
    
    print(f"\n📈 ML Data Summary:")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target vector shape: {y.shape}")
    if price_sequences is not None:
        print(f"   Price sequences shape: {price_sequences.shape}")
    print(f"   Target statistics:")
    print(f"     Mean: ${np.mean(y):.2f}")
    print(f"     Std:  ${np.std(y):.2f}")
    print(f"     Min:  ${np.min(y):.2f}")
    print(f"     Max:  ${np.max(y):.2f}")
    
    return X, y, price_sequences, feature_cols

def create_train_test_split(X, y, price_sequences=None, test_size=0.2):
    """Split chronologically: the test set is the most recent `test_size` share.

    Rows arrive ordered by window_id, which the feature builder assigns in
    time order, so slicing off the tail is a time-based split.

    This deliberately does not shuffle. A shuffled split on a time series lets
    the model train on later hours and be scored on earlier ones, so it sees
    the future and every score it produces is inflated.
    """
    print(f"🔄 Creating chronological train/test split (test_size={test_size})...")

    split_idx = int(len(X) * (1 - test_size))
    if split_idx < 1 or split_idx >= len(X):
        raise ValueError(f"Cannot split {len(X)} samples at test_size={test_size}")

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if price_sequences is not None:
        seq_train, seq_test = price_sequences[:split_idx], price_sequences[split_idx:]
    else:
        seq_train, seq_test = None, None

    print(f"✅ Train/test split created:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Training targets range: ${np.min(y_train):.2f} - ${np.max(y_train):.2f}")
    print(f"   Test targets range: ${np.min(y_test):.2f} - ${np.max(y_test):.2f}")
    
    return X_train, X_test, y_train, y_test, seq_train, seq_test

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    print(f"🔄 Scaling features using StandardScaler...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Features scaled:")
    print(f"   Training mean: {np.mean(X_train_scaled, axis=0)[:3].round(3)} (first 3 features)")
    print(f"   Training std:  {np.std(X_train_scaled, axis=0)[:3].round(3)} (first 3 features)")
    
    return X_train_scaled, X_test_scaled, scaler

class PowerMarketDataset(Dataset):
    """PyTorch Dataset for power market data"""
    
    def __init__(self, X, y, sequences=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequences = torch.FloatTensor(sequences) if sequences is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.sequences is not None:
            return self.X[idx], self.sequences[idx], self.y[idx]
        else:
            return self.X[idx], self.y[idx]

def create_data_loaders(X_train, X_test, y_train, y_test, seq_train=None, seq_test=None, batch_size=8):
    """Create PyTorch DataLoaders"""
    print(f"🔄 Creating PyTorch DataLoaders (batch_size={batch_size})...")
    
    # Create datasets
    train_dataset = PowerMarketDataset(X_train, y_train, seq_train)
    test_dataset = PowerMarketDataset(X_test, y_test, seq_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ DataLoaders created:")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Test data loader
    print(f"\n🧪 Testing DataLoader:")
    for batch_idx, batch_data in enumerate(train_loader):
        if seq_train is not None:
            features, sequences, targets = batch_data
            print(f"   Batch {batch_idx}: features {features.shape}, sequences {sequences.shape}, targets {targets.shape}")
        else:
            features, targets = batch_data
            print(f"   Batch {batch_idx}: features {features.shape}, targets {targets.shape}")
        
        if batch_idx >= 2:  # Show first 3 batches
            break
    
    return train_loader, test_loader

def main():
    """Execute Phase 3.1 workflow"""
    print("🚀 Phase 3.1: Load Feature Matrix into ML Training Script")
    
    try:
        # Step 1: Setup database connection
        engine, db_type = setup_database_connection()
        
        # Step 2: Load feature matrix from SQL (Phase 3.1)
        df_features = load_feature_matrix_from_sql(engine, table_name="features")
        
        if df_features is None or len(df_features) == 0:
            print("❌ Phase 3.1 failed: Could not load feature matrix")
            return None
        
        # Step 3: Prepare ML data
        X, y, price_sequences, feature_cols = prepare_ml_data(df_features)
        
        # Step 4: Create train/test split
        X_train, X_test, y_train, y_test, seq_train, seq_test = create_train_test_split(
            X, y, price_sequences, test_size=0.2
        )
        
        # Step 5: Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Step 6: Create PyTorch DataLoaders
        train_loader, test_loader = create_data_loaders(
            X_train_scaled, X_test_scaled, y_train, y_test, 
            seq_train, seq_test, batch_size=8
        )
        
        print(f"\n✅ Phase 3.1 COMPLETE: Feature matrix successfully loaded for ML training")
        print(f"📊 Data ready for LSTM model:")
        print(f"   Training samples: {len(X_train_scaled)}")
        print(f"   Test samples: {len(X_test_scaled)}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Price sequences: {'Available' if seq_train is not None else 'Not available'}")
        print(f"🔄 Next: Phase 3.2 - Define LSTM Model Architecture")
        
        # Return all prepared data for next phase
        ml_data = {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'seq_train': seq_train,
            'seq_test': seq_test,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'n_features': len(feature_cols)
        }
        
        return ml_data
        
    except Exception as e:
        print(f"❌ Phase 3.1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    ml_data = main()
