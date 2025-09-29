#!/usr/bin/env python3
"""
Test Impact dari Auto-Calculated Multipliers pada Target Distribution

Script ini membandingkan distribusi target antara:
1. Manual multipliers (existing config)  
2. Auto-calculated multipliers (new system)
3. No adaptive thresholds (baseline)

Usage:
  python test_target_distribution_impact.py --timeframe 5m
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.config import load_config
from utils.adaptive_thresholds import get_adaptive_multipliers
from features.targets import label_multi_horizon_directions


def setup_test_data(timeframe='5m', n_rows=1000):
    """Setup test data dari database"""
    import sqlite3
    
    # Load from database
    db_path = f"data/db/btc_{timeframe}.sqlite"
    if not os.path.exists(db_path):
        print(f"❌ Database tidak ditemukan: {db_path}")
        return None
    
    conn = sqlite3.connect(db_path)
    query = f"SELECT timestamp, open, high, low, close, volume FROM features ORDER BY timestamp DESC LIMIT {n_rows}"
    df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
    conn.close()
    
    if df.empty:
        print(f"❌ Tidak ada data di {db_path}")
        return None
    
    # Sort ascending for proper time series
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"✅ Loaded {len(df)} rows untuk timeframe {timeframe}")
    
    return df


def test_distribution_with_config(df, timeframe, config_modifications, label="Test"):
    """Test target distribution dengan config modifications tertentu"""
    
    # Load base config
    cfg = load_config("config/config.yaml")
    
    # Apply modifications
    for key_path, value in config_modifications.items():
        keys = key_path.split('.')
        obj = cfg
        for key in keys[:-1]:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            else:
                obj = obj[key]
        
        if hasattr(obj, keys[-1]):
            setattr(obj, keys[-1], value)
        else:
            obj[keys[-1]] = value
    
    # Apply target labeling
    horizons = [5, 20]  # Test multiple horizons
    df_labeled = label_multi_horizon_directions(
        df.copy(), 
        horizons, 
        sideways_threshold_pct=1.0, 
        timeframe=timeframe, 
        cfg=cfg
    )
    
    # Calculate distributions
    distributions = {}
    for h in horizons:
        col = f'direction_h{h}'
        if col in df_labeled.columns:
            dist = df_labeled[col].value_counts(normalize=True) * 100
            distributions[f'h{h}'] = {
                'DOWN': dist.get(0, 0),
                'SIDEWAYS': dist.get(1, 0), 
                'UP': dist.get(2, 0),
                'total_rows': len(df_labeled[col].dropna())
            }
    
    print(f"\n📊 {label} - Timeframe: {timeframe}")
    for horizon, dist in distributions.items():
        print(f"   {horizon}: DOWN={dist['DOWN']:.1f}% SIDEWAYS={dist['SIDEWAYS']:.1f}% UP={dist['UP']:.1f}% (n={dist['total_rows']})")
    
    return distributions


def main():
    parser = argparse.ArgumentParser(description="Test target distribution impact dari auto-calculated multipliers")
    parser.add_argument('--timeframe', type=str, default='5m', 
                       help='Timeframe to test (default: 5m)')
    parser.add_argument('--rows', type=int, default=2000,
                       help='Number of rows to test (default: 2000)')
    
    args = parser.parse_args()
    
    print(f"🧪 Testing Target Distribution Impact")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Test rows: {args.rows}")
    
    # Setup data
    df = setup_test_data(args.timeframe, args.rows)
    if df is None:
        return
    
    # Get auto-calculated multipliers for comparison
    cfg = load_config("config/config.yaml")
    timeframes = ['5m', '15m', '1h', '2h', '4h', '6h']
    auto_multipliers = get_adaptive_multipliers(cfg, timeframes, force_recalculate=True)
    
    print(f"\n📋 Auto-calculated multipliers:")
    for tf, mult in sorted(auto_multipliers.items()):
        print(f"   {tf}: {mult:.3f}x")
    
    # Test 1: No adaptive thresholds (baseline)
    print(f"\n{'='*60}")
    print("TEST 1: No Adaptive Thresholds (Baseline)")
    print("="*60)
    
    baseline_dist = test_distribution_with_config(
        df, args.timeframe,
        {'target.adaptive_thresholds.enabled': False},
        "Baseline (No Adaptive)"
    )
    
    # Test 2: Manual multipliers (existing)  
    print(f"\n{'='*60}")
    print("TEST 2: Manual Multipliers (Current)")
    print("="*60)
    
    manual_dist = test_distribution_with_config(
        df, args.timeframe,
        {
            'target.adaptive_thresholds.enabled': True,
            'target.adaptive_thresholds.method': 'manual'
        },
        "Manual Multipliers"
    )
    
    # Test 3: Auto-calculated multipliers
    print(f"\n{'='*60}")
    print("TEST 3: Auto-Calculated Multipliers (New)")
    print("="*60)
    
    auto_dist = test_distribution_with_config(
        df, args.timeframe,
        {
            'target.adaptive_thresholds.enabled': True,
            'target.adaptive_thresholds.method': 'auto_volatility'
        },
        "Auto-Calculated Multipliers"
    )
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("📈 COMPARISON SUMMARY")
    print("="*60)
    
    for horizon in ['h5', 'h20']:
        if horizon in baseline_dist:
            print(f"\n🎯 Horizon {horizon[1:]}:")
            print(f"{'Method':<20} {'DOWN':<8} {'SIDEWAYS':<10} {'UP':<8} {'Balance Score'}")
            print("-" * 55)
            
            # Calculate balance score (lower is better)
            # Perfect balance: 33.33% each = score of 0
            # Score = sum of absolute deviations from 33.33%
            
            for method_name, dist in [
                ("Baseline", baseline_dist),
                ("Manual", manual_dist), 
                ("Auto-Calculated", auto_dist)
            ]:
                if horizon in dist:
                    d = dist[horizon]
                    target = 33.33
                    balance_score = abs(d['DOWN'] - target) + abs(d['SIDEWAYS'] - target) + abs(d['UP'] - target)
                    
                    print(f"{method_name:<20} {d['DOWN']:<8.1f} {d['SIDEWAYS']:<10.1f} {d['UP']:<8.1f} {balance_score:<8.1f}")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATIONS:")
    print("- Lower balance score = better distribution balance")
    print("- Look for ~33% each class for optimal ML training")
    print("- Manual vs Auto-calculated trade-offs:")
    print("  * Manual: Based on intuition, may not adapt to market changes")
    print("  * Auto: Data-driven, adapts to recent volatility patterns")
    

if __name__ == '__main__':
    main()