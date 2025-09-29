#!/usr/bin/env python3
"""
Threshold Validation Script
Validasi empiris untuk menentukan optimal threshold berdasarkan real data
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from utils.logging import get_logger
from utils.config import load_config

logger = get_logger('threshold_validation')

def load_real_data(db_path: str, limit_rows: int = None) -> pd.DataFrame:
    """Load real market data from database"""
    try:
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            return pd.DataFrame()
        
        query = "SELECT timestamp, open, high, low, close, volume FROM features ORDER BY timestamp"
        if limit_rows:
            query += f" LIMIT {limit_rows}"
            
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
        
        if df.empty:
            return df
            
        # Calculate returns untuk berbagai horizons
        for horizon in [5, 10, 20, 30]:
            df[f'return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon) * 100
            
        return df.dropna()
        
    except Exception as e:
        logger.error(f"Error loading data from {db_path}: {e}")
        return pd.DataFrame()

def calculate_class_distribution(returns: pd.Series, threshold: float) -> dict:
    """Calculate class distribution for given threshold"""
    up = (returns > threshold).sum()
    down = (returns < -threshold).sum() 
    sideways = ((returns >= -threshold) & (returns <= threshold)).sum()
    total = len(returns)
    
    return {
        'UP': {'count': up, 'pct': up/total*100},
        'SIDEWAYS': {'count': sideways, 'pct': sideways/total*100},
        'DOWN': {'count': down, 'pct': down/total*100},
        'total': total
    }

def calculate_balance_score(distribution: dict) -> float:
    """
    Calculate balance score (0-100)
    Higher score = better balance
    Perfect balance (33.33% each) = 100
    """
    up_pct = distribution['UP']['pct']
    side_pct = distribution['SIDEWAYS']['pct'] 
    down_pct = distribution['DOWN']['pct']
    
    # Ideal distribution: 33.33% each
    ideal = 33.33
    
    # Calculate deviation from ideal
    up_dev = abs(up_pct - ideal)
    side_dev = abs(side_pct - ideal) 
    down_dev = abs(down_pct - ideal)
    
    # Average deviation (lower is better)
    avg_deviation = (up_dev + side_dev + down_dev) / 3
    
    # Convert to score (0-100, higher is better)
    balance_score = max(0, 100 - (avg_deviation * 3))  # Scale factor
    
    return balance_score

def calculate_trading_efficiency(distribution: dict) -> float:
    """
    Calculate trading efficiency score
    Reward reasonable amount of trading signals (not too many, not too few)
    """
    trading_pct = distribution['UP']['pct'] + distribution['DOWN']['pct']
    
    # Optimal trading percentage: 60-70%
    if 60 <= trading_pct <= 70:
        efficiency = 100
    elif 50 <= trading_pct < 60:
        efficiency = 80 + (trading_pct - 50) * 2  # 80-100
    elif 70 < trading_pct <= 80:
        efficiency = 100 - (trading_pct - 70) * 2  # 100-80
    elif 40 <= trading_pct < 50:
        efficiency = 60 + (trading_pct - 40) * 2   # 60-80
    elif 80 < trading_pct <= 90:
        efficiency = 80 - (trading_pct - 80) * 2   # 80-60
    else:
        # Too extreme
        efficiency = max(0, 60 - abs(trading_pct - 65) * 2)
    
    return efficiency

def validate_threshold_range(data_sources: dict, horizons: list = [20], 
                           threshold_range: list = None) -> pd.DataFrame:
    """
    Validate threshold range across multiple data sources and horizons
    """
    if threshold_range is None:
        threshold_range = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    
    results = []
    
    for source_name, df in data_sources.items():
        if df.empty:
            continue
            
        for horizon in horizons:
            return_col = f'return_{horizon}'
            if return_col not in df.columns:
                continue
                
            returns = df[return_col].dropna()
            if len(returns) < 100:  # Minimum data requirement
                continue
            
            for threshold in threshold_range:
                dist = calculate_class_distribution(returns, threshold)
                balance_score = calculate_balance_score(dist)
                efficiency_score = calculate_trading_efficiency(dist)
                
                # Combined score (weighted)
                combined_score = balance_score * 0.6 + efficiency_score * 0.4
                
                results.append({
                    'source': source_name,
                    'horizon': horizon,
                    'threshold': threshold,
                    'up_pct': dist['UP']['pct'],
                    'sideways_pct': dist['SIDEWAYS']['pct'],
                    'down_pct': dist['DOWN']['pct'],
                    'balance_score': balance_score,
                    'efficiency_score': efficiency_score,
                    'combined_score': combined_score,
                    'total_samples': dist['total']
                })
    
    return pd.DataFrame(results)

def find_optimal_thresholds(results_df: pd.DataFrame) -> dict:
    """Find optimal thresholds based on different criteria"""
    if results_df.empty:
        return {}
    
    optimal = {}
    
    # Best overall (combined score)
    best_overall = results_df.loc[results_df['combined_score'].idxmax()]
    optimal['best_overall'] = {
        'threshold': best_overall['threshold'],
        'score': best_overall['combined_score'],
        'distribution': f"UP: {best_overall['up_pct']:.1f}%, SIDE: {best_overall['sideways_pct']:.1f}%, DOWN: {best_overall['down_pct']:.1f}%"
    }
    
    # Best balance
    best_balance = results_df.loc[results_df['balance_score'].idxmax()]
    optimal['best_balance'] = {
        'threshold': best_balance['threshold'],
        'score': best_balance['balance_score'],
        'distribution': f"UP: {best_balance['up_pct']:.1f}%, SIDE: {best_balance['sideways_pct']:.1f}%, DOWN: {best_balance['down_pct']:.1f}%"
    }
    
    # Best efficiency
    best_efficiency = results_df.loc[results_df['efficiency_score'].idxmax()]
    optimal['best_efficiency'] = {
        'threshold': best_efficiency['threshold'],
        'score': best_efficiency['efficiency_score'],
        'distribution': f"UP: {best_efficiency['up_pct']:.1f}%, SIDE: {best_efficiency['sideways_pct']:.1f}%, DOWN: {best_efficiency['down_pct']:.1f}%"
    }
    
    return optimal

def create_validation_plots(results_df: pd.DataFrame, output_dir: str = 'data/plots'):
    """Create validation plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    if results_df.empty:
        logger.warning("No data for plotting")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Score comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Threshold Validation Analysis', fontsize=16, fontweight='bold')
    
    # Group by threshold for plotting
    threshold_summary = results_df.groupby('threshold').agg({
        'balance_score': 'mean',
        'efficiency_score': 'mean', 
        'combined_score': 'mean',
        'up_pct': 'mean',
        'sideways_pct': 'mean',
        'down_pct': 'mean'
    }).reset_index()
    
    # Plot 1: Balance Score
    axes[0,0].plot(threshold_summary['threshold'], threshold_summary['balance_score'], 
                   marker='o', linewidth=2, markersize=6)
    axes[0,0].set_title('Balance Score vs Threshold', fontweight='bold')
    axes[0,0].set_xlabel('Threshold (%)')
    axes[0,0].set_ylabel('Balance Score')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Current (1.0%)')
    axes[0,0].legend()
    
    # Plot 2: Efficiency Score
    axes[0,1].plot(threshold_summary['threshold'], threshold_summary['efficiency_score'], 
                   marker='o', linewidth=2, markersize=6, color='orange')
    axes[0,1].set_title('Trading Efficiency vs Threshold', fontweight='bold')
    axes[0,1].set_xlabel('Threshold (%)')
    axes[0,1].set_ylabel('Efficiency Score')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Current (1.0%)')
    axes[0,1].legend()
    
    # Plot 3: Combined Score
    axes[1,0].plot(threshold_summary['threshold'], threshold_summary['combined_score'], 
                   marker='o', linewidth=2, markersize=6, color='green')
    axes[1,0].set_title('Combined Score vs Threshold', fontweight='bold')
    axes[1,0].set_xlabel('Threshold (%)')
    axes[1,0].set_ylabel('Combined Score')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Current (1.0%)')
    axes[1,0].legend()
    
    # Plot 4: Class Distribution
    axes[1,1].plot(threshold_summary['threshold'], threshold_summary['up_pct'], 
                   marker='o', label='UP', linewidth=2)
    axes[1,1].plot(threshold_summary['threshold'], threshold_summary['sideways_pct'], 
                   marker='s', label='SIDEWAYS', linewidth=2)
    axes[1,1].plot(threshold_summary['threshold'], threshold_summary['down_pct'], 
                   marker='^', label='DOWN', linewidth=2)
    axes[1,1].set_title('Class Distribution vs Threshold', fontweight='bold')
    axes[1,1].set_xlabel('Threshold (%)')
    axes[1,1].set_ylabel('Percentage (%)')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    axes[1,1].axhline(y=33.33, color='gray', linestyle=':', alpha=0.5, label='Ideal (33.33%)')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/threshold_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {output_dir}/threshold_validation.png")

def main():
    """Main validation function"""
    print("🎯 THRESHOLD VALIDATION ANALYSIS")
    print("=" * 50)
    
    # Load config
    cfg = load_config('config/config.yaml')
    
    # Define data sources
    data_sources = {}
    db_base = 'data/db'
    
    # Load data from existing databases
    for pair in ['btc', 'eth', 'doge']:  # Map to our database naming
        for tf in ['5m', '15m', '1h']:
            db_path = f'{db_base}/{pair}_{tf}.sqlite'
            if os.path.exists(db_path):
                print(f"📊 Loading data from {pair}_{tf}...")
                df = load_real_data(db_path, limit_rows=5000)  # Last 5k candles
                if not df.empty:
                    data_sources[f'{pair}_{tf}'] = df
                    print(f"   ✅ Loaded {len(df)} rows")
                else:
                    print(f"   ❌ No data loaded")
            else:
                print(f"   ⚠️  Database not found: {db_path}")
    
    if not data_sources:
        print("❌ No data sources available. Run build_dataset.py first!")
        return
    
    print(f"\n🔍 Analyzing {len(data_sources)} data sources...")
    
    # Run validation
    results_df = validate_threshold_range(
        data_sources, 
        horizons=[20],  # Focus on main horizon
        threshold_range=[0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    )
    
    if results_df.empty:
        print("❌ No validation results generated!")
        return
    
    # Find optimal thresholds
    optimal = find_optimal_thresholds(results_df)
    
    # Print results
    print("\n" + "="*60)
    print("📈 VALIDATION RESULTS")
    print("="*60)
    
    # Current threshold analysis
    current_results = results_df[results_df['threshold'] == 1.0]
    if not current_results.empty:
        avg_scores = current_results.agg({
            'balance_score': 'mean',
            'efficiency_score': 'mean',
            'combined_score': 'mean'
        })
        print(f"\n🎯 CURRENT THRESHOLD (1.0%):")
        print(f"   Balance Score: {avg_scores['balance_score']:.1f}/100")
        print(f"   Efficiency Score: {avg_scores['efficiency_score']:.1f}/100") 
        print(f"   Combined Score: {avg_scores['combined_score']:.1f}/100")
    
    # Optimal thresholds
    print(f"\n🏆 OPTIMAL THRESHOLDS:")
    for criterion, info in optimal.items():
        print(f"\n   {criterion.upper().replace('_', ' ')}:")
        print(f"     Threshold: {info['threshold']}%")
        print(f"     Score: {info['score']:.1f}")
        print(f"     Distribution: {info['distribution']}")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    summary = results_df.groupby('threshold').agg({
        'combined_score': ['mean', 'std'],
        'balance_score': 'mean',
        'efficiency_score': 'mean'
    }).round(1)
    print(summary.to_string())
    
    # Create plots
    create_validation_plots(results_df)
    
    # Save detailed results
    results_df.to_csv('data/threshold_validation_results.csv', index=False)
    
    with open('data/threshold_optimal_summary.json', 'w') as f:
        json.dump(optimal, f, indent=2)
    
    print(f"\n💾 Detailed results saved to:")
    print(f"   - data/threshold_validation_results.csv")
    print(f"   - data/threshold_optimal_summary.json")
    print(f"   - data/plots/threshold_validation.png")
    
    # Recommendation
    if optimal:
        best_threshold = optimal['best_overall']['threshold']
        current_threshold = 1.0
        
        print(f"\n🎯 RECOMMENDATION:")
        if abs(best_threshold - current_threshold) < 0.2:
            print(f"   ✅ Current threshold ({current_threshold}%) is OPTIMAL!")
            print(f"   📈 Continue using {current_threshold}% threshold")
        else:
            print(f"   🔄 Consider adjusting threshold from {current_threshold}% to {best_threshold}%")
            print(f"   📈 Expected improvement: {optimal['best_overall']['score']:.1f} combined score")

if __name__ == '__main__':
    main()