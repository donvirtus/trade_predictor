#!/usr/bin/env python3
"""
Dynamic Threshold Auto-Balancer
Automatically finds optimal threshold to achieve target class distribution
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import sys
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

# Simple print-based logging to avoid circular imports
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

logger = SimpleLogger()

class ThresholdAutoBalancer:
    """
    Automatically finds optimal threshold to achieve target class distribution
    """
    
    def __init__(self, target_distribution: Dict[str, float] = None):
        """
        Initialize auto-balancer
        
        Args:
            target_distribution: Target percentages for each class
                Default: {'UP': 33.0, 'SIDEWAYS': 34.0, 'DOWN': 33.0}
        """
        if target_distribution is None:
            self.target_distribution = {
                'UP': 33.0,      # Target 33% UP signals
                'SIDEWAYS': 34.0, # Target 34% SIDEWAYS (slightly higher for stability)  
                'DOWN': 33.0     # Target 33% DOWN signals
            }
        else:
            self.target_distribution = target_distribution
            
        self.tolerance = 3.0  # ±3% tolerance for "good enough"
        
    def calculate_distribution(self, returns: pd.Series, threshold: float) -> Dict[str, float]:
        """Calculate class distribution for given threshold"""
        up = (returns > threshold).sum()
        down = (returns < -threshold).sum()
        sideways = ((returns >= -threshold) & (returns <= threshold)).sum()
        total = len(returns)
        
        return {
            'UP': (up / total) * 100,
            'SIDEWAYS': (sideways / total) * 100,
            'DOWN': (down / total) * 100
        }
    
    def calculate_distribution_score(self, actual: Dict[str, float]) -> float:
        """
        Calculate how close actual distribution is to target
        Returns score 0-100 (100 = perfect match)
        """
        deviations = []
        for class_name in ['UP', 'SIDEWAYS', 'DOWN']:
            target_pct = self.target_distribution[class_name]
            actual_pct = actual[class_name]
            deviation = abs(actual_pct - target_pct)
            deviations.append(deviation)
        
        avg_deviation = sum(deviations) / len(deviations)
        # Convert to score (0-100, lower deviation = higher score)
        score = max(0, 100 - (avg_deviation * 2))  # Scale factor
        return score
    
    def binary_search_threshold(self, returns: pd.Series, 
                               min_threshold: float = 0.1, 
                               max_threshold: float = 5.0,
                               max_iterations: int = 50) -> Tuple[float, Dict[str, float], float]:
        """
        Use binary search to find optimal threshold
        
        Returns:
            Tuple of (optimal_threshold, distribution, score)
        """
        best_threshold = 1.0
        best_score = 0
        best_distribution = {}
        
        low = min_threshold
        high = max_threshold
        
        for iteration in range(max_iterations):
            mid = (low + high) / 2
            
            distribution = self.calculate_distribution(returns, mid)
            score = self.calculate_distribution_score(distribution)
            
            # Update best if this is better
            if score > best_score:
                best_score = score
                best_threshold = mid
                best_distribution = distribution.copy()
            
            # Check if we're close enough to target
            up_diff = distribution['UP'] - self.target_distribution['UP']
            down_diff = distribution['DOWN'] - self.target_distribution['DOWN']
            
            # If we're within tolerance, we can stop
            if abs(up_diff) <= self.tolerance and abs(down_diff) <= self.tolerance:
                logger.info(f"✅ Converged in {iteration+1} iterations")
                break
            
            # Adjust search range based on UP vs DOWN balance
            # If too many UP signals, increase threshold
            if up_diff > 0:  # Too many UP signals
                low = mid
            else:  # Too few UP signals  
                high = mid
                
            # Prevent infinite loop with very small ranges
            if abs(high - low) < 0.001:
                break
        
        return best_threshold, best_distribution, best_score
    
    def grid_search_threshold(self, returns: pd.Series,
                             threshold_range: List[float] = None) -> Tuple[float, Dict[str, float], float]:
        """
        Grid search for optimal threshold (more thorough but slower)
        """
        if threshold_range is None:
            threshold_range = [x/10 for x in range(1, 51)]  # 0.1 to 5.0 by 0.1
        
        best_threshold = 1.0
        best_score = 0
        best_distribution = {}
        
        for threshold in threshold_range:
            distribution = self.calculate_distribution(returns, threshold)
            score = self.calculate_distribution_score(distribution)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_distribution = distribution.copy()
        
        return best_threshold, best_distribution, best_score
    
    def auto_balance_timeframe(self, df: pd.DataFrame, horizon: int = 20, 
                              method: str = 'binary_search') -> Dict:
        """
        Auto-balance threshold for specific timeframe data
        
        Args:
            df: DataFrame with OHLCV data
            horizon: Future return horizon  
            method: 'binary_search' or 'grid_search'
            
        Returns:
            Dict with optimal threshold and analysis
        """
        try:
            # Calculate returns
            returns = df['close'].pct_change(horizon).shift(-horizon) * 100
            returns = returns.dropna()
            
            if len(returns) < 100:
                logger.warning(f"Insufficient data: {len(returns)} rows")
                return {}
            
            logger.info(f"🔍 Auto-balancing with {len(returns)} samples using {method}")
            
            # Find optimal threshold
            if method == 'binary_search':
                optimal_threshold, distribution, score = self.binary_search_threshold(returns)
            else:
                optimal_threshold, distribution, score = self.grid_search_threshold(returns)
            
            # Calculate additional metrics
            trading_pct = distribution['UP'] + distribution['DOWN']
            
            result = {
                'optimal_threshold': round(optimal_threshold, 3),
                'distribution': {
                    'UP': round(distribution['UP'], 1),
                    'SIDEWAYS': round(distribution['SIDEWAYS'], 1), 
                    'DOWN': round(distribution['DOWN'], 1)
                },
                'balance_score': round(score, 1),
                'trading_percentage': round(trading_pct, 1),
                'total_samples': len(returns),
                'method': method,
                'target_distribution': self.target_distribution.copy()
            }
            
            logger.info(f"✅ Optimal threshold: {optimal_threshold:.3f}%")
            logger.info(f"📊 Distribution: UP={distribution['UP']:.1f}%, SIDE={distribution['SIDEWAYS']:.1f}%, DOWN={distribution['DOWN']:.1f}%")
            logger.info(f"🎯 Balance score: {score:.1f}/100")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in auto-balancing: {e}")
            return {}
    
    def save_auto_balance_config(self, results: Dict, output_file: str = 'data/auto_balanced_thresholds.json'):
        """Save auto-balanced thresholds to config file"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Add metadata
            results['generated_at'] = datetime.now().isoformat()
            results['balancer_version'] = '1.0'
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"💾 Auto-balanced config saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")

def load_data_for_balancing(db_path: str, limit_rows: int = None) -> pd.DataFrame:
    """Load data for threshold balancing"""
    try:
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            return pd.DataFrame()
        
        query = "SELECT timestamp, close FROM features ORDER BY timestamp"
        if limit_rows:
            query += f" DESC LIMIT {limit_rows}"
            
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
        
        if limit_rows:
            df = df.sort_values('timestamp')  # Re-sort if we used DESC LIMIT
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {db_path}: {e}")
        return pd.DataFrame()

def main():
    """Main auto-balancing function"""
    print("🎯 THRESHOLD AUTO-BALANCER")
    print("=" * 50)
    print("Automatically finds optimal threshold for balanced class distribution")
    print()
    
    # Initialize balancer
    balancer = ThresholdAutoBalancer()
    
    print(f"📊 Target Distribution:")
    for class_name, pct in balancer.target_distribution.items():
        print(f"   {class_name}: {pct}%")
    print()
    
    # Test with different datasets
    test_sources = [
        ('data/db/btc_1h.sqlite', 'BTC 1h'),
        ('data/db/eth_1h.sqlite', 'ETH 1h'), 
        ('data/db/doge_1h.sqlite', 'DOGE 1h'),
        ('data/db/btc_5m.sqlite', 'BTC 5m'),
        ('data/db/eth_5m.sqlite', 'ETH 5m'),
    ]
    
    all_results = {}
    
    for db_path, source_name in test_sources:
        if not os.path.exists(db_path):
            print(f"⚠️  Skipping {source_name}: Database not found")
            continue
            
        print(f"🔄 Processing {source_name}...")
        
        # Load data
        df = load_data_for_balancing(db_path, limit_rows=2000)  # Last 2000 candles
        
        if df.empty:
            print(f"   ❌ No data available")
            continue
        
        # Auto-balance
        result = balancer.auto_balance_timeframe(df, horizon=20, method='binary_search')
        
        if result:
            all_results[source_name] = result
            print(f"   ✅ Optimal: {result['optimal_threshold']}% | Score: {result['balance_score']}/100")
            print(f"      Distribution: UP={result['distribution']['UP']}%, SIDE={result['distribution']['SIDEWAYS']}%, DOWN={result['distribution']['DOWN']}%")
        else:
            print(f"   ❌ Auto-balancing failed")
        
        print()
    
    if all_results:
        # Calculate average optimal threshold
        avg_threshold = sum(r['optimal_threshold'] for r in all_results.values()) / len(all_results)
        
        print("=" * 60)
        print("📈 AUTO-BALANCING SUMMARY")
        print("=" * 60)
        print(f"🎯 Average Optimal Threshold: {avg_threshold:.3f}%")
        print()
        
        print("📊 Individual Results:")
        for source, result in all_results.items():
            print(f"   {source:<12}: {result['optimal_threshold']:>6.3f}% (score: {result['balance_score']:>5.1f})")
        
        # Save comprehensive results
        summary = {
            'individual_results': all_results,
            'average_optimal_threshold': round(avg_threshold, 3),
            'recommendation': {
                'threshold': round(avg_threshold, 3),
                'description': f"Auto-calculated optimal threshold for balanced distribution"
            }
        }
        
        balancer.save_auto_balance_config(summary)
        
        print(f"\n💾 Complete results saved to data/auto_balanced_thresholds.json")
        print(f"\n🎯 RECOMMENDATION: Use {avg_threshold:.3f}% as baseline threshold")
        
    else:
        print("❌ No successful auto-balancing results!")

if __name__ == '__main__':
    main()