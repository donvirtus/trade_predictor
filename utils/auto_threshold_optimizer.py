#!/usr/bin/env python3
"""
Auto-Threshold Mode Optimizer

Automatically selects and applies optimal threshold_mode after data collection.
Integrates with existing pipeline to provide intelligent threshold selection.
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

# Add project root to avoid import issues
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

def analyze_threshold_performance(db_path: str, timeframe: str) -> Dict:
    """
    Analyze threshold performance for specific database
    
    Returns:
        Dict with analysis results and recommendations
    """
    try:
        # Load data
        query = "SELECT timestamp, close FROM features ORDER BY timestamp DESC LIMIT 2000"
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
        
        if df.empty or len(df) < 100:
            return {'error': 'Insufficient data'}
        
        df = df.sort_values('timestamp')
        
        # Calculate returns for analysis
        returns = df['close'].pct_change(20).shift(-20) * 100
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 50:
            return {'error': 'Insufficient return data'}
        
        # Test different thresholds
        test_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        results = []
        
        for threshold in test_thresholds:
            up = (returns_clean > threshold).sum()
            down = (returns_clean < -threshold).sum()
            sideways = len(returns_clean) - up - down
            
            distribution = {
                'UP': (up / len(returns_clean)) * 100,
                'SIDEWAYS': (sideways / len(returns_clean)) * 100,
                'DOWN': (down / len(returns_clean)) * 100
            }
            
            # Calculate balance score (closer to 33.3/33.4/33.3 = better)
            balance_score = 100 - (
                abs(33.3 - distribution['UP']) + 
                abs(33.4 - distribution['SIDEWAYS']) + 
                abs(33.3 - distribution['DOWN'])
            )
            balance_score = max(0, balance_score)
            
            results.append({
                'threshold': threshold,
                'distribution': distribution,
                'balance_score': balance_score,
                'trading_signals': distribution['UP'] + distribution['DOWN']
            })
        
        # Find optimal threshold
        best_result = max(results, key=lambda x: x['balance_score'])
        
        # Volatility analysis
        recent_volatility = returns_clean.tail(100).std()
        overall_volatility = returns_clean.std()
        
        # Market trend analysis
        price_trend = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        
        return {
            'optimal_threshold': best_result['threshold'],
            'optimal_distribution': best_result['distribution'],
            'optimal_score': best_result['balance_score'],
            'all_results': results,
            'volatility_metrics': {
                'recent': recent_volatility,
                'overall': overall_volatility,
                'regime': 'high' if overall_volatility > 2.0 else 'medium' if overall_volatility > 1.0 else 'low'
            },
            'market_trend': {
                'direction': 'bullish' if price_trend > 5 else 'bearish' if price_trend < -5 else 'sideways',
                'magnitude': price_trend
            },
            'data_quality': {
                'samples': len(returns_clean),
                'timeframe': timeframe,
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}"
            }
        }
        
    except Exception as e:
        return {'error': str(e)}

def generate_optimal_config(analysis_result: Dict, pair: str, timeframe: str) -> Dict:
    """Generate optimal config.yaml section based on analysis"""
    
    if 'error' in analysis_result:
        return {'error': analysis_result['error']}
    
    optimal_threshold = analysis_result['optimal_threshold']
    optimal_score = analysis_result['optimal_score']
    volatility_regime = analysis_result['volatility_metrics']['regime']
    
    # Decision logic based on analysis
    if optimal_score > 90:
        # Excellent balance with auto-balanced approach
        recommended_mode = 'auto_balanced'
        config_section = {
            'threshold_mode': 'auto_balanced',
            'sideways_threshold_pct': optimal_threshold,
            'auto_balance': {
                'target_distribution': {
                    'UP': 33.0,
                    'SIDEWAYS': 34.0,
                    'DOWN': 33.0
                },
                'tolerance': 3.0,
                'fallback_threshold': optimal_threshold
            }
        }
        reason = f"Auto-balanced optimization achieves excellent balance (score: {optimal_score:.1f})"
        
    elif optimal_score > 75 and volatility_regime in ['medium', 'high']:
        # Good balance + volatile market = adaptive approach
        recommended_mode = 'adaptive'
        
        # Calculate multiplier based on optimal threshold
        base_threshold = 1.0
        multiplier = optimal_threshold / base_threshold
        
        config_section = {
            'threshold_mode': 'adaptive',
            'sideways_threshold_pct': base_threshold,
            'adaptive_thresholds': {
                'enabled': True,
                'method': 'auto_volatility',
                'auto_update': True,
                'update_frequency_days': 7,
                'volatility_calc': {
                    'method': 'rolling_std',
                    'window': 20,
                    'smoothing_factor': 0.7
                },
                'manual_multipliers': {
                    timeframe: round(multiplier, 3)
                }
            }
        }
        reason = f"Adaptive approach suitable for {volatility_regime} volatility market"
        
    else:
        # Conservative manual approach
        recommended_mode = 'adaptive'
        
        base_threshold = 1.0
        multiplier = optimal_threshold / base_threshold
        
        config_section = {
            'threshold_mode': 'adaptive',
            'sideways_threshold_pct': base_threshold,
            'adaptive_thresholds': {
                'enabled': True,
                'method': 'manual',
                'manual_multipliers': {
                    timeframe: round(multiplier, 3)
                }
            }
        }
        reason = f"Manual adaptive approach with optimized threshold"
    
    return {
        'recommended_mode': recommended_mode,
        'config_section': config_section,
        'reasoning': reason,
        'performance_improvement': f"Score improvement from current to optimal: {optimal_score:.1f}/100",
        'optimal_threshold': optimal_threshold,
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'pair': pair,
            'timeframe': timeframe,
            'data_samples': analysis_result['data_quality']['samples']
        }
    }

def save_optimization_results(results: Dict, output_file: str = 'data/threshold_optimization_results.json'):
    """Save optimization results for review"""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Optimization results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    """Main optimization function for CLI usage"""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Auto-Threshold Mode Optimizer')
    parser.add_argument('--db-path', help='Specific database path to analyze')
    parser.add_argument('--timeframe', default='15m', help='Timeframe to analyze')
    parser.add_argument('--pair', default='BTCUSDT', help='Trading pair')
    parser.add_argument('--apply-config', action='store_true', 
                       help='Apply optimal config to config.yaml (EXPERIMENTAL)')
    
    args = parser.parse_args()
    
    print("🎯 AUTO-THRESHOLD MODE OPTIMIZER")
    print("=" * 50)
    
    if args.db_path:
        # Analyze specific database
        db_files = [(args.db_path, args.pair)]
    else:
        # Auto-discover databases
        pattern = 'data/db/*_15m.sqlite'
        db_files = []
        for db_path in glob.glob(pattern):
            pair = os.path.basename(db_path).replace('_15m.sqlite', '').replace('_', '').upper() + 'USDT'
            if pair.endswith('USDTUSDT'):
                pair = pair.replace('USDTUSDT', 'USDT')
            db_files.append((db_path, pair))
    
    if not db_files:
        print("❌ No databases found for analysis")
        return
    
    all_results = {}
    
    for db_path, pair in db_files:
        if not os.path.exists(db_path):
            print(f"⚠️  Skipping {pair}: Database not found")
            continue
        
        print(f"\n🔍 Analyzing {pair} {args.timeframe}...")
        print("-" * 40)
        
        # Perform analysis
        analysis = analyze_threshold_performance(db_path, args.timeframe)
        
        if 'error' in analysis:
            print(f"❌ Analysis failed: {analysis['error']}")
            continue
        
        # Generate optimal config
        optimization = generate_optimal_config(analysis, pair, args.timeframe)
        
        if 'error' in optimization:
            print(f"❌ Config generation failed: {optimization['error']}")
            continue
        
        # Display results
        print(f"📊 ANALYSIS RESULTS:")
        print(f"   Optimal Threshold: {analysis['optimal_threshold']:.3f}%")
        print(f"   Balance Score: {analysis['optimal_score']:.1f}/100")
        
        dist = analysis['optimal_distribution']
        print(f"   Distribution: UP={dist['UP']:.1f}% | SIDE={dist['SIDEWAYS']:.1f}% | DOWN={dist['DOWN']:.1f}%")
        
        vol = analysis['volatility_metrics']
        print(f"   Volatility: {vol['regime']} ({vol['overall']:.3f})")
        
        market = analysis['market_trend']
        print(f"   Market Trend: {market['direction']} ({market['magnitude']:+.1f}%)")
        
        print(f"\n🏆 RECOMMENDATION:")
        print(f"   Mode: {optimization['recommended_mode']}")
        print(f"   Reasoning: {optimization['reasoning']}")
        print(f"   Performance: {optimization['performance_improvement']}")
        
        # Store results
        all_results[f"{pair}_{args.timeframe}"] = {
            'analysis': analysis,
            'optimization': optimization
        }
    
    if all_results:
        # Save comprehensive results
        save_optimization_results(all_results)
        
        # Summary
        print(f"\n📈 OPTIMIZATION SUMMARY:")
        print("=" * 30)
        
        total_pairs = len(all_results)
        modes_recommended = {}
        
        for result in all_results.values():
            mode = result['optimization']['recommended_mode']
            modes_recommended[mode] = modes_recommended.get(mode, 0) + 1
        
        print(f"Total pairs analyzed: {total_pairs}")
        print(f"Mode recommendations:")
        for mode, count in modes_recommended.items():
            print(f"  {mode}: {count} pairs ({count/total_pairs*100:.1f}%)")
        
        # Overall best threshold
        avg_threshold = sum(r['optimization']['optimal_threshold'] for r in all_results.values()) / len(all_results)
        print(f"\nAverage optimal threshold: {avg_threshold:.3f}%")
        
        print(f"\n💡 Next Steps:")
        print(f"1. Review optimization results in data/threshold_optimization_results.json")
        print(f"2. Consider updating config.yaml with recommended settings")
        print(f"3. Rebuild datasets with optimal threshold configuration")
        print(f"4. Re-train models for improved performance")

if __name__ == '__main__':
    main()