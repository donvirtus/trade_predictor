#!/usr/bin/env python3
"""
Test script untuk menguji auto-calculation adaptive threshold system.

Usage:
  python test_adaptive_auto.py [--timeframe TF] [--force] [--debug]

Examples:
  python test_adaptive_auto.py --timeframe 15m
  python test_adaptive_auto.py --force --debug
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.config import load_config
from utils.adaptive_thresholds import AdaptiveThresholdCalculator, get_adaptive_multipliers


def setup_logging(debug=False):
    """Setup logging untuk debugging"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Reduce noise from external libraries
    for noisy_logger in ['urllib3', 'requests', 'ccxt']:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def test_volatility_calculation(calc, timeframes, debug=False):
    """Test basic volatility calculation"""
    print("\n" + "="*50)
    print("🧪 TEST: Volatility Calculation")
    print("="*50)
    
    for tf in timeframes:
        print(f"\n📊 Testing timeframe: {tf}")
        
        # Load data
        df = calc.load_timeframe_data(tf, lookback_days=30)
        if df is None:
            print(f"   ❌ No data available for {tf}")
            continue
        
        print(f"   ✅ Loaded {len(df)} rows")
        
        # Calculate volatility metrics
        vol_metrics = calc.calculate_volatility_metrics(df, method="rolling_std", window=20)
        
        if vol_metrics:
            print(f"   📈 Volatility Metrics:")
            for key, value in vol_metrics.items():
                print(f"      {key}: {value:.6f}")
        else:
            print(f"   ❌ Failed to calculate volatility")


def test_multiplier_calculation(calc, timeframes, debug=False):
    """Test optimal multiplier calculation"""
    print("\n" + "="*50)
    print("🧪 TEST: Optimal Multiplier Calculation")
    print("="*50)
    
    multipliers = calc.calculate_optimal_multipliers(
        timeframes=timeframes,
        base_timeframe="1h",
        lookback_days=30,
        method="rolling_std"
    )
    
    if multipliers:
        print("\n📊 Calculated Multipliers:")
        for tf, mult in sorted(multipliers.items()):
            print(f"   {tf:6s}: {mult:.3f}x")
        
        # Compare dengan manual values dari config
        manual_multipliers = {'5m': 0.3, '15m': 0.6, '1h': 1.0, '2h': 1.4, '4h': 2.0, '6h': 2.5}
        
        print(f"\n📋 Comparison with Manual Values:")
        print(f"{'Timeframe':<10} {'Auto':<8} {'Manual':<8} {'Diff':<8} {'Change'}")
        print("-" * 50)
        
        for tf in timeframes:
            if tf in multipliers and tf in manual_multipliers:
                auto_val = multipliers[tf]
                manual_val = manual_multipliers[tf]
                diff = auto_val - manual_val
                change = f"{(diff/manual_val)*100:+.1f}%" if manual_val != 0 else "N/A"
                print(f"{tf:<10} {auto_val:<8.3f} {manual_val:<8.3f} {diff:+.3f}   {change}")
        
        return multipliers
    else:
        print("❌ Failed to calculate multipliers")
        return {}


def test_config_integration(config_path, timeframes, force_recalc=False, debug=False):
    """Test integration dengan config system"""
    print("\n" + "="*50)
    print("🧪 TEST: Config Integration")
    print("="*50)
    
    try:
        # Load config
        cfg = load_config(config_path)
        print(f"✅ Config loaded from: {config_path}")
        
        # Test auto-calculation via config
        print(f"\n📊 Testing get_adaptive_multipliers()...")
        multipliers = get_adaptive_multipliers(cfg, timeframes, force_recalculate=force_recalc)
        
        if multipliers:
            print(f"✅ Successfully retrieved multipliers:")
            for tf, mult in sorted(multipliers.items()):
                print(f"   {tf:6s}: {mult:.3f}x")
            
            # Show method used
            adaptive_cfg = cfg.target.get('adaptive_thresholds', {})
            method = adaptive_cfg.get('method', 'manual')
            print(f"\n📋 Method used: {method}")
            
            if method == 'auto_volatility':
                vol_cfg = adaptive_cfg.get('volatility_calc', {})
                print(f"   - Volatility method: {vol_cfg.get('method', 'rolling_std')}")
                print(f"   - Window: {vol_cfg.get('window', 20)}")
                print(f"   - Smoothing factor: {vol_cfg.get('smoothing_factor', 0.7)}")
                print(f"   - Lookback days: {adaptive_cfg.get('lookback_days', 30)}")
            
            return True
        else:
            print("❌ Failed to get multipliers via config integration")
            return False
            
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False


def test_cache_system(calc, timeframes, debug=False):
    """Test caching system"""
    print("\n" + "="*50)
    print("🧪 TEST: Cache System")
    print("="*50)
    
    # Clear cache first
    if os.path.exists(calc.cache_file):
        os.remove(calc.cache_file)
        print("🗑️  Cleared existing cache")
    
    # Calculate and save
    multipliers = calc.calculate_optimal_multipliers(timeframes, lookback_days=15)
    if multipliers:
        metadata = {
            'test_run': True,
            'timeframes': timeframes,
            'test_timestamp': datetime.now().isoformat()
        }
        calc.save_multipliers_cache(multipliers, metadata)
        print("✅ Multipliers saved to cache")
        
        # Load from cache
        cached = calc.load_multipliers_cache(max_age_hours=24)
        if cached:
            print("✅ Multipliers loaded from cache")
            print("📋 Cache validation:", "✅ PASS" if cached == multipliers else "❌ FAIL")
            return True
        else:
            print("❌ Failed to load from cache")
            return False
    else:
        print("❌ Failed to calculate multipliers for caching")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test adaptive threshold auto-calculation system")
    parser.add_argument('--timeframe', type=str, default='15m', 
                       help='Primary timeframe to test (default: 15m)')
    parser.add_argument('--force', action='store_true', 
                       help='Force recalculation, ignore cache')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Config file path (default: config/config.yaml)')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.debug)
    
    print("🚀 Starting Adaptive Threshold Auto-Calculation Tests")
    print(f"   Primary timeframe: {args.timeframe}")
    print(f"   Force recalculation: {args.force}")
    print(f"   Debug mode: {args.debug}")
    
    # Initialize
    calc = AdaptiveThresholdCalculator()
    timeframes = ['5m', '15m', '1h', '2h', '4h', '6h']
    
    # Run tests
    success_count = 0
    total_tests = 4
    
    try:
        # Test 1: Basic volatility calculation
        test_volatility_calculation(calc, timeframes[:3], args.debug)  # Test subset first
        success_count += 1
        
        # Test 2: Multiplier calculation
        multipliers = test_multiplier_calculation(calc, timeframes, args.debug)
        if multipliers:
            success_count += 1
        
        # Test 3: Config integration
        if test_config_integration(args.config, timeframes, args.force, args.debug):
            success_count += 1
        
        # Test 4: Cache system
        if test_cache_system(calc, timeframes[:4], args.debug):  # Test subset for speed
            success_count += 1
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests PASSED! Auto-calculation system is working correctly.")
        sys.exit(0)
    else:
        print(f"⚠️  Some tests failed. Check logs above for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()