#!/usr/bin/env python3
"""
Auto-Threshold Pipeline Wrapper

Fully automated threshold optimization and dataset building pipeline.
Eliminates manual steps and ensures optimal thres    def _analyze_threshold_for_db(self, db_path: str, timeframe: str) -> Optional[Dict]:
        \"\"\"Analyze optimal threshold for specific database\"\"\"
        try:\n            # Get horizon from config instead of hardcoding\n            target_horizon = self.config.get('target', {}).get('horizon', 20)\n            \n            # Load data for analysis\n            query = \"SELECT timestamp, close FROM features ORDER BY timestamp DESC LIMIT 2000\"\n            with sqlite3.connect(db_path) as conn:\n                df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])\n            \n            if df.empty or len(df) < 100:\n                return None\n            \n            df = df.sort_values('timestamp')\n            \n            # Calculate returns for configured horizon (default: 20)\n            returns = df['close'].pct_change(target_horizon).shift(-target_horizon) * 100\n            returns_clean = returns.dropna()
Usage:
    python utils/auto_pipeline.py --timeframe 15m --pairs btcusdt --months 3
    
Features:
    - Auto-generates optimal thresholds if not exist
    - Validates threshold performance
    - Builds datasets with optimal configuration
    - Provides comprehensive validation reports
"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
import sqlite3
import subprocess
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

# Add project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

class AutoPipeline:
    """
    Automated Threshold Optimization Pipeline
    
    Fully automated pipeline that:
    1. Auto-detects correct Python environment
    2. Auto-creates missing databases
    3. Performs threshold optimization analysis
    4. Generates auto-balanced configurations
    """
    
    def __init__(self, config_path: str = 'config/config.yaml', 
                 auto_config_path: str = 'data/auto_balanced_thresholds.json'):
        self.config_path = config_path
        self.auto_config_path = auto_config_path
        self.python_executable = self._detect_python_executable()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.pairs = self.config.get('pairs', ['BTCUSDT'])
        self.timeframes = self.config.get('timeframes', ['15m'])
        self.months = 3  # Default to 3 months of data
    
    def _detect_python_executable(self) -> str:
        """
        Auto-detect the correct Python executable for this environment
        
        Returns:
            str: Path to the correct Python executable
        """
        # Try different possible Python paths
        possible_paths = [
            "/home/donvirtus/miniconda3/envs/projects/bin/python",  # Conda projects env
            "/home/donvirtus/miniconda3/bin/python",                # Base conda
            "python3",                                              # System Python3
            "python"                                                # System Python
        ]
        
        for python_path in possible_paths:
            try:
                # Test if this Python has required packages
                result = subprocess.run([
                    python_path, '-c', 
                    'import pandas, numpy, yaml, sqlite3; print("OK")'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and "OK" in result.stdout:
                    print(f"🐍 Detected Python: {python_path}")
                    return python_path
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        # Fallback to system python if nothing else works
        print("⚠️  Could not detect Python with required packages, using 'python'")
        return "python"
        
    def check_auto_config_exists(self) -> bool:
        """Check if auto-balanced config file exists and is valid"""
        if not os.path.exists(self.auto_config_path):
            return False
            
        try:
            with open(self.auto_config_path, 'r') as f:
                config = json.load(f)
            
            # Validate structure
            required_keys = ['individual_results', 'recommendation']
            return all(key in config for key in required_keys)
            
        except Exception:
            return False
    
    def generate_auto_config(self, pairs: List[str], timeframes: List[str]) -> bool:
        """
        Generate auto-balanced thresholds config using threshold analysis
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("🔍 Auto-balanced config not found. Generating optimal thresholds...")
            
            # For each pair-timeframe combination, analyze optimal threshold
            individual_results = {}
            all_thresholds = []
            
            for pair in pairs:
                for timeframe in timeframes:
                    db_path = f"data/db/{pair.lower().replace('usdt', '')}_{timeframe}.sqlite"
                    
                    if os.path.exists(db_path):
                        print(f"📊 Analyzing {pair} {timeframe}...")
                        optimal = self._analyze_threshold_for_db(db_path, timeframe)
                        
                        if optimal:
                            key = f"{pair.replace('USDT', '').upper()} {timeframe.upper()}"
                            individual_results[key] = optimal
                            all_thresholds.append(optimal['optimal_threshold'])
                        else:
                            print(f"⚠️  Analysis failed for {pair} {timeframe}")
                    else:
                        print(f"⚠️  Database not found: {db_path}")
            
            if not individual_results:
                print("❌ No valid analysis results. Using default fallback.")
                return False
            
            # Calculate average optimal threshold
            avg_threshold = sum(all_thresholds) / len(all_thresholds)
            
            # Generate complete config
            auto_config = {
                "individual_results": individual_results,
                "average_optimal_threshold": round(avg_threshold, 3),
                "recommendation": {
                    "threshold": round(avg_threshold, 3),
                    "description": "Auto-generated optimal threshold for balanced distribution"
                },
                "generated_at": datetime.now().isoformat(),
                "balancer_version": "2.0_auto_pipeline",
                "generation_method": "automated_analysis"
            }
            
            # Save config
            os.makedirs(os.path.dirname(self.auto_config_path), exist_ok=True)
            with open(self.auto_config_path, 'w') as f:
                json.dump(auto_config, f, indent=2)
            
            print(f"✅ Auto-config generated: {len(individual_results)} pairs analyzed")
            print(f"📊 Average optimal threshold: {avg_threshold:.3f}%")
            print(f"💾 Saved to: {self.auto_config_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating auto-config: {e}")
            return False
    
    def _analyze_threshold_for_db(self, db_path: str, timeframe: str) -> Optional[Dict]:
        """Analyze optimal threshold for specific database"""
        try:
            # Get horizon from config instead of hardcoding
            target_horizon = self.config.get('target', {}).get('horizon', 20)
            
            # Load data for analysis
            query = "SELECT timestamp, close FROM features ORDER BY timestamp DESC LIMIT 2000"
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
            
            if df.empty or len(df) < 100:
                return None
            
            df = df.sort_values('timestamp')
            
            # Calculate returns for configured horizon (default: 20)
            returns = df['close'].pct_change(target_horizon).shift(-target_horizon) * 100
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 50:
                return None
            
            # Test range of thresholds
            test_thresholds = np.arange(0.05, 1.0, 0.05)  # 0.05% to 0.95% by 0.05%
            best_threshold = 0.5
            best_score = 0
            best_distribution = {}
            
            for threshold in test_thresholds:
                up = (returns_clean > threshold).sum()
                down = (returns_clean < -threshold).sum()
                sideways = len(returns_clean) - up - down
                
                distribution = {
                    'UP': (up / len(returns_clean)) * 100,
                    'SIDEWAYS': (sideways / len(returns_clean)) * 100,
                    'DOWN': (down / len(returns_clean)) * 100
                }
                
                # Get target distribution from config instead of hardcoding
                target_dist = self.config.get('target', {}).get('auto_balance', {}).get('target_distribution', {
                    'UP': 33.0, 'SIDEWAYS': 34.0, 'DOWN': 33.0
                })
                
                # Calculate balance score (closer to target distribution = better)
                balance_score = 100 - (
                    abs(target_dist['UP'] - distribution['UP']) + 
                    abs(target_dist['SIDEWAYS'] - distribution['SIDEWAYS']) + 
                    abs(target_dist['DOWN'] - distribution['DOWN'])
                )
                balance_score = max(0, balance_score)
                
                if balance_score > best_score:
                    best_score = balance_score
                    best_threshold = threshold
                    best_distribution = distribution.copy()
            
            return {
                'optimal_threshold': round(best_threshold, 3),
                'distribution': {k: round(v, 1) for k, v in best_distribution.items()},
                'balance_score': round(best_score, 1),
                'trading_percentage': round(best_distribution['UP'] + best_distribution['DOWN'], 1),
                'total_samples': len(returns_clean),
                'method': 'auto_pipeline_analysis',
                'target_horizon': target_horizon,
                'target_distribution': target_dist
            }
            
        except Exception as e:
            print(f"Error analyzing {db_path}: {e}")
            return None
    
    def build_datasets(self, timeframes: List[str], pairs: List[str], months: int, 
                      force_rebuild: bool = True) -> bool:
        """
        Build datasets with optimal threshold configuration
        
        Returns:
            bool: True if successful
        """
        try:
            print(f"\n🚀 Building datasets with auto-optimized thresholds...")
            print(f"Timeframes: {timeframes}")
            print(f"Pairs: {pairs}")
            print(f"Months: {months}")
            
            for timeframe in timeframes:
                for pair in pairs:
                    print(f"\n📊 Building {pair} {timeframe}...")
                    
                    cmd = [
                        self.python_executable, 'pipeline/build_dataset.py',
                        '--timeframe', timeframe,
                        '--pairs', pair.lower(),
                        '--months', str(months)
                    ]
                    
                    if force_rebuild:
                        cmd.extend(['--force-full-rebuild', '--no-retain-full-history'])
                    
                    # Run build command with timeout
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    
                    if result.returncode == 0:
                        print(f"✅ {pair} {timeframe} completed successfully")
                        
                        # Extract threshold info from output
                        if "Auto-balanced threshold" in result.stderr:
                            for line in result.stderr.split('\n'):
                                if "Auto-balanced threshold" in line:
                                    print(f"   {line.split('INFO - ')[-1]}")
                    else:
                        print(f"❌ {pair} {timeframe} failed:")
                        error_msg = "Unknown error"
                        if result.stderr:
                            # Extract meaningful error from stderr
                            error_lines = result.stderr.strip().split('\n')
                            for line in reversed(error_lines):
                                if 'ERROR' in line or 'Error' in line or 'Exception' in line:
                                    error_msg = line.split('ERROR')[-1].strip() if 'ERROR' in line else line.strip()
                                    break
                        print(f"   Error: {error_msg}")
                        if result.stdout:
                            print(f"   Output: {result.stdout.strip()}")
                        return False
            
            return True
            
        except subprocess.TimeoutExpired:
            print("❌ Dataset building timed out (>10 minutes)")
            return False
        except Exception as e:
            print(f"❌ Dataset building failed: {e}")
            return False
    
    def validate_results(self, timeframes: List[str], pairs: List[str]) -> Dict:
        """
        Comprehensive validation of threshold optimization results
        
        Returns:
            Dict: Validation results summary
        """
        try:
            print(f"\n🔍 VALIDATING OPTIMIZATION RESULTS")
            print("=" * 50)
            
            # Get horizons from config instead of hardcoding
            horizons = self.config.get('multi_horizon', {}).get('horizons', [1, 5, 10, 20])
            threshold_mode = self.config.get('target', {}).get('threshold_mode', 'auto_balanced')
            
            print(f"   Threshold Mode: {threshold_mode}")
            print(f"   Horizons: {horizons}")
            
            validation_summary = {
                'pairs_analyzed': [],
                'threshold_applied': {},
                'distribution_quality': {},
                'threshold_mode': threshold_mode,
                'horizons_analyzed': horizons,
                'overall_score': 0,
                'recommendations': []
            }
            
            total_scores = []
            
            for pair in pairs:
                for timeframe in timeframes:
                    pair_key = f"{pair}_{timeframe}"
                    db_path = f"data/db/{pair.lower().replace('usdt', '')}_{timeframe}.sqlite"
                    
                    if not os.path.exists(db_path):
                        print(f"⚠️  Database not found: {pair} {timeframe}")
                        continue
                    
                    print(f"\n📊 Validating {pair} {timeframe}...")
                    
                    # Build dynamic SQL query based on config horizons
                    direction_columns = [f"direction_h{h}" for h in horizons]
                    sql_columns = ", ".join(direction_columns)
                    query = f"SELECT {sql_columns} FROM features"
                    
                    # Load dataset and analyze distribution
                    with sqlite3.connect(db_path) as conn:
                        df = pd.read_sql(query, conn)
                    
                    if df.empty:
                        continue
                    
                    # Analyze each horizon dynamically
                    horizon_scores = []
                    for horizon in horizons:
                        col = f'direction_{horizon}'
                        if col in df.columns:
                            dist = df[col].value_counts(normalize=True).sort_index()
                            
                            # Calculate distribution percentages
                            down_pct = dist.get(0, 0) * 100
                            sideways_pct = dist.get(1, 0) * 100
                            up_pct = dist.get(2, 0) * 100
                            
                            # Balance score
                            balance_score = 100 - (
                                abs(33.3 - up_pct) + 
                                abs(33.4 - sideways_pct) + 
                                abs(33.3 - down_pct)
                            )
                            balance_score = max(0, balance_score)
                            horizon_scores.append(balance_score)
                            
                            print(f"   {horizon}: DOWN={down_pct:.1f}%, SIDE={sideways_pct:.1f}%, UP={up_pct:.1f}% (Score: {balance_score:.1f})")
                    
                    # Average score for this pair-timeframe
                    avg_score = sum(horizon_scores) / len(horizon_scores) if horizon_scores else 0
                    total_scores.append(avg_score)
                    
                    validation_summary['pairs_analyzed'].append(pair_key)
                    validation_summary['distribution_quality'][pair_key] = avg_score
                    
                    # Quality assessment
                    if avg_score > 85:
                        quality = "EXCELLENT ✅"
                    elif avg_score > 70:
                        quality = "GOOD ✅"
                    elif avg_score > 50:
                        quality = "FAIR ⚠️"
                    else:
                        quality = "POOR ❌"
                    
                    print(f"   Overall Quality: {quality} (Score: {avg_score:.1f}/100)")
            
                # Overall assessment
                if total_scores:
                    overall_score = sum(total_scores) / len(total_scores)
                    validation_summary['overall_score'] = overall_score
                    
                    print(f"\n🏆 VALIDATION SUMMARY:")
                    print("=" * 30)
                    print(f"Threshold Mode: {threshold_mode}")
                    print(f"Horizons: {horizons}")
                    print(f"Pairs Analyzed: {len(validation_summary['pairs_analyzed'])}")
                    print(f"Overall Score: {overall_score:.1f}/100")
                    
                    if overall_score > 85:
                        print("✅ EXCELLENT - Auto-optimization highly successful!")
                        validation_summary['recommendations'].append("System is optimally configured")
                    elif overall_score > 70:
                        print("✅ GOOD - Auto-optimization successful")
                        validation_summary['recommendations'].append("Consider minor threshold adjustments for specific pairs")
                    else:
                        print("⚠️  NEEDS IMPROVEMENT - Consider threshold re-optimization")
                        validation_summary['recommendations'].append("Re-run auto-optimization with more data")
                
            return validation_summary
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return {'error': str(e)}
    
    def run_full_pipeline(self, timeframes: List[str], pairs: List[str], months: int = 3) -> bool:
        """
        Execute complete automated pipeline with intelligent database handling
        
        Returns:
            bool: True if pipeline completed successfully
        """
        try:
            print("🤖 AUTOMATED THRESHOLD OPTIMIZATION PIPELINE")
            print("=" * 60)
            print(f"Target: {pairs} | Timeframes: {timeframes} | Data: {months} months")
            print()
            
            # Step 0: Smart database check and creation
            print("🔍 Step 0: Checking database requirements...")
            if not self._ensure_databases_exist():
                print("❌ Failed to ensure databases exist. Pipeline aborted.")
                return False
            
            # Step 1: Check/Generate auto-config
            if not self.check_auto_config_exists():
                print("🔧 Step 1: Generating optimal threshold configuration...")
                if not self.generate_auto_config(pairs, timeframes):
                    print("❌ Failed to generate auto-config. Pipeline aborted.")
                    return False
            else:
                print("✅ Step 1: Auto-config exists and valid")
                
                # Show current config
                with open(self.auto_config_path, 'r') as f:
                    config = json.load(f)
                avg_threshold = config.get('recommendation', {}).get('threshold', 'unknown')
                print(f"   Current optimal threshold: {avg_threshold}%")
            
            # Step 2: Build datasets
            print("\n🚀 Step 2: Building datasets with optimal thresholds...")
            if not self.build_datasets(timeframes, pairs, months):
                print("❌ Dataset building failed. Pipeline aborted.")
                return False
            
            # Step 3: Validate results
            print("\n🔍 Step 3: Validating optimization results...")
            validation = self.validate_results(timeframes, pairs)
            
            if 'error' not in validation:
                self.validation_results = validation
                
                # Save validation report
                report_path = f"data/threshold_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_path, 'w') as f:
                    json.dump(validation, f, indent=2)
                print(f"📄 Validation report saved: {report_path}")
                
                print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
                return True
            else:
                print("❌ Validation failed. Check results manually.")
                return False
                
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            return False
    
    def _ensure_databases_exist(self) -> bool:
        """
        Check if required databases exist and create them if missing
        
        Returns:
            bool: True if all databases exist or were created successfully
        """
        try:
            print("🔍 Step 0: Checking database requirements...")
            
            missing_dbs = []
            
            for pair in self.pairs:
                for timeframe in self.timeframes:
                    db_filename = f"{pair.lower().replace('usdt', '')}_{timeframe}.sqlite"
                    db_path = os.path.join("data", "db", db_filename)
                    
                    if not os.path.exists(db_path):
                        missing_dbs.append((pair, timeframe, db_path))
            
            if not missing_dbs:
                print("   ✅ All required databases exist")
                return True
            
            print(f"   🔧 Found {len(missing_dbs)} missing databases")
            for pair, timeframe, db_path in missing_dbs:
                print(f"      Missing: {db_path}")
            
            print("   🚀 Auto-creating missing databases...")
            
            # Create missing databases using the detected Python executable
            for pair, timeframe, db_path in missing_dbs:
                print(f"\n   📊 Building {pair} {timeframe}...")
                
                cmd = [
                    self.python_executable, 'pipeline/build_dataset.py',
                    '--config', 'config/config.yaml',
                    '--pair', pair,
                    '--timeframe', timeframe,
                    '--output-dir', 'data/db',
                    '--months', str(self.months)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"      ✅ Successfully created {pair} {timeframe}")
                    # Verify the database was actually created
                    if os.path.exists(db_path):
                        print(f"      📊 Database verified: {db_path}")
                    else:
                        print(f"      ⚠️  Database not found after build: {db_path}")
                        return False
                else:
                    print(f"      ❌ Failed to create {pair} {timeframe}")
                    error_msg = "Build failed"
                    if result.stderr:
                        # Extract meaningful error from stderr
                        error_lines = result.stderr.strip().split('\n')
                        for line in reversed(error_lines):
                            if 'ERROR' in line or 'Error' in line or 'Exception' in line:
                                error_msg = line.split('ERROR')[-1].strip() if 'ERROR' in line else line.strip()
                                break
                    print(f"         Error: {error_msg}")
                    if result.stdout:
                        print(f"         Output: {result.stdout.strip()}")
                    return False
            
            return True
            
        except subprocess.TimeoutExpired:
            print("❌ Database creation timed out (>5 minutes)")
            return False
        except Exception as e:
            print(f"❌ Database check/creation failed: {e}")
            return False


def main():
    """CLI interface for auto-threshold pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Threshold Optimization Pipeline')
    parser.add_argument('--timeframes', nargs='+', default=['15m'], 
                       help='Timeframes to process (default: 15m)')
    parser.add_argument('--pairs', nargs='+', default=['btcusdt'], 
                       help='Trading pairs to process (default: btcusdt)')
    parser.add_argument('--months', type=int, default=3, 
                       help='Months of data to collect (default: 3)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation on existing datasets')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regeneration of auto-config even if exists')
    
    args = parser.parse_args()
    
    # Convert pairs to uppercase
    pairs = [pair.upper() + ('USDT' if not pair.upper().endswith('USDT') else '') for pair in args.pairs]
    
    pipeline = AutoPipeline()
    
    if args.validate_only:
        # Validation only mode
        print("🔍 VALIDATION-ONLY MODE")
        validation = pipeline.validate_results(args.timeframes, pairs)
        return
    
    if args.force_regenerate and os.path.exists(pipeline.auto_config_path):
        print("🔧 Force regenerating auto-config...")
        os.remove(pipeline.auto_config_path)
    
    # Run full pipeline
    success = pipeline.run_full_pipeline(args.timeframes, pairs, args.months)
    
    if success:
        print("\n✅ AUTO-THRESHOLD PIPELINE COMPLETED SUCCESSFULLY!")
        print("🎯 Your datasets are now optimally balanced for ML training!")
    else:
        print("\n❌ Pipeline encountered errors. Check output above.")


if __name__ == '__main__':
    main()