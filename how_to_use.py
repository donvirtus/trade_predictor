#!/usr/bin/env python3
"""
🗺️ TRADE PREDICTOR PROJECT MAP - Interactive Explorer

Script untuk memahami struktur, alur, dan hubungan antar file dalam project trade_predictor.
Memberikan overview lengkap dengan cara yang mudah dipahami.

Usage:
  python how_to_use.py [--detail] [--flow] [--dependencies]

Created: 2025-09-29
Author: AI Assistant for User Understanding
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add project root to path
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class ProjectExplorer:
    """Interactive explorer untuk memahami struktur project trade_predictor"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.structure = self._analyze_structure()
        
    def _analyze_structure(self) -> Dict:
        """Analyze project structure dan buat mapping"""
        
        structure = {
            "🎯 CORE PIPELINE": {
                "pipeline/build_dataset.py": {
                    "purpose": "🏗️ Main data pipeline - build datasets dengan technical indicators (MULTI-PAIR SUPPORT 🆕)",
                    "key_functions": [
                        "build_separate_timeframes() - Build per-timeframe databases", 
                        "fetch_and_process_timeframe() - Process single pair-timeframe",
                        "merge_external() - Merge external data sources"
                    ],
                    "calls": [
                        "data/binance_fetch.py → fetch_ohlcv_history()",
                        "features/indicators.py → add_price_indicators()", 
                        "features/targets.py → label_multi_horizon_directions(pair=pair)",
                        "utils/adaptive_thresholds.py → get_adaptive_multipliers(config, pair) 🚀",
                        "utils/path_utils.py → generate_db_path(pair, timeframe) 🆕"
                    ],
                    "output": "data/db/{pair}_{timeframe}.sqlite files (dynamic pairs: btc_15m.sqlite, eth_1h.sqlite, etc.) 🆕 SUPPORTS --pairs FLAG",
                    "complexity": "⭐⭐⭐⭐⭐ (MASTER ORCHESTRATOR)"
                }
            },
            
            "📊 DATA SOURCES": {
                "data/binance_fetch.py": {
                    "purpose": "🌐 Fetch OHLCV data dari Binance API",
                    "key_functions": [
                        "init_exchange() - Initialize Binance connection",
                        "fetch_ohlcv_history() - Fetch historical data with gaps handling",
                        "fetch_ohlcv() - Simple OHLCV fetch"
                    ],
                    "calls": ["ccxt.binance()"],
                    "output": "Raw OHLCV DataFrame",
                    "complexity": "⭐⭐⭐ (EXTERNAL API)"
                },
                
                "data/continuity.py": {
                    "purpose": "🔍 Data quality control - gap detection & backfill",
                    "key_functions": [
                        "detect_gaps() - Find missing data periods",
                        "backfill_gaps() - Fill missing data",
                        "validate_continuity() - Check data quality"
                    ],
                    "calls": ["data/binance_fetch.py"],
                    "output": "Clean continuous DataFrame",
                    "complexity": "⭐⭐⭐ (DATA QUALITY)"
                },
                
                "data/external/": {
                    "purpose": "🌍 External data sources (CoinGecko, CoinMetrics, Dune)",
                    "key_functions": [
                        "coingecko.py → fetch_coingecko_snapshot()",
                        "coinmetrics.py → fetch_coinmetrics()", 
                        "dune.py → fetch_dune_query()"
                    ],
                    "calls": ["External APIs"],
                    "output": "External features DataFrame",
                    "complexity": "⭐⭐ (OPTIONAL FEATURES)"
                }
            },
            
            "⚙️ FEATURE ENGINEERING": {
                "features/indicators.py": {
                    "purpose": "📈 Technical indicators (SMA, EMA, RSI, Bollinger Bands, etc.)",
                    "key_functions": [
                        "add_price_indicators() - Add all technical indicators",
                        "Individual functions for each indicator"
                    ],
                    "calls": ["pandas, numpy calculations"],
                    "output": "DataFrame + technical indicators",
                    "complexity": "⭐⭐⭐ (TECHNICAL ANALYSIS)"
                },
                
                "features/derivatives.py": {
                    "purpose": "🔢 Derivative features (ROC, momentum, volatility, etc.)",
                    "key_functions": [
                        "add_derivative_features() - Add derived calculations",
                        "ROC, momentum, volatility calculations"
                    ],
                    "calls": ["pandas calculations"],
                    "output": "DataFrame + derivative features", 
                    "complexity": "⭐⭐ (MATH CALCULATIONS)"
                },
                
                "features/targets.py": {
                    "purpose": "🎯 Target labeling - UP/DOWN/SIDEWAYS classification",
                    "key_functions": [
                        "label_multi_horizon_directions() - Multi-horizon labels 🚀",
                        "label_future_direction() - Basic direction labels",
                        "label_extremes_horizon() - Max/min future prices"
                    ],
                    "calls": ["utils/adaptive_thresholds.py → get_adaptive_multipliers(config, pair) 🚀"],
                    "output": "DataFrame + direction labels (0=DOWN, 1=SIDEWAYS, 2=UP)",
                    "complexity": "⭐⭐⭐⭐ (TARGET CREATION)"
                }
            },
            
            "🧠 MACHINE LEARNING": {
                "scripts/train_tree_model.py": {
                    "purpose": "🌳 Train tree-based models (LightGBM, XGBoost, CatBoost) 🆕 SUPPORTS --pairs FLAG",
                    "key_functions": [
                        "main() - Main training loop with multi-asset support",
                        "UniversalModelTrainer() - Per-pair model training",
                        "Model-specific training functions"
                    ],
                    "calls": ["models/ → trained model files per asset"],
                    "output": "Trained models in models/ directory (per asset: btc_15m_lightgbm_*.txt, eth_15m_lightgbm_*.txt)",
                    "complexity": "⭐⭐⭐⭐ (ML TRAINING)"
                },
                
                "scripts/train_timeseries_proper.py": {
                    "purpose": "📊 Train time series models dengan proper validation",
                    "key_functions": [
                        "Enhanced training dengan XGBoost/CatBoost support",
                        "Time series validation"
                    ],
                    "calls": ["Build datasets → Train models"],
                    "output": "Time series models",
                    "complexity": "⭐⭐⭐⭐ (TIME SERIES ML)"
                },
                
                "scripts/predict_enhanced.py": {
                    "purpose": "🔮 Make predictions dengan trained models",
                    "key_functions": [
                        "Load models dan make predictions",
                        "Output JSON atau rich format"
                    ],
                    "calls": ["models/ → predictions"],
                    "output": "Prediction results",
                    "complexity": "⭐⭐⭐ (INFERENCE)"
                }
            },
            
            "🔧 UTILITIES": {
                "utils/config.py": {
                    "purpose": "⚙️ Configuration management - load YAML configs",
                    "key_functions": ["load_config() - Load and parse config.yaml"],
                    "calls": ["config/config.yaml"],
                    "output": "Configuration object",
                    "complexity": "⭐ (SIMPLE LOADER)"
                },
                
                "utils/logging.py": {
                    "purpose": "📝 Logging system - structured logging",
                    "key_functions": ["get_logger() - Get configured logger"],
                    "calls": ["Python logging"],
                    "output": "Logger instances",
                    "complexity": "⭐ (LOGGING)"
                },
                
                "utils/adaptive_thresholds.py": {
                    "purpose": "🚀 AUTO-CALCULATION ENGINE - Dynamic threshold calculation",
                    "key_functions": [
                        "get_adaptive_multipliers() - Main entry point 🎯",
                        "AdaptiveThresholdCalculator.calculate_optimal_multipliers()",
                        "Volatility analysis + hybrid approach"
                    ],
                    "calls": ["data/db/*.sqlite → volatility analysis"],
                    "output": "Dynamic multipliers dict {'5m': 0.45, '1h': 1.0, ...}",
                    "complexity": "⭐⭐⭐⭐⭐ (ADVANCED AUTO-CALCULATION)"
                },
                
                "utils/threshold_balancer.py": {
                    "purpose": "🎯 SCIENTIFIC THRESHOLD OPTIMIZER - Auto-balanced class distribution",
                    "key_functions": [
                        "ThresholdAutoBalancer.auto_balance_timeframe() - Binary search optimization",
                        "find_optimal_threshold() - Target 33%/33%/33% distribution",
                        "calculate_balance_score() - Performance evaluation"
                    ],
                    "calls": ["data/db/*.sqlite → empirical analysis"],
                    "output": "Optimal thresholds: data/auto_balanced_thresholds.json (BTC:0.406%, ETH:0.713%, DOGE:1.325%)",
                    "complexity": "⭐⭐⭐⭐⭐ (SCIENTIFIC OPTIMIZATION)"
                },
                
                "utils/path_utils.py": {
                    "purpose": "🔧 DYNAMIC PATH UTILITIES - Multi-pair path generation 🆕",
                    "key_functions": [
                        "generate_db_path() - Dynamic database paths per pair",
                        "generate_model_filename() - Consistent model naming",
                        "normalize_pair_name() - BTCUSDT → btc conversion"
                    ],
                    "calls": ["Used by all scripts for consistent naming"],
                    "output": "Dynamic paths: btc_15m.sqlite, eth_1h.sqlite, etc.",
                    "complexity": "⭐⭐ (PATH UTILITIES)"
                },
                
                "utils/path_utils.py": {
                    "purpose": "🔧 DYNAMIC PATH UTILITIES - Multi-pair path generation 🆕",
                    "key_functions": [
                        "generate_db_path() - Dynamic database paths per pair",
                        "generate_model_filename() - Consistent model naming",
                        "normalize_pair_name() - BTCUSDT → btc conversion"
                    ],
                    "calls": ["Used by all scripts for consistent naming"],
                    "output": "Dynamic paths: btc_15m.sqlite, eth_1h.sqlite, etc.",
                    "complexity": "⭐⭐ (PATH UTILITIES)"
                }
            },
            
            "📋 CONFIGURATION": {
                "config/config.yaml": {
                    "purpose": "⚙️ Master configuration file",
                    "key_sections": [
                        "data: Binance API settings",
                        "pairs: Trading pairs to process", 
                        "timeframes: Time intervals",
                        "target: Threshold settings + adaptive_thresholds 🚀",
                        "indicators: Technical indicator settings",
                        "model configs: ML model parameters"
                    ],
                    "calls": ["Used by ALL scripts"],
                    "output": "Configuration for entire project",
                    "complexity": "⭐⭐ (CONFIGURATION)"
                },
                
                "THRESHOLD_MODES_GUIDE.md": {
                    "purpose": "📚 COMPLETE THRESHOLD GUIDE - Manual, Auto-Balanced, Adaptive modes",
                    "key_sections": [
                        "Manual: Fixed threshold (37-38/100 score)",
                        "Auto-Balanced: Scientific optimization (89-96/100 score) 🏆",
                        "Adaptive: Volatility-based dynamic (65-82/100 score)",
                        "Configuration examples + troubleshooting"
                    ],
                    "calls": ["Documentation reference"],
                    "output": "Complete understanding of threshold optimization",
                    "complexity": "⭐⭐⭐ (COMPREHENSIVE GUIDE)"
                }
            },
            
            "💾 DATA STORAGE": {
                "data/db/": {
                    "purpose": "🗄️ SQLite databases per timeframe",
                    "files": [
                        "Dynamic per pair: btc_5m.sqlite, eth_5m.sqlite, ada_5m.sqlite",
                        "Examples: btc_15m.sqlite, eth_15m.sqlite - 15-minute data", 
                        "Examples: btc_1h.sqlite, eth_1h.sqlite - 1-hour data",
                        "Supports all pairs in config.yaml timeframes"
                    ],
                    "tables": [
                        "features - Main OHLCV + indicators + targets",
                        "metadata - Processing metadata"
                    ],
                    "complexity": "⭐⭐ (DATA STORAGE)"
                },
                
                "models/": {
                    "purpose": "🧠 Trained model storage",
                    "files": [
                        "tree_models/ - LightGBM, XGBoost, CatBoost models",
                        "timeseries_models/ - Time series models",
                        "metadata/ - Model metadata JSON files"
                    ],
                    "complexity": "⭐⭐ (MODEL STORAGE)"
                }
            }
        }
        
        return structure

    def show_overview(self):
        """Show high-level project overview"""
        print("🗺️" + "="*80)
        print("📊 TRADE PREDICTOR PROJECT - COMPLETE OVERVIEW")
        print("="*80)
        print()
        
        for category, items in self.structure.items():
            print(f"\n{category}")
            print("-" * 60)
            
            for file_path, details in items.items():
                print(f"\n📁 {file_path}")
                print(f"   Purpose: {details['purpose']}")
                print(f"   Complexity: {details['complexity']}")
                
                if 'key_functions' in details:
                    print(f"   Key Functions:")
                    for func in details['key_functions'][:2]:  # Show first 2
                        print(f"     • {func}")
                
                if 'output' in details:
                    print(f"   Output: {details['output']}")
        
        print(f"\n{'='*80}")
        print("💡 TIP: Use --detail untuk detail lengkap, --flow untuk execution flow")

    def show_execution_flow(self):
        """Show step-by-step execution flow"""
        print("🔄" + "="*80)
        print("📋 EXECUTION FLOW - How Everything Works Together")
        print("="*80)
        
        flow_steps = [
            {
                "step": 1,
                "title": "🚀 START: User Command",
                "action": "python pipeline/build_dataset.py --timeframe 15m",
                "description": "User menjalankan data pipeline"
            },
            {
                "step": 2, 
                "title": "📝 Configuration Loading",
                "action": "utils/config.py → load_config('config/config.yaml')",
                "description": "Load master configuration dari YAML"
            },
            {
                "step": 3,
                "title": "💱 Exchange Initialization", 
                "action": "data/binance_fetch.py → init_exchange('future')",
                "description": "Initialize Binance API connection"
            },
            {
                "step": 4,
                "title": "🌐 Data Fetching",
                "action": "data/binance_fetch.py → fetch_ohlcv_history()",
                "description": "Fetch historical OHLCV data dari Binance"
            },
            {
                "step": 5,
                "title": "🔍 Data Quality Control",
                "action": "data/continuity.py → detect_gaps() + backfill_gaps()",
                "description": "Check dan fix missing data periods"
            },
            {
                "step": 6,
                "title": "📈 Technical Indicators",
                "action": "features/indicators.py → add_price_indicators()",
                "description": "Calculate SMA, EMA, RSI, Bollinger Bands, etc."
            },
            {
                "step": 7,
                "title": "🔢 Derivative Features",
                "action": "features/derivatives.py → add_derivative_features()",
                "description": "Calculate ROC, momentum, volatility, etc."
            },
            {
                "step": 8,
                "title": "🌍 External Data",
                "action": "data/external/ → merge_external()",
                "description": "Merge CoinGecko, CoinMetrics data (optional)"
            },
            {
                "step": 9,
                "title": "🎯 THRESHOLD OPTIMIZATION (OPTIONAL)",
                "action": "utils/threshold_balancer.py → auto_balance_timeframe()",
                "description": "Scientific calculation of optimal thresholds for perfect class balance (89-96/100 score)",
                "highlight": True
            },
            {
                "step": 10,
                "title": "🚀 AUTO-CALCULATION ENGINE",
                "action": "utils/adaptive_thresholds.py → get_adaptive_multipliers(config, pair, timeframes)",
                "description": "Apply threshold mode: manual/auto_balanced/adaptive based on config",
                "highlight": True
            },
            {
                "step": 11,
                "title": "🎯 Target Labeling",
                "action": "features/targets.py → label_multi_horizon_directions(pair=pair)",
                "description": "Create UP/DOWN/SIDEWAYS labels menggunakan optimal thresholds per-pair",
                "highlight": True
            },
            {
                "step": 12,
                "title": "🧹 Data Cleaning",
                "action": "Smart NaN handling + deduplication",
                "description": "Clean data preserving valid information"
            },
            {
                "step": 13,
                "title": "💾 Database Storage",
                "action": "SQLite → data/db/{pair}_{timeframe}.sqlite (dynamic per pair)",
                "description": "Save processed dataset to timeframe-specific database"
            },
            {
                "step": 14,
                "title": "🧠 Model Training (Optional)",
                "action": "scripts/train_tree_model.py",
                "description": "Train LightGBM/XGBoost/CatBoost models"
            },
            {
                "step": 15,
                "title": "🔮 Prediction (Optional)",
                "action": "scripts/predict_enhanced.py",
                "description": "Make predictions dengan trained models"
            }
        ]
        
        for step_info in flow_steps:
            step_num = step_info["step"]
            title = step_info["title"]
            action = step_info["action"]
            desc = step_info["description"]
            highlight = step_info.get("highlight", False)
            
            if highlight:
                print(f"\n🌟 STEP {step_num}: {title}")
                print("🔥" + "-" * 70)
            else:
                print(f"\nSTEP {step_num}: {title}")
                print("-" * 60)
            
            print(f"Action: {action}")
            print(f"Description: {desc}")
        
        print(f"\n{'='*80}")
        print("💡 KEY: 🌟 = Auto-calculation engine steps (our enhancement!)")

    def show_dependencies(self):
        """Show dependency relationships"""
        print("🔗" + "="*80)  
        print("📊 DEPENDENCY MAP - Who Calls Who")
        print("="*80)
        
        dependencies = {
            "pipeline/build_dataset.py": {
                "direct_imports": [
                    "utils/config.py → load_config()",
                    "utils/logging.py → get_logger()",
                    "data/binance_fetch.py → fetch_ohlcv_history()",
                    "data/continuity.py → detect_gaps(), backfill_gaps()",
                    "features/indicators.py → add_price_indicators()",
                    "features/derivatives.py → add_derivative_features()",
                    "features/targets.py → label_multi_horizon_directions()"
                ],
                "runtime_imports": [
                    "🚀 utils/adaptive_thresholds.py → get_adaptive_multipliers() (DYNAMIC!)"
                ],
                "data_dependencies": [
                    "config/config.yaml → Configuration",
                    "data/db/*.sqlite → Output databases"
                ]
            },
            
            "features/targets.py": {
                "direct_imports": [
                    "pandas, numpy → Data processing"
                ],
                "runtime_imports": [
                    "🚀 utils/adaptive_thresholds.py → get_adaptive_multipliers() (DYNAMIC!)"
                ],
                "data_dependencies": [
                    "OHLCV DataFrame → Input data",
                    "config.yaml → Adaptive threshold settings"
                ]
            },
            
            "utils/adaptive_thresholds.py": {
                "direct_imports": [
                    "pandas, numpy → Data processing",
                    "sqlite3 → Database access",
                    "datetime → Time calculations"
                ],
                "runtime_imports": [
                    "No runtime imports - self-contained"
                ],
                "data_dependencies": [
                    "data/db/{pair}_{timeframe}.sqlite → Historical volatility data (dynamic pairs)",
                    "data/adaptive_multipliers_cache_{pair}.json → Per-pair cache storage",
                    "config.yaml → Calculation parameters"
                ]
            }
        }
        
        for script, deps in dependencies.items():
            print(f"\n📁 {script}")
            print("=" * 60)
            
            print(f"\n🔗 Direct Imports:")
            for imp in deps["direct_imports"]:
                print(f"   • {imp}")
            
            print(f"\n⚡ Runtime Imports:")
            for imp in deps["runtime_imports"]:
                print(f"   • {imp}")
            
            print(f"\n💾 Data Dependencies:")
            for dep in deps["data_dependencies"]:
                print(f"   • {dep}")
        
        print(f"\n{'='*80}")
        print("💡 RUNTIME IMPORTS = Imported only when needed (conditional)")

    def show_detailed_breakdown(self):
        """Show detailed breakdown of each component"""
        print("🔍" + "="*80)
        print("📊 DETAILED COMPONENT BREAKDOWN")
        print("="*80)
        
        for category, items in self.structure.items():
            print(f"\n{category}")
            print("="*60)
            
            for file_path, details in items.items():
                print(f"\n📁 {file_path}")
                print("-" * 50)
                print(f"Purpose: {details['purpose']}")
                print(f"Complexity: {details['complexity']}")
                
                if 'key_functions' in details:
                    print(f"\nKey Functions:")
                    for func in details['key_functions']:
                        print(f"  • {func}")
                
                if 'calls' in details:
                    print(f"\nCalls/Dependencies:")
                    for call in details['calls']:
                        print(f"  → {call}")
                
                if 'output' in details:
                    print(f"\nOutput: {details['output']}")
                
                if file_path == "utils/adaptive_thresholds.py":
                    print(f"\n🚀 AUTO-CALCULATION DETAILS:")
                    print(f"  • Hybrid approach: 80% manual + 20% volatility-based")
                    print(f"  • Weekly auto-updates dengan 30-day analysis")
                    print(f"  • Conservative bounds: max 50% deviation")
                    print(f"  • Cache system untuk performance")
                    print(f"  • Method: rolling standard deviation")
                
                print()

    def interactive_help(self):
        """Interactive help menu"""
        print("📚" + "="*80)
        print("🎯 INTERACTIVE HELP - How to Understand This Project")
        print("="*80)
        
        help_sections = [
            {
                "title": "🚀 QUICK START - Baru Pertama Kali?",
                "steps": [
                    "1. Baca OVERVIEW dulu: python how_to_use.py",
                    "2. Pahami FLOW: python how_to_use.py --flow", 
                    "3. Lihat DEPENDENCIES: python how_to_use.py --dependencies",
                    "4. Detail lengkap: python how_to_use.py --detail"
                ]
            },
            {
                "title": "📖 LEARNING PATH - Step by Step",
                "steps": [
                    "LEVEL 1: Start dengan config/config.yaml (understand settings)",
                    "LEVEL 2: Read utils/config.py (how config is loaded)",
                    "LEVEL 3: Understand data/binance_fetch.py (data source)",
                    "LEVEL 4: Learn features/indicators.py (technical analysis)",
                    "LEVEL 5: Study features/targets.py (target creation) 🎯",
                    "LEVEL 6: Master utils/adaptive_thresholds.py (auto-calculation) 🚀",
                    "LEVEL 6.5: Learn utils/path_utils.py (multi-pair support) 🆕",
                    "LEVEL 7: Full pipeline/build_dataset.py (orchestration)"
                ]
            },
            {
                "title": "🔍 DEBUGGING PATH - Ada Masalah?",
                "steps": [
                    "Check 1: config/config.yaml → Settings benar?",
                    "Check 2: data/db/ → Database files exist?",
                    "Check 3: Log outputs → Error messages?",
                    "Check 4: utils/adaptive_thresholds.py → Auto-calc working?",
                    "Check 5: features/targets.py → Labels generated?"
                ]
            },
            {
                "title": "⚡ POWER USER - Advanced Usage",
                "steps": [
                    "Modify adaptive_thresholds parameters dalam config.yaml",
                    "Add custom indicators dalam features/indicators.py",
                    "Extend auto-calculation logic dalam utils/adaptive_thresholds.py",
                    "Create custom target labels dalam features/targets.py",
                    "Build custom models dalam scripts/train_*.py"
                ]
            }
        ]
        
        for section in help_sections:
            print(f"\n{section['title']}")
            print("-" * 60)
            for step in section['steps']:
                print(f"  {step}")
        
        print(f"\n{'='*80}")
        print("💡 TIP: Start simple, then go deeper as you understand more!")


def main():
    parser = argparse.ArgumentParser(
        description="🗺️ Trade Predictor Project Explorer - Understand Everything!", 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python how_to_use.py              # Quick overview
  python how_to_use.py --flow        # Execution flow  
  python how_to_use.py --detail      # Detailed breakdown
  python how_to_use.py --deps        # Dependency map
  python how_to_use.py --help-me     # Interactive help
        """
    )
    
    parser.add_argument('--detail', action='store_true', 
                       help='Show detailed breakdown of all components')
    parser.add_argument('--flow', action='store_true',
                       help='Show step-by-step execution flow')
    parser.add_argument('--dependencies', '--deps', action='store_true',
                       help='Show dependency relationships')
    parser.add_argument('--help-me', action='store_true',
                       help='Interactive help untuk learning path')
    
    args = parser.parse_args()
    
    explorer = ProjectExplorer()
    
    if args.detail:
        explorer.show_detailed_breakdown()
    elif args.flow:
        explorer.show_execution_flow()
    elif args.dependencies:
        explorer.show_dependencies()
    elif args.help_me:
        explorer.interactive_help()
    else:
        explorer.show_overview()
        print(f"\n💡 Use --help untuk see all options")


if __name__ == '__main__':
    main()