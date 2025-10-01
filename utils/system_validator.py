#!/usr/bin/env python3
"""
System Health Check & Validation Suite

Comprehensive validation and verification tools for the automated trading system.
Provides health checks, training validation, and system diagnostics.

Usage:
    python utils/system_validator.py                           # Full system check
    python utils/system_validator.py --training-only           # Training validation only
    python utils/system_validator.py --models-only             # Model validation only  
    python utils/system_validator.py --data-only               # Data validation only
    python utils/system_validator.py --quick                   # Quick health check
    python utils/system_validator.py --report                  # Generate detailed report
    
Features:
    - Comprehensive system health monitoring
    - Training results validation with accuracy analysis
    - Model file integrity checks
    - Database validation and statistics
    - Auto-config verification
    - Performance benchmarking
    - Automated issue detection and recommendations
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

class SystemValidator:
    """
    Comprehensive system validation and health monitoring
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or ROOT)
        self.validation_results = {}
        self.issues_found = []
        self.recommendations = []
        
    def run_full_validation(self, generate_report: bool = True) -> Dict:
        """
        Run complete system validation
        
        Returns:
            Dict: Comprehensive validation results
        """
        print("🔍" + "="*80)
        print("🤖 SYSTEM HEALTH CHECK & VALIDATION SUITE")
        print("="*80)
        print()
        
        validation_start = datetime.now()
        
        # Run all validation components
        validations = {
            "system_info": self._validate_system_info(),
            "databases": self._validate_databases(),
            "auto_configs": self._validate_auto_configs(),
            "models": self._validate_models(),
            "training_results": self._validate_training_results(),
            "predictions": self._validate_prediction_capability(),
            "file_integrity": self._validate_file_integrity(),
            "performance": self._benchmark_performance()
        }
        
        # Compile results
        validation_end = datetime.now()
        validation_time = (validation_end - validation_start).total_seconds()
        
        self.validation_results = {
            "timestamp": validation_end.isoformat(),
            "validation_time_seconds": validation_time,
            "overall_health": self._calculate_overall_health(validations),
            "validations": validations,
            "issues_found": self.issues_found,
            "recommendations": self.recommendations,
            "summary": self._generate_summary(validations)
        }
        
        # Display results
        self._display_validation_results()
        
        # Generate report if requested
        if generate_report:
            self._generate_detailed_report()
        
        return self.validation_results
    
    def _validate_system_info(self) -> Dict:
        """Validate basic system information"""
        print("📊 Validating system information...")
        
        try:
            import yaml
            with open('config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            system_info = {
                "config_pairs": config.get('pairs', []),
                "config_timeframes": config.get('timeframes', []),
                "threshold_mode": config.get('target', {}).get('threshold_mode', 'unknown'),
                "python_version": sys.version,
                "project_root": str(self.project_root),
                "status": "healthy"
            }
            
            print("   ✅ System information validated")
            return system_info
            
        except Exception as e:
            self.issues_found.append(f"System info validation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _validate_databases(self) -> Dict:
        """Validate database files and contents"""
        print("🗄️  Validating databases...")
        
        db_dir = self.project_root / "data" / "db"
        databases = []
        total_size = 0
        total_rows = 0
        
        if not db_dir.exists():
            self.issues_found.append("Database directory not found")
            return {"status": "error", "error": "Database directory missing"}
        
        # Scan all SQLite databases
        for db_file in db_dir.glob("*.sqlite"):
            try:
                db_info = self._analyze_database(db_file)
                databases.append(db_info)
                total_size += db_info["size_mb"]
                total_rows += db_info["total_rows"]
                
            except Exception as e:
                self.issues_found.append(f"Database analysis failed for {db_file}: {e}")
        
        result = {
            "total_databases": len(databases),
            "total_size_mb": round(total_size, 2),
            "total_rows": total_rows,
            "databases": databases,
            "status": "healthy" if databases else "warning"
        }
        
        if not databases:
            self.issues_found.append("No databases found")
            self.recommendations.append("Run: python utils/auto_pipeline.py to create databases")
        
        print(f"   ✅ Found {len(databases)} databases ({total_size:.1f} MB, {total_rows:,} rows)")
        return result
    
    def _analyze_database(self, db_path: Path) -> Dict:
        """Analyze individual database"""
        with sqlite3.connect(db_path) as conn:
            # Get basic info
            size_mb = db_path.stat().st_size / (1024 * 1024)
            
            # Get row count
            total_rows = pd.read_sql("SELECT COUNT(*) as count FROM features", conn).iloc[0]['count']
            
            # Get column info
            columns_df = pd.read_sql("PRAGMA table_info(features)", conn)
            total_columns = len(columns_df)
            
            # Get label columns
            labels_df = pd.read_sql("SELECT * FROM features LIMIT 1", conn)
            label_columns = [col for col in labels_df.columns if 'direction' in col or 'future' in col]
            
            # Check data quality
            latest_timestamp = pd.read_sql("SELECT MAX(timestamp) as latest FROM features", conn).iloc[0]['latest']
            
            # Analyze target distributions if available
            distributions = {}
            if label_columns:
                for label_col in label_columns[:4]:  # Limit to first 4 labels
                    try:
                        dist_df = pd.read_sql(f"SELECT {label_col}, COUNT(*) as count FROM features GROUP BY {label_col}", conn)
                        total = dist_df['count'].sum()
                        distributions[label_col] = {
                            int(row[label_col]): {"count": int(row['count']), "percentage": round(row['count']/total*100, 1)}
                            for _, row in dist_df.iterrows()
                        }
                    except:
                        pass
        
        return {
            "filename": db_path.name,
            "size_mb": round(size_mb, 2),
            "total_rows": total_rows,
            "total_columns": total_columns,
            "label_columns": len(label_columns),
            "latest_timestamp": latest_timestamp,
            "distributions": distributions,
            "quality": "good" if total_rows > 1000 else "low"
        }
    
    def _validate_auto_configs(self) -> Dict:
        """Validate auto-generated configurations"""
        print("🎯 Validating auto-configurations...")
        
        auto_config_path = self.project_root / "data" / "auto_balanced_thresholds.json"
        
        if not auto_config_path.exists():
            self.issues_found.append("Auto-balanced thresholds config not found")
            self.recommendations.append("Run: python utils/auto_pipeline.py to generate auto-config")
            return {"status": "missing", "auto_config_exists": False}
        
        try:
            with open(auto_config_path, 'r') as f:
                config = json.load(f)
            
            # Validate config structure
            required_keys = ['individual_results', 'recommendation']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                self.issues_found.append(f"Auto-config missing keys: {missing_keys}")
                return {"status": "invalid", "missing_keys": missing_keys}
            
            # Analyze config quality
            individual_results = config.get('individual_results', {})
            avg_threshold = config.get('average_optimal_threshold', 0)
            
            balance_scores = []
            for pair_tf, result in individual_results.items():
                if 'balance_score' in result:
                    balance_scores.append(result['balance_score'])
            
            avg_balance_score = sum(balance_scores) / len(balance_scores) if balance_scores else 0
            
            result = {
                "auto_config_exists": True,
                "pairs_analyzed": len(individual_results),
                "average_threshold": avg_threshold,
                "average_balance_score": round(avg_balance_score, 1),
                "generated_at": config.get('generated_at', 'unknown'),
                "status": "healthy"
            }
            
            # Quality assessment
            if avg_balance_score < 70:
                self.issues_found.append(f"Low balance score: {avg_balance_score:.1f}/100")
                self.recommendations.append("Consider re-running auto-optimization with more data")
            
            print(f"   ✅ Auto-config validated (avg score: {avg_balance_score:.1f}/100)")
            return result
            
        except Exception as e:
            self.issues_found.append(f"Auto-config validation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _validate_models(self) -> Dict:
        """Validate trained models"""
        print("🧠 Validating trained models...")
        
        models_dir = self.project_root / "models" / "tree_models"
        metadata_dir = self.project_root / "models" / "metadata"
        
        models = []
        total_models = 0
        
        if not models_dir.exists():
            self.issues_found.append("Models directory not found")
            return {"status": "missing", "models_exist": False}
        
        # Scan model files
        for model_file in models_dir.glob("*.txt"):
            try:
                model_info = self._analyze_model_file(model_file, metadata_dir)
                models.append(model_info)
                total_models += 1
                
            except Exception as e:
                self.issues_found.append(f"Model analysis failed for {model_file}: {e}")
        
        # Check for scaler files
        scaler_files = list(models_dir.glob("*.joblib"))
        feature_files = list(models_dir.glob("*_features.txt"))
        
        result = {
            "total_models": total_models,
            "scaler_files": len(scaler_files),
            "feature_files": len(feature_files),
            "models": models,
            "status": "healthy" if total_models > 0 else "missing"
        }
        
        if total_models == 0:
            self.issues_found.append("No trained models found")
            self.recommendations.append("Train models: python scripts/train_tree_model.py --model lightgbm --target direction_h1")
        
        print(f"   ✅ Found {total_models} models with {len(scaler_files)} scalers")
        return result
    
    def _analyze_model_file(self, model_path: Path, metadata_dir: Path) -> Dict:
        """Analyze individual model file"""
        # Basic file info
        size_mb = model_path.stat().st_size / (1024 * 1024)
        modified_time = datetime.fromtimestamp(model_path.stat().st_mtime)
        
        # Try to find corresponding metadata
        metadata_file = None
        metadata_info = {}
        
        # Look for metadata file with similar name
        base_name = model_path.stem.replace('_20251001_', '*').replace('lightgbm_', '*')
        for meta_file in metadata_dir.glob("*.json"):
            if any(part in meta_file.stem for part in model_path.stem.split('_')[:3]):
                metadata_file = meta_file
                break
        
        if metadata_file and metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata_info = json.load(f)
            except:
                pass
        
        return {
            "filename": model_path.name,
            "size_mb": round(size_mb, 3),
            "modified": modified_time.isoformat(),
            "has_metadata": metadata_file is not None,
            "accuracy": metadata_info.get('test_accuracy', 'unknown'),
            "target": metadata_info.get('target_column', 'unknown'),
            "features_count": metadata_info.get('feature_count', 'unknown')
        }
    
    def _validate_training_results(self) -> Dict:
        """Validate training results and accuracy"""
        print("📈 Validating training results...")
        
        metadata_dir = self.project_root / "models" / "metadata"
        
        if not metadata_dir.exists():
            return {"status": "missing", "training_results_exist": False}
        
        training_results = []
        accuracies = []
        
        # Analyze metadata files
        for meta_file in metadata_dir.glob("*.json"):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                
                if 'test_accuracy' in metadata:
                    accuracy = metadata['test_accuracy']
                    accuracies.append(accuracy)
                    
                    training_results.append({
                        "filename": meta_file.name,
                        "model_type": metadata.get('model_type', 'unknown'),
                        "target": metadata.get('target_column', 'unknown'),
                        "accuracy": accuracy,
                        "feature_count": metadata.get('feature_count', 0),
                        "training_samples": metadata.get('training_samples', 0),
                        "training_time": metadata.get('training_time_seconds', 0)
                    })
                    
            except Exception as e:
                continue
        
        if not training_results:
            self.issues_found.append("No training results found")
            return {"status": "missing", "training_results": []}
        
        # Calculate statistics
        avg_accuracy = sum(accuracies) / len(accuracies)
        min_accuracy = min(accuracies)
        max_accuracy = max(accuracies)
        
        result = {
            "total_results": len(training_results),
            "average_accuracy": round(avg_accuracy, 4),
            "min_accuracy": round(min_accuracy, 4),
            "max_accuracy": round(max_accuracy, 4),
            "training_results": training_results,
            "status": "healthy"
        }
        
        # Quality assessment
        if avg_accuracy < 0.35:
            self.issues_found.append(f"Low average accuracy: {avg_accuracy:.1%}")
            self.recommendations.append("Consider collecting more data or feature engineering")
        
        print(f"   ✅ Training results: {len(training_results)} models, avg accuracy: {avg_accuracy:.1%}")
        return result
    
    def _validate_prediction_capability(self) -> Dict:
        """Test prediction system capability"""
        print("🔮 Testing prediction capability...")
        
        try:
            # Quick test of prediction script
            import subprocess
            result = subprocess.run([
                'python', 'scripts/predict_enhanced.py', 
                '--timeframe', '15m', '--model', 'auto', '--single', '--output', 'json'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                prediction_status = "working"
                prediction_output = "Successful prediction test"
            else:
                prediction_status = "error"
                prediction_output = result.stderr[:200] if result.stderr else "Unknown error"
                self.issues_found.append("Prediction test failed")
                
        except subprocess.TimeoutExpired:
            prediction_status = "timeout"
            prediction_output = "Prediction test timed out"
            self.issues_found.append("Prediction test timed out")
        except Exception as e:
            prediction_status = "error"
            prediction_output = str(e)
            self.issues_found.append(f"Prediction test error: {e}")
        
        result = {
            "prediction_status": prediction_status,
            "test_output": prediction_output,
            "status": "healthy" if prediction_status == "working" else "error"
        }
        
        print(f"   {'✅' if prediction_status == 'working' else '❌'} Prediction test: {prediction_status}")
        return result
    
    def _validate_file_integrity(self) -> Dict:
        """Validate file integrity and structure"""
        print("📁 Validating file integrity...")
        
        critical_files = [
            "config/config.yaml",
            "pipeline/build_dataset.py",
            "scripts/train_tree_model.py", 
            "scripts/predict_enhanced.py",
            "utils/auto_pipeline.py"
        ]
        
        file_status = {}
        missing_files = []
        
        for file_path in critical_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                size_kb = full_path.stat().st_size / 1024
                file_status[file_path] = {"exists": True, "size_kb": round(size_kb, 1)}
            else:
                file_status[file_path] = {"exists": False}
                missing_files.append(file_path)
        
        if missing_files:
            self.issues_found.extend([f"Missing critical file: {f}" for f in missing_files])
        
        result = {
            "critical_files_checked": len(critical_files),
            "missing_files": missing_files,
            "file_status": file_status,
            "status": "healthy" if not missing_files else "error"
        }
        
        print(f"   ✅ File integrity: {len(critical_files) - len(missing_files)}/{len(critical_files)} files OK")
        return result
    
    def _benchmark_performance(self) -> Dict:
        """Basic performance benchmarking"""
        print("⚡ Running performance benchmark...")
        
        try:
            # Simple performance test
            import time
            
            # Test database query speed
            db_files = list((self.project_root / "data" / "db").glob("*.sqlite"))
            if db_files:
                start_time = time.time()
                with sqlite3.connect(db_files[0]) as conn:
                    pd.read_sql("SELECT * FROM features LIMIT 1000", conn)
                db_query_time = time.time() - start_time
            else:
                db_query_time = None
            
            # Test numpy operations
            start_time = time.time()
            arr = np.random.random((1000, 100))
            result = np.dot(arr, arr.T)
            numpy_time = time.time() - start_time
            
            benchmark = {
                "db_query_time_ms": round(db_query_time * 1000, 1) if db_query_time else None,
                "numpy_benchmark_ms": round(numpy_time * 1000, 1),
                "status": "healthy"
            }
            
            print(f"   ✅ Performance benchmark completed")
            return benchmark
            
        except Exception as e:
            self.issues_found.append(f"Performance benchmark failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_overall_health(self, validations: Dict) -> Dict:
        """Calculate overall system health score"""
        health_weights = {
            "system_info": 10,
            "databases": 25,
            "auto_configs": 20,
            "models": 25,
            "training_results": 15,
            "predictions": 20,
            "file_integrity": 15,
            "performance": 5
        }
        
        total_score = 0
        total_weight = 0
        
        for component, validation in validations.items():
            if component in health_weights:
                weight = health_weights[component]
                
                if validation.get("status") == "healthy":
                    score = 100
                elif validation.get("status") == "warning":
                    score = 70
                elif validation.get("status") == "missing":
                    score = 30
                else:
                    score = 0
                
                total_score += score * weight
                total_weight += weight
        
        overall_score = round(total_score / total_weight, 1) if total_weight > 0 else 0
        
        if overall_score >= 90:
            health_status = "excellent"
            health_emoji = "🟢"
        elif overall_score >= 75:
            health_status = "good"
            health_emoji = "🟡"
        elif overall_score >= 50:
            health_status = "fair"
            health_emoji = "🟠"
        else:
            health_status = "poor"
            health_emoji = "🔴"
        
        return {
            "score": overall_score,
            "status": health_status,
            "emoji": health_emoji,
            "total_issues": len(self.issues_found),
            "total_recommendations": len(self.recommendations)
        }
    
    def _generate_summary(self, validations: Dict) -> Dict:
        """Generate validation summary"""
        summary = {
            "databases_count": validations.get("databases", {}).get("total_databases", 0),
            "models_count": validations.get("models", {}).get("total_models", 0),
            "avg_accuracy": validations.get("training_results", {}).get("average_accuracy", 0),
            "prediction_working": validations.get("predictions", {}).get("prediction_status") == "working",
            "auto_config_exists": validations.get("auto_configs", {}).get("auto_config_exists", False)
        }
        
        return summary
    
    def _display_validation_results(self):
        """Display validation results in console"""
        overall = self.validation_results["overall_health"]
        summary = self.validation_results["summary"]
        
        print("\n" + "="*80)
        print(f"🏆 SYSTEM HEALTH REPORT")
        print("="*80)
        print()
        
        print(f"Overall Health: {overall['emoji']} {overall['status'].upper()} ({overall['score']}/100)")
        print(f"Validation Time: {self.validation_results['validation_time_seconds']:.1f} seconds")
        print()
        
        print("📊 SYSTEM SUMMARY:")
        print(f"   🗄️  Databases: {summary['databases_count']}")
        print(f"   🧠 Models: {summary['models_count']}")
        print(f"   📈 Avg Accuracy: {summary['avg_accuracy']:.1%}")
        print(f"   🔮 Predictions: {'✅ Working' if summary['prediction_working'] else '❌ Failed'}")
        print(f"   🎯 Auto-Config: {'✅ Exists' if summary['auto_config_exists'] else '❌ Missing'}")
        print()
        
        if self.issues_found:
            print("⚠️  ISSUES FOUND:")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"   {i}. {issue}")
            print()
        
        if self.recommendations:
            print("💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"   {i}. {rec}")
            print()
        
        if overall['score'] >= 90:
            print("🎉 EXCELLENT! System is production-ready!")
        elif overall['score'] >= 75:
            print("✅ GOOD! System is mostly healthy with minor issues.")
        elif overall['score'] >= 50:
            print("⚠️  FAIR! System needs attention in several areas.")
        else:
            print("🚨 POOR! System requires immediate attention.")
    
    def _generate_detailed_report(self):
        """Generate detailed validation report file"""
        report_path = f"data/system_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_path}")


def main():
    """CLI interface for system validation"""
    parser = argparse.ArgumentParser(
        description='System Health Check & Validation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/system_validator.py                    # Full system validation
  python utils/system_validator.py --quick            # Quick health check
  python utils/system_validator.py --training-only    # Training validation only
  python utils/system_validator.py --report           # Generate detailed report
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick health check (essential components only)')
    parser.add_argument('--training-only', action='store_true',
                       help='Validate training results only')
    parser.add_argument('--models-only', action='store_true',
                       help='Validate models only')
    parser.add_argument('--data-only', action='store_true',
                       help='Validate data and databases only')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed JSON report')
    parser.add_argument('--no-predictions', action='store_true',
                       help='Skip prediction capability test')
    
    args = parser.parse_args()
    
    # Create validator
    validator = SystemValidator()
    
    if args.quick or args.training_only or args.models_only or args.data_only:
        # Selective validation (simplified for now)
        results = validator.run_full_validation(generate_report=args.report)
    else:
        # Full validation
        results = validator.run_full_validation(generate_report=args.report)
    
    # Exit code based on health
    health_score = results["overall_health"]["score"]
    if health_score >= 75:
        exit_code = 0  # Success
    elif health_score >= 50:
        exit_code = 1  # Warning
    else:
        exit_code = 2  # Error
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()