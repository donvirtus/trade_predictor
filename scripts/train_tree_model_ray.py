#!/usr/bin/env python3
"""
Universal training script for all timeframes with flexible MA configuration using Ray for distributed training
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import warnings
import yaml
import argparse
warnings.filterwarnings('ignore')
from tqdm import tqdm
import hashlib

# Add Ray import
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray not available. Please install: pip install ray")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.logging import get_logger
from utils.config import load_config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = get_logger('train_universal_model')

class UniversalModelTrainer:
    def __init__(self, timeframe="15m", models_dir="models/tree_models", config_path="config/config.yaml", overwrite: bool = False,
                 target_name: str = 'direction'):
        self.timeframe = timeframe
        self.models_dir = models_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = load_config(config_path)
        self.overwrite = overwrite
        self.target_name = target_name  # selected target column (may be direction_h5, future_max_high_h20, etc.)
        
        # Determine database path based on timeframe (support extended list)
        timeframe_map = {
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
        }
        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        self.db_path = f"data/db/btc_{timeframe_map[timeframe]}.sqlite"
        
        # Create models directory if not exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info(f"Initialized UniversalModelTrainer for {timeframe} timeframe")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Models dir: {self.models_dir}")

    def load_data(self):
        """Load and prepare data from database with flexible MA configuration"""
        logger.info(f"Loading data from {self.db_path}")
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load all data first
        df = pd.read_sql_query("SELECT * FROM features ORDER BY timestamp", conn)
        conn.close()
        
        logger.info(f"Loaded {len(df)} rows from database")
        
        if df.empty:
            raise ValueError("No data found in database")
        
        return df

    def prepare_features(self, df):
        """Prepare features: auto-select numeric engineered features safely for a chosen target.

        Supports classification (direction-like) and regression (extremes: future_max_high_h*, future_min_low_h*).
        """
        logger.info("Preparing features (auto-detect numeric engineered features) ...")

        target_column = self.target_name
        if target_column not in df.columns:
            # Fallback logic for legacy default
            if target_column == 'direction' and 'direction' not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset.")
            else:
                raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")

        # Determine task type
        if target_column.startswith('future_max_high_') or target_column.startswith('future_min_low_'):
            task_type = 'regression'
        else:
            # Assume classification for direction-like targets
            task_type = 'classification'
        self.task_type = task_type
        logger.info(f"Detected task type: {task_type} for target {target_column}")

        leakage_cols = {
            'future_return_pct', 'future_close', 'future_open', 'future_high', 'future_low'
        }
        # Add all other label columns to exclusion to avoid leakage across horizons
        label_prefixes = [
            'direction_h', 'future_return_pct_h', 'future_max_high_h', 'future_min_low_h', 'time_to_max_high_h', 'time_to_min_low_h'
        ]
        # Explicit single-label names that must never appear as features (classification base)
        explicit_label_names = {'direction'}
        for c in df.columns:
            if c in explicit_label_names and c != target_column:
                leakage_cols.add(c)
                continue
            if any(c.startswith(p) for p in label_prefixes) and c != target_column:
                leakage_cols.add(c)

        id_meta_cols = {'timestamp', 'ts', 'time', 'pair', 'symbol', 'timeframe', 'index', 'id'}
        exclude = leakage_cols | id_meta_cols | {target_column}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [c for c in numeric_cols if c not in exclude]
        if not available_features:
            raise ValueError("No numeric features found after exclusion. Please check dataset columns.")

        y = df[target_column].copy()
        X = df[available_features].copy()

        mask = y.notna()
        X = X[mask]
        y = y[mask]

        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)

        logger.info(f"Using target column: {target_column} (task={task_type})")
        logger.info(f"Feature count: {len(available_features)}")
        logger.info(f"Sample features (<=50): {available_features[:50]}")
        logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
        if task_type == 'classification':
            try:
                logger.info(f"Target distribution:\n{y.value_counts().sort_index()}")
            except Exception:
                pass
        else:
            logger.info(f"Target stats: mean={y.mean():.5f} std={y.std():.5f} min={y.min():.5f} max={y.max():.5f}")

        return X, y, available_features

    def train_lightgbm(self, X_train, X_test, y_train, y_test):
        """Train LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available")
            return None, 0.0
        logger.info("Training LightGBM model ...")
        # Progress bar for boosting rounds
        total_rounds = 1000
        pbar = tqdm(total=total_rounds, desc="LightGBM Training", leave=False)

        def tqdm_callback(env):
            # Called every iteration
            try:
                pbar.update(1)
            except Exception:
                pass
            return False
        # Create datasets (ensure feature names preserved)
        feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=feature_names)
        
        # Parameters
        if getattr(self, 'task_type', 'classification') == 'regression':
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        else:
            params = {
                'objective': 'multiclass',
                'num_class': len(np.unique(y_train)),
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        
        # Train
        try:
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=total_rounds,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0), tqdm_callback]
            )
        finally:
            pbar.close()
        
        # Evaluate
        if getattr(self, 'task_type', 'classification') == 'regression':
            preds = model.predict(X_test)
            # Compute simple RMSE
            rmse = float(np.sqrt(np.mean((preds - y_test.values) ** 2)))
            logger.info(f"LightGBM RMSE: {rmse:.6f}")
            return model, rmse
        else:
            y_pred = np.argmax(model.predict(X_test), axis=1)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"LightGBM Accuracy: {accuracy:.4f}")
            return model, accuracy

    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available")
            return None, 0.0
        
        logger.info("Training XGBoost model...")
        # Some distributions of xgboost do not expose XGBClassifier at the top level
        try:
            XGBClassifier = xgb.XGBClassifier  # type: ignore[attr-defined]
        except AttributeError:
            try:
                from xgboost.sklearn import XGBClassifier  # type: ignore
            except Exception as e:
                logger.error(f"Failed to import XGBClassifier: {e}")
                return None, 0.0

        if getattr(self, 'task_type', 'classification') == 'regression':
            model = XGBClassifier(  # Will patch below if regression
                n_estimators=800,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric='rmse',
                early_stopping_rounds=100
            )
            # For regression use single output; override in predict path if needed
        else:
            model = XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric='mlogloss',
                early_stopping_rounds=100
            )
        # Progress bar via XGBoost callback
        try:
            from xgboost.callback import TrainingCallback
        except Exception:
            TrainingCallback = object  # type: ignore

        class TQDMCallback(TrainingCallback):  # type: ignore
            def __init__(self, total=1000, desc="XGBoost Training"):
                self.pbar = tqdm(total=total, desc=desc, leave=False)
            def after_iteration(self, model, epoch, evals_log):
                try:
                    self.pbar.update(1)
                except Exception:
                    pass
                return False
            def after_training(self, model):
                try:
                    self.pbar.close()
                except Exception:
                    pass
                return model

        # Train
        import inspect
        supports_callbacks = False
        try:
            sig = inspect.signature(model.fit)
            supports_callbacks = 'callbacks' in sig.parameters
        except Exception:
            supports_callbacks = False

        if supports_callbacks:
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
                callbacks=[TQDMCallback(total=model.get_params().get('n_estimators', 1000))]
            )
        else:
            logger.info("XGBoost version without sklearn callbacks; showing textual iteration logs (verbose=True).")
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=True
            )
        
        # Evaluate
        if getattr(self, 'task_type', 'classification') == 'regression':
            preds = model.predict(X_test)
            rmse = float(np.sqrt(np.mean((preds - y_test.values) ** 2)))
            logger.info(f"XGBoost RMSE: {rmse:.6f}")
            return model, rmse
        else:
            accuracy = model.score(X_test, y_test)
            logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
            return model, accuracy

    def train_catboost(self, X_train, X_test, y_train, y_test):
        """Train CatBoost model"""
        if not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not available")
            return None, 0.0
        
        logger.info("Training CatBoost model...")
        
        if getattr(self, 'task_type', 'classification') == 'regression':
            model = cb.CatBoostRegressor(
                iterations=1000,
                depth=6,
                learning_rate=0.05,
                loss_function='RMSE',
                random_seed=42,
                verbose=50,
                early_stopping_rounds=100
            )
        else:
            model = cb.CatBoostClassifier(
                iterations=1000,
                depth=6,
                learning_rate=0.05,
                loss_function='MultiClass',
                eval_metric='Accuracy',
                random_seed=42,
                verbose=50,
                early_stopping_rounds=100
            )
        
        # Train
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            use_best_model=True
        )
        
        # Evaluate
        if getattr(self, 'task_type', 'classification') == 'regression':
            preds = model.predict(X_test)
            rmse = float(np.sqrt(np.mean((preds - y_test.values) ** 2)))
            logger.info(f"CatBoost RMSE: {rmse:.6f}")
            return model, rmse
        else:
            accuracy = model.score(X_test, y_test)
            logger.info(f"CatBoost Accuracy: {accuracy:.4f}")
            return model, accuracy

    def save_model(self, model, model_type, metric_value, scaler, features):
        """Simpan model + scaler + fitur + metadata per TARGET.

        Perubahan utama:
        - File metadata kini unik per target: btc_<tf>_<model>_<target>_metadata.json
        - Scaler & features juga per target agar tidak saling override ketika multi-target.
        - Mode overwrite kini hanya menghapus artefak untuk target yang sama, bukan semua target lain.
        - Tambahkan/refresh registry sederhana (models/metadata/registry_<model>_<tf>.json) berisi daftar target yang tersedia.
        """
        # Timestamp retained for metadata versioning only
        timestamp = self.timestamp

        import glob, json

        model_type_lower = model_type.lower()

        # Helper: remove files matching patterns
        def _safe_remove(paths):
            for p in paths:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        logger.info(f"[overwrite] Removed old artifact: {p}")
                    except Exception as e:
                        logger.warning(f"[overwrite] Failed to remove {p}: {e}")

        target_suffix = self.target_name
        target_suffix_clean = target_suffix.replace('future_', '').replace('pct_', '').replace('__', '_')

        if self.overwrite:
            try:
                # Hapus hanya artefak untuk target sama (model, scaler, fitur, metadata)
                patterns = [
                    os.path.join(self.models_dir, f"btc_{self.timeframe}_{model_type_lower}_{target_suffix_clean}.*"),
                    os.path.join(self.models_dir, f"btc_{self.timeframe}_scaler_{target_suffix_clean}.*"),
                    os.path.join(self.models_dir, f"btc_{self.timeframe}_{model_type_lower}_features_{target_suffix_clean}.txt"),
                ]
                meta_pattern = os.path.join('models','metadata', f"btc_{self.timeframe}_{model_type_lower}_{target_suffix_clean}_metadata.json")
                for pat in patterns:
                    for old in glob.glob(pat):
                        _safe_remove([old])
                if os.path.exists(meta_pattern):
                    _safe_remove([meta_pattern])
            except Exception as e:
                logger.warning(f"Target-specific overwrite cleanup error: {e}")

        # Tentukan nama file model + scaler + fitur per target
        if self.overwrite:
            if model_type == 'LightGBM':
                model_filename = f"btc_{self.timeframe}_lightgbm_{target_suffix_clean}.txt"
            elif model_type == 'XGBoost':
                model_filename = f"btc_{self.timeframe}_xgboost_{target_suffix_clean}.json"
            elif model_type == 'CatBoost':
                model_filename = f"btc_{self.timeframe}_catboost_{target_suffix_clean}.cbm"
            else:
                model_filename = f"btc_{self.timeframe}_{model_type_lower}_{target_suffix_clean}.joblib"
            scaler_filename = f"btc_{self.timeframe}_scaler_{target_suffix_clean}.joblib"
        else:
            if model_type == 'LightGBM':
                model_filename = f"btc_{self.timeframe}_lightgbm_{target_suffix_clean}_{timestamp}.txt"
            elif model_type == 'XGBoost':
                model_filename = f"btc_{self.timeframe}_xgboost_{target_suffix_clean}_{timestamp}.json"
            elif model_type == 'CatBoost':
                model_filename = f"btc_{self.timeframe}_catboost_{target_suffix_clean}_{timestamp}.cbm"
            else:
                model_filename = f"btc_{self.timeframe}_{model_type_lower}_{target_suffix_clean}_{timestamp}.joblib"
            scaler_filename = f"btc_{self.timeframe}_scaler_{target_suffix_clean}_{timestamp}.joblib"

        model_path = os.path.join(self.models_dir, model_filename)
        scaler_path = os.path.join(self.models_dir, scaler_filename)

        # Persist model
        if model_type == 'LightGBM':
            model.save_model(model_path)
        elif model_type == 'XGBoost':
            model.save_model(model_path)
        elif model_type == 'CatBoost':
            model.save_model(model_path)
        else:
            joblib.dump(model, model_path)

        # Persist scaler
        joblib.dump(scaler, scaler_path)

        # Features list per target
        features_filename = f"btc_{self.timeframe}_{model_type_lower}_features_{target_suffix_clean}.txt"
        features_path = os.path.join(self.models_dir, features_filename)
        with open(features_path, 'w') as f:
            f.write('\n'.join(features))

        # Metadata per target
        metadata_filename = f"btc_{self.timeframe}_{model_type_lower}_{target_suffix_clean}_metadata.json"
        metadata_dir = os.path.join("models", "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, metadata_filename)

        # Compute feature hash
        features_hash = hashlib.sha256('\n'.join(features).encode('utf-8')).hexdigest()

        metadata = {
            "model_type": model_type,
            "timeframe": self.timeframe,
            "training_timestamp": timestamp,
            "metric_value": float(metric_value),
            "metric_name": 'rmse' if getattr(self, 'task_type', 'classification') == 'regression' else 'accuracy',
            "features_count": len(features),
            "features_hash": features_hash,
            "features_hash_algorithm": "sha256",
            "features_version": timestamp,
            "ma_features_enabled": getattr(self.config, 'enable_ma_features', True),
            "ma_periods": getattr(self.config, 'ma_periods', []),
            "bb_periods": getattr(self.config, 'bb_periods', []),
            "model_file": model_filename,
            "scaler_file": scaler_filename,
            "features_file": features_filename,
            "overwrite_mode": bool(self.overwrite),
            "target_name": self.target_name,
            "task_type": getattr(self, 'task_type', 'classification')
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update registry per model_type + timeframe
        try:
            registry_dir = metadata_dir
            registry_file = os.path.join(registry_dir, f"registry_{model_type_lower}_{self.timeframe}.json")
            import json as _json
            if os.path.exists(registry_file):
                with open(registry_file, 'r') as rf:
                    reg = _json.load(rf)
            else:
                reg = {"timeframe": self.timeframe, "model_type": model_type, "targets": []}
            # Replace or append target entry
            reg['targets'] = [t for t in reg.get('targets', []) if t.get('target_name') != self.target_name]
            reg['targets'].append({
                "target_name": self.target_name,
                "model_file": model_filename,
                "scaler_file": scaler_filename,
                "features_file": features_filename,
                "metadata_file": metadata_filename,
                "metric_name": metadata['metric_name'],
                "metric_value": metadata['metric_value'],
                "task_type": metadata['task_type'],
                "timestamp": metadata['training_timestamp']
            })
            with open(registry_file, 'w') as rf:
                _json.dump(reg, rf, indent=2)
            logger.info(f"Updated registry: {registry_file}")
        except Exception as e:
            logger.warning(f"Failed updating registry: {e}")

        logger.info(f"Saved {model_type} model: {model_path}")
        logger.info(f"Saved scaler: {scaler_path}")
        logger.info(f"Saved features: {features_path}")
        logger.info(f"Saved metadata: {metadata_path}")

    def train_single_model(self, model_name):
        """Train a specific model by name for the configured target."""
        logger.info(f"Starting training for {model_name} (target={self.target_name}) ...")
        df = self.load_data()
        X, y, features = self.prepare_features(df)
        if len(X) == 0:
            raise ValueError("No valid samples after feature preparation")

        stratify_arg = y if getattr(self, 'task_type', 'classification') == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_arg
        )

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        if model_name == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ValueError("LightGBM is not available. Please install it: pip install lightgbm")
            model, metric_value = self.train_lightgbm(X_train_scaled, X_test_scaled, y_train, y_test)
            display_name = 'LightGBM'
        elif model_name == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is not available. Please install it: pip install xgboost")
            model, metric_value = self.train_xgboost(X_train_scaled, X_test_scaled, y_train, y_test)
            display_name = 'XGBoost'
        elif model_name == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ValueError("CatBoost is not available. Please install it: pip install catboost")
            model, metric_value = self.train_catboost(X_train_scaled, X_test_scaled, y_train, y_test)
            display_name = 'CatBoost'
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if model:
            self.save_model(model, display_name, metric_value, scaler, features)
            logger.info(f"Training completed for {display_name} (target={self.target_name})!")
            return (display_name, metric_value)
        logger.error(f"Training failed for {display_name}")
        return None

    def train_all_models(self):
        """Train all available models"""
        logger.info("Starting model training process...")
        
        # Load data
        df = self.load_data()
        
        # Prepare features
        X, y, features = self.prepare_features(df)
        
        if len(X) == 0:
            raise ValueError("No valid samples after feature preparation")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features but keep DataFrame with column names
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        models_results = {}
        
        # Train LightGBM
        if LIGHTGBM_AVAILABLE:
            lgb_model, lgb_accuracy = self.train_lightgbm(X_train_scaled, X_test_scaled, y_train, y_test)
            if lgb_model:
                self.save_model(lgb_model, 'LightGBM', lgb_accuracy, scaler, features)
                models_results['LightGBM'] = lgb_accuracy
        
        # Train XGBoost
        if XGBOOST_AVAILABLE:
            xgb_model, xgb_accuracy = self.train_xgboost(X_train_scaled, X_test_scaled, y_train, y_test)
            if xgb_model:
                self.save_model(xgb_model, 'XGBoost', xgb_accuracy, scaler, features)
                models_results['XGBoost'] = xgb_accuracy
        
        # Train CatBoost
        if CATBOOST_AVAILABLE:
            cb_model, cb_accuracy = self.train_catboost(X_train_scaled, X_test_scaled, y_train, y_test)
            if cb_model:
                self.save_model(cb_model, 'CatBoost', cb_accuracy, scaler, features)
                models_results['CatBoost'] = cb_accuracy
        
        # Summary
        logger.info("Training completed!")
        logger.info(f"Models trained for timeframe: {self.timeframe}")
        
        if models_results:
            logger.info("Model Performance Summary:")
            for model_name, accuracy in models_results.items():
                logger.info(f"  {model_name}: {accuracy:.4f}")
            
            best_model = max(models_results.items(), key=lambda x: x[1])
            logger.info(f"Best model: {best_model[0]} with accuracy {best_model[1]:.4f}")
        else:
            logger.error("No models were successfully trained!")
        
        return models_results

@ray.remote(num_cpus=1)  # Use 1 CPU per task to maximize parallelism and let Ray balance based on node speed
def train_single_remote(timeframe, config_path, overwrite, target_name, model_name):
    """Remote function for training a single model combination using Ray"""
    try:
        trainer = UniversalModelTrainer(
            timeframe=timeframe,
            config_path=config_path,
            overwrite=overwrite,
            target_name=target_name
        )
        result = trainer.train_single_model(model_name)
        return (timeframe, target_name, model_name, result)
    except Exception as e:
        logger.error(f"Training failed for {timeframe}/{target_name}/{model_name}: {e}")
        return (timeframe, target_name, model_name, None)

def main():
    if not RAY_AVAILABLE:
        print("Ray is required for distributed training. Please install: pip install ray")
        sys.exit(1)

    class CaseInsensitiveChoice(argparse.Action):
        def __init__(self, option_strings, dest, choices, **kwargs):
            self.choices = [choice.lower() for choice in choices]
            self.original_choices = choices
            super().__init__(option_strings, dest, **kwargs)
        
        def __call__(self, parser, namespace, values, option_string=None):
            if values is None:
                return
            if str(values).lower() not in [str(choice).lower() for choice in self.original_choices]:
                parser.error(f"argument {option_string}: invalid choice: '{values}' "
                           f"(choose from {', '.join(self.original_choices)})")
            setattr(namespace, self.dest, str(values).lower())
    
    parser = argparse.ArgumentParser(description='Universal Model Training for Crypto Prediction (multi-horizon + extremes aware) with Ray distributed training')
    parser.add_argument('--timeframe', choices=['5m', '15m', '30m', '1h', '2h', '4h', '6h', 'all', 'config'], default='15m',
                       help="Timeframe to train models for. Use 'all' or 'config' to train all timeframes listed in config.yaml")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', action=CaseInsensitiveChoice, 
                       choices=['LightGBM', 'XGBoost', 'CatBoost'],
                       help='Specific model to train (case insensitive). If not specified, train all available models')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite stable filenames (produce stable artifacts per target)')
    parser.add_argument('--target', type=str, default='direction',
                       help='Target column to train (e.g.: direction, direction_h1, direction_h5, direction_h20, future_max_high_h20, future_min_low_h20)')
    parser.add_argument('--multi-horizon-all', action='store_true',
                       help='Train all direction_* horizons present (direction_h1,h5,h20) for selected model(s)')
    parser.add_argument('--include-extremes', action='store_true',
                       help='Also train extremes targets (future_max_high_h{base}, future_min_low_h{base}) if present')
    parser.add_argument('--ray-address', type=str, default='auto',
                       help='Ray cluster address. Use "auto" for auto-detection, or specify head node IP:port')
    
    args = parser.parse_args()
    
    # Initialize Ray
    print(f"Initializing Ray with address: {args.ray_address}")
    ray.init(address=args.ray_address)
    
    # Model name is already lowercase from CaseInsensitiveChoice
    model_name = args.model
    target_arg = args.target
    
    try:
        # Determine requested timeframes
        cfg = load_config(args.config)
        requested = args.timeframe
        if requested in ('all', 'config'):
            tfs = list(getattr(cfg, 'timeframes', []) or [])
            if not tfs:
                tfs = ['5m', '15m', '30m', '1h', '2h', '4h', '6h']
        else:
            tfs = [requested]

        # Collect all training tasks
        training_tasks = []
        
        for tf in tfs:
            print(f"\n=== Collecting tasks for timeframe: {tf} ===")
            # Load dataset once to inspect available targets
            temp_trainer = UniversalModelTrainer(timeframe=tf, config_path=args.config, overwrite=args.overwrite, target_name=target_arg)
            df_preview = temp_trainer.load_data()
            available_cols = set(df_preview.columns)
            direction_targets = [c for c in ['direction_h1','direction_h5','direction_h10','direction_h20','direction'] if c in available_cols]
            extremes_targets = [c for c in available_cols if c.startswith('future_max_high_h') or c.startswith('future_min_low_h')]

            # Build target list according to args
            targets_to_train = []
            if args.multi_horizon_all:
                targets_to_train.extend(direction_targets)
            else:
                targets_to_train.append(target_arg)
            if args.include_extremes:
                targets_to_train.extend(extremes_targets)
            # Deduplicate preserving order
            seen = set(); ordered_targets = []
            for t in targets_to_train:
                if t in available_cols and t not in seen:
                    ordered_targets.append(t); seen.add(t)

            if not ordered_targets:
                print(f"⚠️  No valid targets resolved for timeframe {tf}. Available sample labels: {direction_targets + extremes_targets}")
                continue

            # Collect tasks for each target and model
            for tgt in ordered_targets:
                if model_name:
                    # Single model specified
                    training_tasks.append((tf, args.config, args.overwrite, tgt, model_name))
                else:
                    # Train all models
                    for candidate in ['lightgbm','xgboost','catboost']:
                        training_tasks.append((tf, args.config, args.overwrite, tgt, candidate))

        if not training_tasks:
            print("No training tasks to execute.")
            sys.exit(1)

        print(f"\nSubmitting {len(training_tasks)} training tasks to Ray cluster...")
        
        # Submit all tasks to Ray
        futures = [train_single_remote.remote(*task) for task in training_tasks]
        
        # Collect results
        results = ray.get(futures)
        
        # Process results
        overall_ok = True
        for result in results:
            tf, tgt, mdl, res = result
            if res:
                model_name_display, metric_value = res
                metric_str = f"{metric_value:.4f}" if 'RMSE' not in str(metric_value) else f"RMSE={metric_value:.6f}"
                print(f"✅ {model_name_display} trained (tf={tf}, target={tgt}) -> {metric_str}")
            else:
                print(f"❌ Training failed (tf={tf}, target={tgt}, model={mdl})")
                overall_ok = False

        if not overall_ok:
            sys.exit(1)
        else:
            print("\n🎉 All training tasks completed successfully!")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"❌ Training failed: {str(e)}")
        sys.exit(1)
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">/home/donvirtus/crypto/bare_metal/trade_predictor/scripts/train_tree_model_ray.py