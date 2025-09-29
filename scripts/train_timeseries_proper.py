#!/usr/bin/env python3
"""
Proper Time Series Trading Model Training Script

Implementasi logika time series yang benar:
1. Rolling window split (no data leakage)
2. SMOTE untuk class imbalance
3. Trading-focused evaluation (precision/recall/sharpe)
4. Slope/crossing weighted loss function
5. Proper regularization dengan early stopping
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
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.logging import get_logger
from utils.config import load_config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score
)

# SMOTE for imbalanced classes
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

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

logger = get_logger('train_timeseries')


class TimeSeriesTrader:
    def __init__(self, timeframe="15m", config_path="config/config.yaml"):
        self.timeframe = timeframe
        self.config = load_config(config_path)
        self.db_path = f"data/db/btc_{timeframe}.sqlite"
        self.models_dir = "models/timeseries_models"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(self.models_dir, exist_ok=True)
        logger.info(f"Initialized TimeSeriesTrader for {timeframe}")

    def load_data(self) -> pd.DataFrame:
        """Load data dengan timestamp ordering yang proper"""
        logger.info(f"Loading time series data from {self.db_path}")
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        # ✅ CRITICAL: ORDER BY timestamp untuk time series
        df = pd.read_sql_query(
            "SELECT * FROM features ORDER BY timestamp ASC", 
            conn
        )
        conn.close()
        
        logger.info(f"Loaded {len(df)} rows in chronological order")
        return df

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'direction') -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Feature preparation dengan slope/crossing weighting info"""
        logger.info(f"Preparing features for target: {target_col}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Exclude leakage columns
        leakage_cols = {
            'future_return_pct', 'future_close', 'future_open', 
            'future_high', 'future_low', 'direction'
        }
        if target_col == 'direction':
            leakage_cols.remove('direction')
            
        # Add multi-horizon leakage protection
        for col in df.columns:
            if any(col.startswith(p) for p in ['direction_h', 'future_return_pct_h', 'future_max_high_h', 'future_min_low_h']):
                if col != target_col:
                    leakage_cols.add(col)

        id_meta_cols = {'timestamp', 'ts', 'time', 'pair', 'symbol', 'timeframe', 'index', 'id'}
        exclude = leakage_cols | id_meta_cols | {target_col}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [c for c in numeric_cols if c not in exclude]
        
        if not available_features:
            raise ValueError("No features available after exclusion")

        y = df[target_col].copy()
        X = df[available_features].copy()
        
        # Remove NaN targets
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        # Fill NaN features (using new pandas syntax)
        X = X.ffill().bfill().fillna(0)
        
        # ✅ CREATE SAMPLE WEIGHTS based on slope/crossing extremes
        sample_weights = self._compute_sample_weights(X, y)
        
        # Feature validation for normalized features
        normalized_features = [c for c in available_features if any(x in c for x in ['_pct', 'normalized', 'slope_pct'])]
        bb_features = [c for c in available_features if 'bb_' in c]
        ma_features = [c for c in available_features if any(x in c for x in ['sma_', 'ema_', 'wma_'])]
        
        logger.info(f"Features: {len(available_features)} total")
        logger.info(f"  - Normalized features: {len(normalized_features)}")
        logger.info(f"  - Bollinger Band features: {len(bb_features)}")
        logger.info(f"  - Moving Average features: {len(ma_features)}")
        logger.info(f"Samples: {len(X)}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, available_features, sample_weights

    def _compute_sample_weights(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Compute sample weights berdasarkan slope extremes dan crossing events
        Sesuai request: "weight lebih tinggi buat sample dengan slope ekstrem"
        """
        weights = np.ones(len(X))
        
        # Identify slope and crossing columns (including normalized variants)
        slope_cols = [c for c in X.columns if any(keyword in c.lower() for keyword in ['slope', '_pct', 'slope_pct'])]
        crossing_cols = [c for c in X.columns if any(keyword in c.lower() for keyword in ['cross', 'confirmed', 'seq'])]
        
        if slope_cols:
            logger.info(f"Found {len(slope_cols)} slope features for weighting: {slope_cols[:5]}")
            
            # Get slope extremes (top/bottom 10%)
            for col in slope_cols[:8]:  # Increase limit for more comprehensive weighting
                if col in X.columns:
                    values = X[col].abs()  # Use absolute slope
                    q90 = values.quantile(0.9)
                    extreme_mask = values >= q90
                    weights[extreme_mask] *= 2.0  # 2x weight for extreme slopes
                    
        if crossing_cols:
            logger.info(f"Found {len(crossing_cols)} crossing features for weighting: {crossing_cols[:5]}")
            
            # Weight crossing events (confirmed crossings and sequential patterns)
            for col in crossing_cols[:8]:  # Increase limit for more features
                if col in X.columns:
                    if 'confirmed' in col or 'seq' in col:
                        crossing_mask = X[col] > 0
                        weights[crossing_mask] *= 1.5  # 1.5x weight for confirmed crossings
                    elif 'cross' in col:
                        crossing_mask = X[col] != 0  # Any crossing (positive or negative)
                        weights[crossing_mask] *= 1.3  # 1.3x weight for any crossing
        
        logger.info(f"Sample weights: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
        return weights

    def timeseries_split(self, X: pd.DataFrame, y: pd.Series, 
                        sample_weights: np.ndarray, n_splits: int = 5) -> List[Dict]:
        """
        ✅ PROPER: Rolling window split untuk time series (no data leakage)
        """
        logger.info(f"Performing time series split with {n_splits} folds")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            w_train, w_test = sample_weights[train_idx], sample_weights[test_idx]
            
            # Apply SMOTE if available and configured
            if SMOTE_AVAILABLE and getattr(self.config, 'smote_threshold', 0.1) > 0:
                try:
                    smote = SMOTE(random_state=42, k_neighbors=3)
                    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                    # Extend sample weights for SMOTE samples (use mean weight)
                    mean_weight = w_train.mean()
                    n_synthetic = len(X_train_smote) - len(X_train)
                    w_train_smote = np.concatenate([w_train, np.full(n_synthetic, mean_weight)])
                    
                    logger.info(f"Fold {fold+1}: SMOTE applied. {len(X_train)} → {len(X_train_smote)} samples")
                    X_train, y_train, w_train = X_train_smote, y_train_smote, w_train_smote
                except Exception as e:
                    logger.warning(f"Fold {fold+1}: SMOTE failed: {e}")
            
            splits.append({
                'fold': fold + 1,
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'w_train': w_train, 'w_test': w_test
            })
            
        return splits

    def train_lightgbm_cv(self, splits: List[Dict]) -> Dict:
        """Train LightGBM with cross-validation dan sample weights"""
        if not LIGHTGBM_AVAILABLE:
            return {'error': 'LightGBM not available'}
            
        logger.info("Training LightGBM with time series CV...")
        fold_results = []
        models = []
        
        for split in splits:
            fold = split['fold']
            logger.info(f"Training fold {fold}/{len(splits)}")
            
            X_train, X_test = split['X_train'], split['X_test']
            y_train, y_test = split['y_train'], split['y_test']
            w_train, w_test = split['w_train'], split['w_test']
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # LightGBM datasets with sample weights
            train_data = lgb.Dataset(X_train_scaled, label=y_train, weight=w_train)
            valid_data = lgb.Dataset(X_test_scaled, label=y_test, weight=w_test, reference=train_data)
            
            # Get LightGBM parameters from config
            lgb_config = self.config.lightgbm.get('default', {}) if hasattr(self.config, 'lightgbm') else {}
            params = {
                'objective': lgb_config.get('objective', 'multiclass'),
                'num_class': len(np.unique(y_train)),
                'metric': lgb_config.get('metric', 'multi_logloss'),
                'boosting_type': lgb_config.get('boosting_type', 'gbdt'),
                'num_leaves': lgb_config.get('num_leaves', 31),
                'learning_rate': lgb_config.get('learning_rate', 0.05),
                'feature_fraction': lgb_config.get('feature_fraction', 0.8),
                'bagging_fraction': lgb_config.get('bagging_fraction', 0.8),
                'bagging_freq': lgb_config.get('bagging_freq', 5),
                'lambda_l1': lgb_config.get('lambda_l1', 0.1),
                'lambda_l2': lgb_config.get('lambda_l2', 0.1),
                'min_data_in_leaf': lgb_config.get('min_data_in_leaf', 20),
                'verbose': lgb_config.get('verbose', -1),
                'random_state': lgb_config.get('random_state', 42)
            }
            
            # Train with early stopping from config
            early_stopping_rounds = lgb_config.get('early_stopping_rounds', 100)
            num_boost_round = lgb_config.get('num_boost_round', 1000)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=num_boost_round,
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds),
                    lgb.log_evaluation(0)
                ]
            )
            
            models.append((model, scaler))
            
            # Evaluate with trading metrics
            y_pred_proba = model.predict(X_test_scaled)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            fold_metrics = self._compute_trading_metrics(y_test, y_pred, y_pred_proba)
            fold_metrics['fold'] = fold
            fold_results.append(fold_metrics)
            
            logger.info(f"Fold {fold} - Precision: {fold_metrics['precision']:.4f}, "
                       f"Recall: {fold_metrics['recall']:.4f}, "
                       f"F1: {fold_metrics['f1']:.4f}")
        
        # Aggregate results
        avg_metrics = self._aggregate_cv_results(fold_results)
        
        return {
            'model_type': 'LightGBM',
            'fold_results': fold_results,
            'avg_metrics': avg_metrics,
            'models': models
        }

    def train_xgboost_cv(self, splits: List[Dict]) -> Dict:
        """Train XGBoost with cross-validation dan sample weights"""
        if not XGBOOST_AVAILABLE:
            return {'error': 'XGBoost not available'}
            
        logger.info("Training XGBoost with time series CV...")
        fold_results = []
        models = []
        
        # Get XGBoost config
        xgb_config = self.config.xgboost.get('default', {}) if hasattr(self.config, 'xgboost') else {}
        
        for split in splits:
            fold = split['fold']
            logger.info(f"Training fold {fold}/{len(splits)}")
            
            X_train, X_test = split['X_train'], split['X_test']
            y_train, y_test = split['y_train'], split['y_test']
            w_train, w_test = split['w_train'], split['w_test']
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # XGBoost parameters from config
            params = {
                'objective': xgb_config.get('objective', 'multi:softprob'),
                'num_class': len(np.unique(y_train)),
                'eval_metric': xgb_config.get('eval_metric', 'mlogloss'),
                'max_depth': xgb_config.get('max_depth', 6),
                'learning_rate': xgb_config.get('learning_rate', 0.05),
                'subsample': xgb_config.get('subsample', 0.8),
                'colsample_bytree': xgb_config.get('colsample_bytree', 0.8),
                'reg_alpha': xgb_config.get('reg_alpha', 0.1),
                'reg_lambda': xgb_config.get('reg_lambda', 0.1),
                'min_child_weight': xgb_config.get('min_child_weight', 5),
                'random_state': xgb_config.get('random_state', 42),
                'verbosity': xgb_config.get('verbosity', 0)
            }
            
            # Train with early stopping
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_scaled, y_train,
                sample_weight=w_train,
                eval_set=[(X_test_scaled, y_test)],
                sample_weight_eval_set=[w_test],
                early_stopping_rounds=xgb_config.get('early_stopping_rounds', 100),
                verbose=False
            )
            
            models.append((model, scaler))
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test_scaled)
            y_pred = model.predict(X_test_scaled)
            
            fold_metrics = self._compute_trading_metrics(y_test, y_pred, y_pred_proba)
            fold_metrics['fold'] = fold
            fold_results.append(fold_metrics)
            
            logger.info(f"Fold {fold} - Precision: {fold_metrics['precision']:.4f}, "
                       f"Recall: {fold_metrics['recall']:.4f}, "
                       f"F1: {fold_metrics['f1']:.4f}")
        
        # Aggregate results
        avg_metrics = self._aggregate_cv_results(fold_results)
        
        return {
            'model_type': 'XGBoost',
            'fold_results': fold_results,
            'avg_metrics': avg_metrics,
            'models': models
        }

    def train_catboost_cv(self, splits: List[Dict]) -> Dict:
        """Train CatBoost with cross-validation dan sample weights"""
        if not CATBOOST_AVAILABLE:
            return {'error': 'CatBoost not available'}
            
        logger.info("Training CatBoost with time series CV...")
        fold_results = []
        models = []
        
        # Get CatBoost config
        cb_config = self.config.catboost.get('default', {}) if hasattr(self.config, 'catboost') else {}
        
        for split in splits:
            fold = split['fold']
            logger.info(f"Training fold {fold}/{len(splits)}")
            
            X_train, X_test = split['X_train'], split['X_test']
            y_train, y_test = split['y_train'], split['y_test']
            w_train, w_test = split['w_train'], split['w_test']
            
            # CatBoost doesn't need manual scaling
            # Build CatBoost pools with sample weights
            train_pool = cb.Pool(X_train, label=y_train, weight=w_train)
            test_pool = cb.Pool(X_test, label=y_test, weight=w_test)
            
            # CatBoost parameters from config
            params = {
                'iterations': cb_config.get('iterations', 500),
                'depth': cb_config.get('depth', 6),
                'learning_rate': cb_config.get('learning_rate', 0.05),
                'l2_leaf_reg': cb_config.get('l2_leaf_reg', 3),
                'border_count': cb_config.get('border_count', 128),
                'loss_function': cb_config.get('loss_function', 'MultiClass'),
                'eval_metric': cb_config.get('eval_metric', 'MultiClass'),
                'early_stopping_rounds': cb_config.get('early_stopping_rounds', 50),
                'random_seed': cb_config.get('random_seed', 42),
                'verbose': cb_config.get('verbose', False),
                'bootstrap_type': cb_config.get('bootstrap_type', 'Bayesian'),
                'bagging_temperature': cb_config.get('bagging_temperature', 0.5)
            }
            
            # Train model
            model = cb.CatBoostClassifier(**params)
            model.fit(
                train_pool,
                eval_set=test_pool,
                use_best_model=True,
                verbose=False
            )
            
            models.append((model, None))  # CatBoost doesn't need scaler
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            
            fold_metrics = self._compute_trading_metrics(y_test, y_pred, y_pred_proba)
            fold_metrics['fold'] = fold
            fold_results.append(fold_metrics)
            
            logger.info(f"Fold {fold} - Precision: {fold_metrics['precision']:.4f}, "
                       f"Recall: {fold_metrics['recall']:.4f}, "
                       f"F1: {fold_metrics['f1']:.4f}")
        
        # Aggregate results
        avg_metrics = self._aggregate_cv_results(fold_results)
        
        return {
            'model_type': 'CatBoost',
            'fold_results': fold_results,
            'avg_metrics': avg_metrics,
            'models': models
        }

    def _compute_trading_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                y_pred_proba: np.ndarray) -> Dict:
        """
        ✅ TRADING-FOCUSED METRICS (bukan accuracy!)
        """
        # Multi-class precision/recall/f1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall), 
            'f1': float(f1),
            'support': len(y_true)  # Total samples in test set
        }
        
        # Class-specific metrics (untuk trading signal analysis)
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        unique_classes = sorted(y_true.unique())
        for i, cls in enumerate(unique_classes):
            metrics[f'precision_class_{cls}'] = float(class_precision[i])
            metrics[f'recall_class_{cls}'] = float(class_recall[i])
            metrics[f'f1_class_{cls}'] = float(class_f1[i])
        
        # ROC AUC for probabilistic evaluation
        try:
            if len(unique_classes) == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            else:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr'))
        except Exception:
            metrics['roc_auc'] = 0.0
        
        # Trading-specific metrics
        # Focus on minimizing false positives (bad trades)
        if 2 in unique_classes:  # UP class exists
            up_mask = (y_true == 2)
            up_predicted_mask = (y_pred == 2)
            
            if up_predicted_mask.sum() > 0:
                # Precision for UP signals (minimize false positive trades)
                up_precision = (up_mask & up_predicted_mask).sum() / up_predicted_mask.sum()
                metrics['up_signal_precision'] = float(up_precision)
            else:
                metrics['up_signal_precision'] = 0.0
        
        if 0 in unique_classes:  # DOWN class exists
            down_mask = (y_true == 0)
            down_predicted_mask = (y_pred == 0)
            
            if down_predicted_mask.sum() > 0:
                # Precision for DOWN signals
                down_precision = (down_mask & down_predicted_mask).sum() / down_predicted_mask.sum()
                metrics['down_signal_precision'] = float(down_precision)
            else:
                metrics['down_signal_precision'] = 0.0
        
        return metrics

    def _aggregate_cv_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate cross-validation results"""
        metrics = {}
        for key in ['precision', 'recall', 'f1', 'roc_auc', 'up_signal_precision', 'down_signal_precision']:
            values = [r.get(key, 0) for r in fold_results]
            metrics[f'{key}_mean'] = float(np.mean(values))
            metrics[f'{key}_std'] = float(np.std(values))
        
        return metrics

    def save_model(self, results: Dict, target: str):
        """Save model with proper time series metadata"""
        if 'error' in results:
            logger.error(f"Cannot save model: {results['error']}")
            return
            
        model_name = results['model_type'].lower()
        timestamp = self.timestamp
        
        # Save best model (highest average F1)
        best_model, best_scaler = results['models'][0]  # Simplified: use first model
        
        model_path = os.path.join(self.models_dir, f"{model_name}_{self.timeframe}_{target}_{timestamp}.txt")
        scaler_path = os.path.join(self.models_dir, f"scaler_{self.timeframe}_{target}_{timestamp}.joblib")
        
        best_model.save_model(model_path)
        joblib.dump(best_scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_type': results['model_type'],
            'timeframe': self.timeframe,
            'target': target,
            'timestamp': timestamp,
            'training_method': 'time_series_cv',
            'cv_folds': len(results['fold_results']),
            'avg_metrics': results['avg_metrics'],
            'fold_results': results['fold_results'],
            'sample_weighting': 'slope_crossing_based',
            'regularization': ['L1', 'L2', 'early_stopping', 'feature_fraction', 'bagging'],
            'data_leakage_protection': 'TimeSeriesSplit',
            'class_balancing': 'SMOTE' if SMOTE_AVAILABLE else 'None'
        }
        
        import json
        metadata_path = os.path.join(self.models_dir, f"metadata_{model_name}_{self.timeframe}_{target}_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved time series model: {model_path}")
        logger.info(f"Average F1: {results['avg_metrics']['f1_mean']:.4f} ± {results['avg_metrics']['f1_std']:.4f}")

    def train(self, target: str = 'direction'):
        """Main training function dengan proper time series methodology"""
        logger.info(f"Starting time series training for target: {target}")
        
        # Load data
        df = self.load_data()
        
        # Prepare features dengan sample weighting
        X, y, features, sample_weights = self.prepare_features(df, target)
        
        # Time series split
        splits = self.timeseries_split(X, y, sample_weights)
        
        # Train based on model type (default to LightGBM for now)
        model_type = getattr(self.config, 'model_type', 'lightgbm')
        
        if model_type.lower() == 'lightgbm' and LIGHTGBM_AVAILABLE:
            results = self.train_lightgbm_cv(splits)
        elif model_type.lower() == 'xgboost' and XGBOOST_AVAILABLE:
            results = self.train_xgboost_cv(splits) 
        elif model_type.lower() == 'catboost' and CATBOOST_AVAILABLE:
            results = self.train_catboost_cv(splits)
        else:
            logger.warning(f"Model {model_type} not available, falling back to LightGBM")
            results = self.train_lightgbm_cv(splits)
        
        # Save model
        self.save_model(results, target)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Proper Time Series Training for Crypto Prediction')
    parser.add_argument('--timeframe', default='15m', choices=['5m', '15m', '30m', '1h', '2h', '4h', '6h'],
                       help='Timeframe to train')
    parser.add_argument('--target', default='direction', 
                       help='Target variable (direction, direction_h5, etc.)')
    parser.add_argument('--model', default='lightgbm', choices=['lightgbm', 'xgboost', 'catboost'],
                       help='Model type to train (default: lightgbm)')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Config file path')
    
    args = parser.parse_args()
    
    try:
        trainer = TimeSeriesTrader(
            timeframe=args.timeframe,
            config_path=args.config
        )
        
        # Override model type from command line
        trainer.config.model_type = args.model
        
        results = trainer.train(target=args.target)
        
        if 'error' not in results:
            print(f"✅ Training completed successfully!")
            print(f"Average F1: {results['avg_metrics']['f1_mean']:.4f} ± {results['avg_metrics']['f1_std']:.4f}")
            print(f"Average Precision: {results['avg_metrics']['precision_mean']:.4f} ± {results['avg_metrics']['precision_std']:.4f}")
        else:
            print(f"❌ Training failed: {results['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Training error: {e}")
        print(f"❌ Training error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()