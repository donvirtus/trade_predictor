#!/usr/bin/env python3
"""
Universal training script for all timeframes with flexible MA configuration
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
    def __init__(self, timeframe="15m", models_dir="models/tree_models", config_path="config/config.yaml"):
        self.timeframe = timeframe
        self.models_dir = models_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = load_config(config_path)
        
        # Determine database path based on timeframe
        timeframe_map = {"5m": "5m", "15m": "15m", "1h": "1h"}
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
        """Prepare features based on configuration"""
        logger.info("Preparing features based on configuration...")
        
        # Get MA configuration
        enable_ma = getattr(self.config, 'enable_ma_features', True)
        ma_periods = getattr(self.config, 'ma_periods', [])
        
        logger.info(f"MA Features enabled: {enable_ma}")
        logger.info(f"MA Periods: {ma_periods}")
        
        # Define base features (always included)
        base_features = []
        
        # Bollinger Bands features
        bb_periods = getattr(self.config, 'bb_periods', [48, 96])
        for period in bb_periods:
            base_features.append(f'bb_{period}_middle')
            for dev in getattr(self.config, 'bb_devs', [1.0, 2.0]):
                base_features.extend([
                    f'bb_{period}_upper_{dev}',
                    f'bb_{period}_lower_{dev}',
                    f'bb_{period}_percent_b_{dev}',
                    f'bb_{period}_bandwidth_{dev}'
                ])
            base_features.append(f'bb_{period}_squeeze_flag')
        
        # MA features (conditional)
        ma_features = []
        if enable_ma and ma_periods:
            for period in ma_periods:
                ma_features.extend([
                    f'ma_{period}',
                    f'ma_{period}_lag_2',
                    f'close_to_ma_{period}'
                ])
        
        # Other technical indicators
        other_features = [
            'price_range_5', 'volatility_5', 'rsi_14', 'macd', 'macd_signal', 
            'macd_histogram', 'adx_14', 'obv', 'vwap', 'close_lag_2',
            'rsi_14_lag_2', 'close_to_vwap', 'macd_diff'
        ]
        
        # Add BB-related features
        for period in bb_periods:
            other_features.append(f'close_to_bb_{period}_middle')
        
        # Combine all features and remove duplicates while preserving order
        all_features = []
        for feature_list in [base_features, ma_features, other_features]:
            for feature in feature_list:
                if feature not in all_features:
                    all_features.append(feature)
        
        # Filter features that actually exist in the dataframe
        available_features = [f for f in all_features if f in df.columns]
        missing_features = [f for f in all_features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features in database: {missing_features}")
        
        logger.info(f"Using {len(available_features)} features")
        logger.info(f"Features: {available_features}")
        
        # Prepare feature matrix
        X = df[available_features].copy()
        
        # Handle target variable - try different column names
        target_column = None
        possible_targets = ['target', 'direction', 'label']
        for col in possible_targets:
            if col in df.columns:
                target_column = col
                break
        
        if target_column is None:
            raise ValueError(f"No target column found in database. Tried: {possible_targets}")
        
        y = df[target_column].copy()
        logger.info(f"Using target column: {target_column}")
        
        # Remove rows with missing target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        # Handle missing values in features
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
        logger.info(f"Target distribution:\n{y.value_counts().sort_index()}")
        
        return X, y, available_features

    def train_lightgbm(self, X_train, X_test, y_train, y_test):
        """Train LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available")
            return None, 0.0
        
        logger.info("Training LightGBM model...")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Parameters
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
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Evaluate
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
        
        model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='mlogloss',
            early_stopping_rounds=100
        )
        
        # Train
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        
        logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
        return model, accuracy

    def train_catboost(self, X_train, X_test, y_train, y_test):
        """Train CatBoost model"""
        if not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not available")
            return None, 0.0
        
        logger.info("Training CatBoost model...")
        
        model = cb.CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            loss_function='MultiClass',
            eval_metric='Accuracy',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=100
        )
        
        # Train
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            use_best_model=True
        )
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        
        logger.info(f"CatBoost Accuracy: {accuracy:.4f}")
        return model, accuracy

    def save_model(self, model, model_type, accuracy, scaler, features):
        """Save model and associated files with secure formats"""
        
        # Create secure model paths based on model type
        timestamp = self.timestamp
        
        if model_type == 'LightGBM':
            model_filename = f"btc_{self.timeframe}_lightgbm_{timestamp}.txt"
            model_path = os.path.join(self.models_dir, model_filename)
            model.save_model(model_path)
        elif model_type == 'XGBoost':
            model_filename = f"btc_{self.timeframe}_xgboost_{timestamp}.json"
            model_path = os.path.join(self.models_dir, model_filename)
            model.save_model(model_path)
        elif model_type == 'CatBoost':
            model_filename = f"btc_{self.timeframe}_catboost_{timestamp}.cbm"
            model_path = os.path.join(self.models_dir, model_filename)
            model.save_model(model_path)
        else:
            # Fallback to joblib for unknown models
            model_filename = f"btc_{self.timeframe}_{model_type.lower()}_{timestamp}.joblib"
            model_path = os.path.join(self.models_dir, model_filename)
            joblib.dump(model, model_path)
        
        # Save scaler with joblib (more secure than pickle)
        scaler_filename = f"btc_{self.timeframe}_scaler_{timestamp}.joblib"
        scaler_path = os.path.join(self.models_dir, scaler_filename)
        joblib.dump(scaler, scaler_path)
        
        # Save features list
        features_filename = f"btc_{self.timeframe}_{model_type.lower()}_features.txt"
        features_path = os.path.join(self.models_dir, features_filename)
        with open(features_path, 'w') as f:
            f.write('\n'.join(features))
        
        # Save metadata as JSON in metadata folder
        metadata_filename = f"btc_{self.timeframe}_{model_type.lower()}_metadata.json"
        metadata_path = os.path.join("models/metadata", metadata_filename)
        
        metadata = {
            "model_type": model_type,
            "timeframe": self.timeframe,
            "training_timestamp": timestamp,
            "accuracy": float(accuracy),
            "features_count": len(features),
            "ma_features_enabled": getattr(self.config, 'enable_ma_features', True),
            "ma_periods": getattr(self.config, 'ma_periods', []),
            "bb_periods": getattr(self.config, 'bb_periods', []),
            "model_file": model_filename,
            "scaler_file": scaler_filename,
            "features_file": features_filename
        }
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {model_type} model: {model_path}")
        logger.info(f"Saved scaler: {scaler_path}")
        logger.info(f"Saved features: {features_path}")
        logger.info(f"Saved metadata: {metadata_path}")

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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
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

def main():
    parser = argparse.ArgumentParser(description='Universal Model Training for Crypto Prediction')
    parser.add_argument('--timeframe', choices=['5m', '15m', '1h'], default='15m',
                       help='Timeframe to train models for')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = UniversalModelTrainer(
            timeframe=args.timeframe,
            config_path=args.config
        )
        
        # Train all models
        results = trainer.train_all_models()
        
        if results:
            print(f"\n✅ Training completed successfully for {args.timeframe} timeframe!")
            print("Model Performance:")
            for model_name, accuracy in results.items():
                print(f"  {model_name}: {accuracy:.2%}")
        else:
            print("❌ Training failed - no models were successfully trained")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"❌ Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()