#!/usr/bin/env python3
"""
Enhanced Live Prediction with Bollinger Bands Analysis
=====================================================

Features:
 - Multi-timeframe support: 5m, 15m, 30m, 1h, 2h, 4h, 6h
 - Automatic model selection (--model auto) prefers LightGBM > XGBoost > CatBoost
 - Historical context loading (--history-rows) for slope / volatility assessment
 - Confidence override (--confidence-threshold) forces HOLD below threshold
 - JSON output mode (--output json) for integration / piping (line-delimited JSON)
 - Feature integrity check (warn if training features missing; caution if >10%)
 - Rich terminal visualization: Bollinger Bands, risk/reward, recommendations
"""

import os
import sys
import time
import sqlite3
import argparse
import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
import joblib
warnings.filterwarnings('ignore')
import csv
from pathlib import Path

# Restore internal project utilities
from utils.config import load_config
from utils.logging import get_logger

# =============================================================
# Utility / Indicator Integration Functions (restored)
# =============================================================

def integrate_bb_signals(bb_analysis, slope_analysis=None):
    """Integrate Bollinger Band signals with weighting.

    Returns dict: {integrated_signal, confidence, score}
    """
    signal_weights = {
        'bb_96': 0.5,
        'bb_48': 0.3,
        'bb_24': 0.2
    }
    if 'bb_24' not in bb_analysis:
        signal_weights['bb_96'] = 0.6
        signal_weights['bb_48'] = 0.4

    signal_scores = {
        'STRONG BUY': 2,
        'BUY': 1,
        'HOLD': 0,
        'NEUTRAL': 0,
        'CAUTION': 0,
        'SELL': -1,
        'STRONG SELL': -2
    }

    weighted_score = 0
    total_weight = 0
    for period, weight in signal_weights.items():
        if period in bb_analysis:
            sig = bb_analysis[period].get('signal', 'HOLD')
            score = signal_scores.get(sig, 0)
            weighted_score += score * weight
            total_weight += weight

    normalized_score = weighted_score / total_weight if total_weight else 0

    if slope_analysis and 'bb_48' in slope_analysis:
        trend = slope_analysis['bb_48'].get('trend', 'FLAT')
        if trend == 'UPTREND' and normalized_score > 0:
            normalized_score = min(1.0, normalized_score * 1.2)
        elif trend == 'DOWNTREND' and normalized_score < 0:
            normalized_score = max(-1.0, normalized_score * 1.2)

    if normalized_score >= 1.5:
        integrated_signal = 'STRONG BUY'
    elif normalized_score >= 0.5:
        integrated_signal = 'BUY'
    elif normalized_score > -0.5:
        integrated_signal = 'NEUTRAL'
    elif normalized_score > -1.5:
        integrated_signal = 'SELL'
    else:
        integrated_signal = 'STRONG SELL'

    confidence = min(0.95, 0.5 + abs(normalized_score) * 0.25)

    return {
        'integrated_signal': integrated_signal,
        'confidence': confidence,
        'score': normalized_score
    }

def resolve_signal_conflict(bb_signal, ml_prediction, ml_confidence, bb_confidence):
    """
    Resolve conflicts between ML predictions and BB signals
    
    Args:
        bb_signal: Integrated BB signal (STRONG BUY, BUY, NEUTRAL, SELL, STRONG SELL)
        ml_prediction: ML model prediction (0=SELL, 1=HOLD, 2=BUY)
        ml_confidence: ML prediction confidence (0-1)
        bb_confidence: BB signal confidence (0-1)
    
    Returns:
        Dictionary with resolved signal, reasoning, and action
    """
    ml_signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    ml_signal = ml_signal_map.get(ml_prediction, "UNKNOWN")
    
    # Check for extreme oversold/overbought conditions with signal conflicts
    extreme_condition = False
    if (bb_signal in ['STRONG BUY'] and ml_signal == 'SELL') or \
       (bb_signal in ['STRONG SELL'] and ml_signal == 'BUY'):
        extreme_condition = True
    
    # Calculate weighted decision based on confidence levels
    if extreme_condition and ml_confidence < 0.85:
        # In extreme conditions, trust BB more unless ML is very confident
        bb_weight = 0.7
        ml_weight = 0.3
        reasoning = f"Extreme market condition: {bb_signal}, trusting BB signals"
    else:
        # Normal weighting - ML has higher weight due to trained nature
        if ml_confidence > 0.8:
            ml_weight = 0.8
            bb_weight = 0.2
        elif ml_confidence > 0.6:
            ml_weight = 0.7
            bb_weight = 0.3
        else:
            ml_weight = 0.6
            bb_weight = 0.4
        
        reasoning = f"ML confidence ({ml_confidence:.0%}) {ml_weight*100:.0f}% weight, " \
                   f"BB ({bb_confidence:.0%}) {bb_weight*100:.0f}% weight"
    
    # Check for signal conflicts
    conflict = False
    if (bb_signal in ['BUY', 'STRONG BUY'] and ml_signal == 'SELL') or \
       (bb_signal in ['SELL', 'STRONG SELL'] and ml_signal == 'BUY'):
        conflict = True
        reasoning += " | WARNING: SIGNAL CONFLICT - Use caution"
    
    # Determine final resolved signal based on weighted confidences
    signal_scores = {
        'STRONG BUY': 2,
        'BUY': 1,
        'NEUTRAL': 0,
        'HOLD': 0,
        'SELL': -1,
        'STRONG SELL': -2
    }
    
    ml_score_map = {0: -1, 1: 0, 2: 1}  # SELL=-1, HOLD=0, BUY=1
    bb_score = signal_scores.get(bb_signal, 0)
    ml_score = ml_score_map.get(ml_prediction, 0)
    
    weighted_score = (bb_score * bb_weight) + (ml_score * ml_weight)
    
    # Convert weighted score back to signal
    final_signal = ""
    position = ""
    if weighted_score >= 0.7:
        final_signal = "BUY"
        position = "LONG"
    elif weighted_score >= 0.2:
        final_signal = "WEAK BUY"
        position = "LONG"
    elif weighted_score <= -0.7:
        final_signal = "SELL"
        position = "SHORT"
    elif weighted_score <= -0.2:
        final_signal = "WEAK SELL"
        position = "SHORT"
    else:
        final_signal = "NEUTRAL"
        position = "WAIT"
    
    # Override recommendation in extreme conditions with conflicting signals
    if conflict and extreme_condition and ml_confidence < 0.9:
        if bb_signal in ["STRONG BUY", "BUY"]:
            final_signal = "NEUTRAL LEANING BUY"
            position = "WAIT"
            reasoning += " | Mixed signals in extreme condition: Consider waiting"
        elif bb_signal in ["STRONG SELL", "SELL"]:
            final_signal = "NEUTRAL LEANING SELL"
            position = "WAIT"
            reasoning += " | Mixed signals in extreme condition: Consider waiting"
    
    return {
        'signal': final_signal,
        'position': position,
        'ml_weight': ml_weight,
        'bb_weight': bb_weight,
        'reasoning': reasoning,
        'conflict': conflict,
        'weighted_score': weighted_score
    }

# ================= Additional Restored Helper Functions ==============
def add_trading_outputs(df, live_price, ml_prediction, ml_confidence):
    """Reconstructed simplified trading analytics helper."""
    try:
        ml_signal = "SELL" if ml_prediction == 0 else "BUY" if ml_prediction == 2 else "HOLD"
        entry = live_price
        if ml_signal == 'SELL':
            take_profit = entry * 0.985
            stop_loss = entry * 1.01
        elif ml_signal == 'BUY':
            take_profit = entry * 1.015
            stop_loss = entry * 0.99
        else:
            take_profit = entry
            stop_loss = entry
        rr = abs((take_profit - entry) / max(1e-9, (entry - stop_loss))) if entry != stop_loss else 1.0
        expected_return = (take_profit - entry) / entry if ml_signal == 'BUY' else (entry - take_profit) / entry if ml_signal == 'SELL' else 0.0
        return {
            'risk_reward_ratio': rr,
            'expected_return': expected_return * 100,
            'next_candle_prediction': 1.0 if ml_signal == 'BUY' else -1.0 if ml_signal == 'SELL' else 0.0,
            'trend_strength': {'value': 25, 'state': 'Moderate'},
            'volume_confirmation': {'ratio': 1.0, 'state': 'Moderate'},
            'historical_performance': {'win_rate': 60, 'avg_profit': 1.2},
            'alerts': []
        }
    except Exception:
        return {
            'risk_reward_ratio': 1.0,
            'expected_return': 0.0,
            'next_candle_prediction': 0.0,
            'trend_strength': {'value': 0, 'state': 'Unknown'},
            'volume_confirmation': {'ratio': 1.0, 'state': 'Unknown'},
            'historical_performance': {'win_rate': 0, 'avg_profit': 0},
            'alerts': ['analytics_error']
        }

def analyze_bb_slope(df, period=48, window=5):
    """Simplified slope analysis restoration."""
    try:
        if 'close' not in df.columns or len(df) < period + window:
            return {'trend': 'FLAT', 'slope': 0, 'bandwidth_trend': 'STABLE'}
        mb = df['close'].rolling(period).mean()
        recent = mb.dropna().tail(window)
        if len(recent) < window:
            return {'trend': 'FLAT', 'slope': 0, 'bandwidth_trend': 'STABLE'}
        slope = (recent.iloc[-1] - recent.iloc[0]) / (window - 1)
        trend = 'UPTREND' if slope > 0 else 'DOWNTREND' if slope < 0 else 'FLAT'
        return {'trend': trend, 'slope': float(slope), 'bandwidth_trend': 'STABLE'}
    except Exception:
        return {'trend': 'FLAT', 'slope': 0, 'bandwidth_trend': 'STABLE'}

class EnhancedPredictionSystem:
    """Enhanced prediction with live price and Bollinger Bands analysis"""
    
    def __init__(self, timeframe: str = "15m", model_type: str = "XGBoost", config_path: str = None):
        self.console = Console()
        self.timeframe = timeframe
        self.model_type = model_type
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = get_logger(f'enhanced_prediction_{timeframe}')
        
        # Model and data storage
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.prediction_db_path = f"data/db/enhanced_predictions_{timeframe}.sqlite"
        
        # Pasang market & simbol default dari config, dengan fallback aman
        try:
            cfg_data = getattr(self.config, 'data', {}) or {}
            self.market_type = (cfg_data.get('market_type') if isinstance(cfg_data, dict) else getattr(cfg_data, 'market_type', 'spot')) or 'spot'
            # Simbol prioritas: data.symbol lalu pairs[0]
            sym = (cfg_data.get('symbol') if isinstance(cfg_data, dict) else getattr(cfg_data, 'symbol', None))
            if not sym:
                pairs = getattr(self.config, 'pairs', None)
                if isinstance(pairs, list) and pairs:
                    sym = pairs[0]
            if not sym:
                sym = 'BTCUSDT'
        except Exception:
            self.market_type = 'spot'
            sym = 'BTCUSDT'

        # Pastikan pairing USDT (paksa kalau bukan USDT)
        if not str(sym).upper().endswith('USDT'):
            base = ''.join([c for c in str(sym).upper() if not c.isdigit()])
            if base.endswith('USDT'):
                sym = base
            else:
                # Heuristik sederhana: paksa BTCUSDT jika bukan USDT
                sym = 'BTCUSDT'
        self.symbol = str(sym).upper()

        # Binance API base: spot vs futures (UM)
        self._set_binance_api_base()
        
        # Check if TA-Lib is available
        try:
            import talib
            self.talib_available = True
            self.logger.info("TA-Lib is available for advanced indicators")
        except ImportError:
            self.talib_available = False
            self.logger.warning("TA-Lib is not available. Some advanced indicators will be disabled.")
            self.console.print("[yellow]⚠️  TA-Lib is not installed. Some advanced indicators will be disabled.[/yellow]")
            
        # Initialize prediction database
        self.init_prediction_db()
        
        # Load model (may auto-select)
        if not self.load_model():
            self.model = None
        # Coba load multi-model registry (tidak fatal jika gagal)
        try:
            self.load_multi_models()
        except Exception as e:
            self.logger.warning(f"Multi-model load skipped: {e}")
        # Paper-trade engine state holder
        self._paper_engine = None

    # ---------------------- Paper Trade Engine ----------------------
    class _PaperTradeEngine:
        def __init__(self, log_path: str, timeout_mins: int = 120):
            self.log_path = log_path
            self.timeout_mins = int(timeout_mins)
            if self.log_path:
                Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

        def _read_rows(self) -> list:
            if not self.log_path or not os.path.exists(self.log_path):
                return []
            try:
                import pandas as _pd
                return _pd.read_csv(self.log_path).to_dict('records')
            except Exception:
                rows = []
                with open(self.log_path, 'r', newline='') as f:
                    r = csv.DictReader(f)
                    for row in r:
                        rows.append(row)
                return rows

        def _append_row(self, row: dict):
            if not self.log_path:
                return
            file_exists = os.path.exists(self.log_path)
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        def load_open_trade(self) -> Optional[dict]:
            rows = self._read_rows()
            open_rows = [r for r in rows if str(r.get('event')) == 'OPEN']
            close_rows = [r for r in rows if str(r.get('event')) == 'CLOSE']
            closed_ids = {str(r.get('trade_id')) for r in close_rows}
            for r in reversed(open_rows):
                if str(r.get('trade_id')) not in closed_ids:
                    return r
            return None

        def open_trade(self, trade: dict):
            trade_row = {'event': 'OPEN', **trade}
            self._append_row(trade_row)

        def close_trade(self, trade_id: str, now_iso: str, close_price: float, outcome: str):
            rows = self._read_rows()
            entry = None
            for r in reversed(rows):
                if str(r.get('event')) == 'OPEN' and str(r.get('trade_id')) == str(trade_id):
                    entry = r; break
            pnl_pct = None
            direction = None
            if entry is not None:
                try:
                    entry_price = float(entry.get('entry_price'))
                    direction = str(entry.get('direction')).upper()
                    if direction == 'LONG':
                        pnl_pct = (close_price - entry_price) / entry_price * 100.0
                    elif direction == 'SHORT':
                        pnl_pct = (entry_price - close_price) / entry_price * 100.0
                except Exception:
                    pnl_pct = None
            close_row = {
                'event': 'CLOSE',
                'trade_id': trade_id,
                'close_time': now_iso,
                'close_price': close_price,
                'outcome': outcome,
                'pnl_pct': pnl_pct,
                'direction': direction
            }
            self._append_row(close_row)

    def _set_binance_api_base(self):
        """Atur base URL Binance sesuai market (spot/futures-usdm)."""
        mt = (self.market_type or 'spot').lower()
        if mt in ('future', 'futures', 'futures-usdm', 'usdm'):
            # USD-M Futures
            self.binance_api = "https://fapi.binance.com/fapi/v1"
        else:
            # Spot
            self.binance_api = "https://api.binance.com/api/v3"
        
    def load_model(self) -> bool:
        """Load the most recent trained model for the timeframe.

        If self.model_type == 'auto', tries LightGBM, then XGBoost, then CatBoost.
        """
        try:
            import glob, json, hashlib, joblib
            base_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            models_dir = os.path.join(base_models_dir, 'tree_models') if os.path.isdir(os.path.join(base_models_dir, 'tree_models')) else base_models_dir
            metadata_dir = os.path.join(base_models_dir, 'metadata')

            # Priority list
            if self.model_type.lower() == 'auto':
                priority = ['lightgbm', 'xgboost', 'catboost']
            else:
                priority = [self.model_type.lower()]

            def stable(mt):
                return {
                    'lightgbm': f"btc_{self.timeframe}_lightgbm.txt",
                    'xgboost': f"btc_{self.timeframe}_xgboost.json",
                    'catboost': f"btc_{self.timeframe}_catboost.cbm"
                }.get(mt)

            for mt in priority:
                candidates = []
                st = stable(mt)
                if st and os.path.exists(os.path.join(models_dir, st)):
                    candidates.append(os.path.join(models_dir, st))
                pattern = os.path.join(models_dir, f"btc_{self.timeframe}_{mt}_*")
                ts = sorted(glob.glob(pattern), reverse=True)
                for c in ts:
                    if c not in candidates:
                        candidates.append(c)
                if not candidates:
                    continue

                # Reorder: prefer direction / direction_h* before extremes future_max/min
                def pref_key(path):
                    name = os.path.basename(path)
                    if '_direction' in name and ('_max_high_' not in name and '_min_low_' not in name):
                        return (0, -os.path.getmtime(path))
                    if '_direction_h' in name:
                        return (1, -os.path.getmtime(path))
                    if '_max_high_' in name or '_min_low_' in name:
                        return (2, -os.path.getmtime(path))
                    return (3, -os.path.getmtime(path))
                candidates = sorted(candidates, key=pref_key)

                model_obj = None
                chosen = None
                for path in candidates:
                    # Skip clearly non-model auxiliary files (feature lists, metadata, logs)
                    lower_name = os.path.basename(path).lower()
                    if ('_features' in lower_name) or ('_metadata' in lower_name) or lower_name.endswith('.log'):
                        self.logger.debug(f"Skipping non-model file in candidates: {path}")
                        continue
                    # Basic extension filtering per model type
                    valid_ext = {
                        'lightgbm': ('.txt', '.model'),
                        'xgboost': ('.json', '.ubj', '.model'),
                        'catboost': ('.cbm',)
                    }.get(mt, ())
                    if valid_ext and not path.lower().endswith(valid_ext):
                        self.logger.debug(f"Skipping file with incompatible extension for {mt}: {path}")
                        continue
                    try:
                        if mt == 'lightgbm':
                            import lightgbm as lgb
                            model_obj = lgb.Booster(model_file=path)
                        elif mt == 'xgboost':
                            import xgboost as xgb
                            booster = xgb.Booster(); booster.load_model(path); model_obj = booster
                        elif mt == 'catboost':
                            import catboost as cb
                            model_obj = cb.CatBoostClassifier(); model_obj.load_model(path)
                        chosen = path
                        break
                    except Exception as e:
                        self.logger.warning(f"Model load failed {path}: {e}")
                if not model_obj:
                    continue

                # Scaler
                scaler = None
                stable_scaler = os.path.join(models_dir, f"btc_{self.timeframe}_scaler.joblib")
                if os.path.exists(stable_scaler):
                    try:
                        scaler = joblib.load(stable_scaler)
                    except Exception:
                        pass
                if scaler is None:
                    for sp in sorted(glob.glob(os.path.join(models_dir, f"btc_{self.timeframe}_scaler_*.joblib")), reverse=True):
                        try:
                            scaler = joblib.load(sp); break
                        except Exception:
                            continue

                # Features
                feat_path = os.path.join(models_dir, f"btc_{self.timeframe}_{mt}_features.txt")
                features = []
                if os.path.exists(feat_path):
                    try:
                        with open(feat_path, 'r') as f:
                            features = [ln.strip() for ln in f if ln.strip()]
                    except Exception:
                        pass

                # Metadata
                meta_path = os.path.join(metadata_dir, f"btc_{self.timeframe}_{mt}_metadata.json")
                meta = None
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as mf:
                            meta = json.load(mf)
                    except Exception:
                        pass

                if meta and features:
                    calc_hash = hashlib.sha256('\n'.join(features).encode('utf-8')).hexdigest()
                    if meta.get('features_hash') == calc_hash:
                        self.console.print(f"[green]🔐 Feature hash OK ({calc_hash[:10]}...)")
                    else:
                        self.console.print("[yellow]⚠️  Feature hash mismatch (metadata vs file)")

                self.model = model_obj
                self.scaler = scaler
                self.feature_columns = features if features else None
                self.model_type = mt.capitalize()

                if mt == 'lightgbm' and hasattr(self.model, 'feature_name') and self.feature_columns:
                    booster_feats = self.model.feature_name()
                    if booster_feats and set(booster_feats) != set(self.feature_columns):
                        self.console.print("[yellow]⚠️  Booster features differ; adopting booster ordering")
                        self.feature_columns = booster_feats

                self.console.print(f"[green]✅ Loaded {self.model_type} model for {self.timeframe}")
                self.console.print(f"[green]📁 Model: {os.path.basename(chosen) if chosen else 'unknown'} (selection order: stable -> newest timestamp)")
                self.console.print(f"[green]📁 Scaler: {'present' if self.scaler else 'None'}")
                self.console.print(f"[green]🔢 Features: {len(self.feature_columns) if self.feature_columns else 'Unknown'}")
                return True

            self.console.print("[red]❌ No available model found (stable or timestamped)")
            return False
        except Exception as e:
            self.console.print(f"[red]❌ Error loading model: {e}")
            return False

    # ================= Multi-Model (Direction + Extremes) Support =================
    def load_multi_models(self):
        """Memuat beberapa model per target (direction + extremes) berdasarkan registry.

        Struktur yang dihasilkan:
          self.multi_models = {
             'direction': { 'model': ..., 'scaler': ..., 'features': [...], 'meta': {...}},
             'future_max_high_h20': {...},
             'future_min_low_h20': {...},
          }
        Jika registry tidak ada, fallback ke model tunggal existing.
        """
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        metadata_dir = os.path.join(base_dir, 'metadata')
        models_dir = os.path.join(base_dir, 'tree_models')
        registry_path = os.path.join(metadata_dir, f'registry_lightgbm_{self.timeframe}.json')  # fokus LightGBM dulu
        self.multi_models = {}
        import json
        if not os.path.exists(registry_path):
            self.logger.info(f"Registry tidak ditemukan: {registry_path}. Lewati multi-model.")
            return False
        try:
            with open(registry_path, 'r') as f:
                reg = json.load(f)
        except Exception as e:
            self.logger.error(f"Gagal membaca registry: {e}")
            return False
        targets = reg.get('targets', [])
        if not targets:
            self.logger.warning("Registry kosong.")
            return False
        # Muat setiap target
        for t in targets:
            tgt_name = t.get('target_name')
            model_file = t.get('model_file')
            scaler_file = t.get('scaler_file')
            features_file = t.get('features_file')
            meta_file = t.get('metadata_file')
            try:
                model_path = os.path.join(models_dir, model_file)
                scaler_path = os.path.join(models_dir, scaler_file)
                features_path = os.path.join(models_dir, features_file)
                meta_path = os.path.join(metadata_dir, meta_file)
                if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path) and os.path.exists(meta_path)):
                    self.logger.warning(f"Lewati target {tgt_name} karena artefak tidak lengkap")
                    continue
                # Load artefak
                if model_file.endswith('.txt'):  # LightGBM
                    import lightgbm as lgb
                    model_obj = lgb.Booster(model_file=model_path)
                elif model_file.endswith('.json'):
                    import xgboost as xgb
                    model_obj = xgb.Booster(); model_obj.load_model(model_path)
                elif model_file.endswith('.cbm'):
                    import catboost as cb
                    model_obj = cb.CatBoost(); model_obj.load_model(model_path)
                else:
                    model_obj = joblib.load(model_path)
                scaler_obj = joblib.load(scaler_path)
                with open(features_path,'r') as ff:
                    feats = [ln.strip() for ln in ff if ln.strip()]
                with open(meta_path,'r') as mf:
                    meta = json.load(mf)
                self.multi_models[tgt_name] = {
                    'model': model_obj,
                    'scaler': scaler_obj,
                    'features': feats,
                    'meta': meta
                }
            except Exception as e:
                self.logger.error(f"Gagal load target {tgt_name}: {e}")
        if not self.multi_models:
            self.logger.warning("Tidak ada target berhasil dimuat dari registry.")
            return False
        self.logger.info(f"Multi-model dimuat: {list(self.multi_models.keys())}")
        return True

    def predict_multi(self, latest_df: pd.DataFrame) -> Dict[str, Any]:
        """Lakukan prediksi multi-target dan hitung R:R.

        Asumsi latest_df sudah punya kolom fitur lengkap; kita akan subset & scale per target.
        Return dict ringkas.
        """
        if not hasattr(self, 'multi_models') or not self.multi_models:
            return {'error': 'multi_models_not_loaded'}
        results = {}
        # Gunakan baris terakhir
        row = latest_df.tail(1)
        close_price = float(row['close'].iloc[0]) if 'close' in row.columns else np.nan
        for tgt, bundle in self.multi_models.items():
            feats = bundle['features']
            missing = [f for f in feats if f not in row.columns]
            if missing:
                self.logger.warning(f"Target {tgt}: fitur hilang {len(missing)} -> skip")
                continue
            X = row[feats].copy()
            # scale
            try:
                X_scaled = bundle['scaler'].transform(X)
            except Exception:
                # fallback jika scaler gagal
                X_scaled = X.values
            model_obj = bundle['model']
            meta = bundle['meta']
            task = meta.get('task_type','classification')
            pred_val = None
            pred_probs = None
            try:
                if task == 'classification':
                    if hasattr(model_obj, 'predict'):
                        raw = model_obj.predict(X_scaled)
                        arr = np.array(raw)
                        if arr.ndim == 1:
                            arr = arr.reshape(1,-1)
                        pred_class = int(np.argmax(arr[0]))
                        pred_val = pred_class
                        pred_probs = arr[0].tolist()
                else:  # regression
                    pred_val = float(model_obj.predict(X_scaled)[0]) if hasattr(model_obj,'predict') else None
            except Exception as e:
                self.logger.error(f"Infer gagal target {tgt}: {e}")
                continue
            results[tgt] = {
                'prediction': pred_val,
                'probs': pred_probs,
                'task': task
            }
        # Hitung R:R jika extremes tersedia
        rr = None; upside_pct = None; downside_pct = None
        max_key = next((k for k in results if k.startswith('future_max_high_h')), None)
        min_key = next((k for k in results if k.startswith('future_min_low_h')), None)
        if max_key and min_key and close_price==close_price:
            up_abs = results[max_key]['prediction']
            dn_abs = results[min_key]['prediction']
            upside_pct = up_abs * 100.0 if up_abs is not None else None
            downside_pct = abs(dn_abs) * 100.0 if dn_abs is not None else None
            if downside_pct and downside_pct > 0 and upside_pct is not None:
                rr = (upside_pct / downside_pct)
        # Arah utama pilih preferensi: direction_h5 > direction > direction_h1
        dir_key = None
        for cand in ['direction_h5','direction','direction_h1']:
            if cand in results:
                dir_key = cand; break
        direction_signal = None; direction_conf = None
        if dir_key:
            probs = results[dir_key].get('probs')
            if probs:
                direction_conf = float(max(probs))
                cls = int(np.argmax(probs))
                direction_signal = {0:'DOWN',1:'SIDEWAYS',2:'UP'}.get(cls, str(cls))
        return {
            'targets': results,
            'direction_primary': direction_signal,
            'direction_confidence': direction_conf,
            'upside_pct': upside_pct,
            'downside_pct': downside_pct,
            'rr': rr
        }

    # ---------------- Helper: compute TP/SL from extremes or BB ----------------
    def _compute_tp_sl(self, live_price: float, bb_levels_48: dict | None, up_pct: Optional[float], dn_pct: Optional[float], direction: str) -> Tuple[float, float, Optional[float]]:
        entry = live_price
        if up_pct is not None and dn_pct is not None and up_pct > 0 and dn_pct > 0:
            if direction == 'LONG':
                tp = entry * (1 + up_pct/100.0)
                sl = entry * (1 - dn_pct/100.0)
                rr = (up_pct / dn_pct) if dn_pct > 0 else None
            else:
                tp = entry * (1 - dn_pct/100.0)
                sl = entry * (1 + up_pct/100.0)
                rr = (dn_pct / up_pct) if up_pct > 0 else None
            return tp, sl, rr
        # Fallback BB48
        bb = bb_levels_48 or {}
        upper_2 = bb.get('upper_2', entry*1.028)
        upper_1 = bb.get('upper_1', entry*1.018)
        middle  = bb.get('middle', entry)
        lower_1 = bb.get('lower_1', entry*0.982)
        lower_2 = bb.get('lower_2', entry*0.972)
        if direction == 'LONG':
            tp = max(middle, upper_1, upper_2)
            if tp <= entry:
                tp = entry * 1.025
            sl = min(lower_1, lower_2)
            if sl >= entry:
                sl = entry * 0.972
            risk = abs(entry - sl); reward = abs(tp - entry)
            rr = (reward / risk) if risk > 0 else None
        else:
            tp = min(middle, lower_1, lower_2)
            if tp >= entry:
                tp = entry * 0.975
            sl = max(upper_1, upper_2)
            if sl <= entry:
                sl = entry * 1.028
            risk = abs(entry - sl); reward = abs(entry - tp)
            rr = (reward / risk) if risk > 0 else None
        return tp, sl, rr
            
    def init_prediction_db(self):
        """Initialize SQLite database for storing predictions"""
        try:
            os.makedirs(os.path.dirname(self.prediction_db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.prediction_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS enhanced_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    prediction_proba_0 REAL,
                    prediction_proba_1 REAL,  
                    prediction_proba_2 REAL,
                    confidence REAL NOT NULL,
                    database_price REAL NOT NULL,
                    live_price REAL NOT NULL,
                    price_difference_pct REAL,
                    bb_48_upper_2 REAL,
                    bb_48_upper_1 REAL,
                    bb_48_middle REAL,
                    bb_48_lower_1 REAL,
                    bb_48_lower_2 REAL,
                    closest_bb_level TEXT,
                    bb_signal TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.console.print(f"[green]✅ Enhanced prediction database initialized: {self.prediction_db_path}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Error initializing prediction database: {str(e)}[/red]")
            
    def get_live_price(self) -> Optional[float]:
        """Get current live price from Binance API"""
        try:
            # Endpoint berbeda untuk spot vs futures; kita sudah set self.binance_api tepat
            # Spot: /api/v3/ticker/price, Futures: /fapi/v1/ticker/price
            url = f"{self.binance_api}/ticker/price"
            params = {"symbol": self.symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            live_price = float(data['price'])
            
            return live_price
            
        except Exception as e:
            self.logger.error(f"Error fetching live price: {str(e)}")
            return None
            
    def fetch_latest_features(self, lookback_days: Optional[int] = None, lookback_candles: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Fetch latest data and compute features real-time (like build_dataset.py but no labeling)
        
        Args:
            lookback_days: Days of historical data to fetch for feature calculation
            lookback_candles: Number of candles to fetch (overrides lookback_days if set)
            
        Returns:
            DataFrame with latest computed features (single row for prediction)
        """
        try:
            self.logger.info("🔄 Fetching latest data and computing features real-time...")
            
            # Get prediction config settings
            pred_config = getattr(self.config, 'prediction', {})
            enable_realtime = pred_config.get('enable_realtime_fetch', True)
            
            # If realtime fetch disabled, fallback to database
            if not enable_realtime:
                self.logger.info("Realtime fetch disabled, using database fallback")
                fallback_rows = pred_config.get('database_fallback_rows', 300)
                return self.load_latest_database_data(fallback_rows)
            
            # Determine lookback window
            if lookback_candles is not None:
                limit = lookback_candles
                self.logger.info(f"Using lookback_candles: {limit}")
            elif lookback_days is not None:
                # Convert days to approximate candles based on timeframe
                timeframe_minutes = {
                    '5m': 5, '15m': 15, '30m': 30, '1h': 60, 
                    '2h': 120, '4h': 240, '6h': 360
                }
                minutes_per_day = 24 * 60
                tf_min = timeframe_minutes.get(self.timeframe, 15)
                limit = int((lookback_days * minutes_per_day) / tf_min)
                self.logger.info(f"Using lookback_days: {lookback_days} → {limit} candles")
            else:
                # Use config defaults
                config_days = pred_config.get('lookback_days', 14)
                config_candles = pred_config.get('lookback_candles')
                if config_candles:
                    limit = config_candles
                    self.logger.info(f"Using config lookback_candles: {limit}")
                else:
                    timeframe_minutes = {
                        '5m': 5, '15m': 15, '30m': 30, '1h': 60, 
                        '2h': 120, '4h': 240, '6h': 360
                    }
                    tf_min = timeframe_minutes.get(self.timeframe, 15)
                    limit = int((config_days * 24 * 60) / tf_min)
                    self.logger.info(f"Using config lookback_days: {config_days} → {limit} candles")
            
            # Ensure reasonable bounds
            limit = max(500, min(limit, 5000))  # Between 500-5000 candles
            
            # Initialize exchange and fetch data
            from data.binance_fetch import init_exchange, normalize_symbol, fetch_ohlcv

            # Gunakan market & symbol yang sudah distandardisasi
            exchange = init_exchange(self.market_type)
            normalized_symbol = normalize_symbol(exchange, self.symbol)
            self.logger.info(f"Fetching {limit} candles of {self.symbol} {self.timeframe} from Binance ({self.market_type})...")

            # Fetch raw OHLCV data
            raw_df = fetch_ohlcv(exchange, normalized_symbol, self.timeframe, limit=limit)
            
            if raw_df.empty:
                raise ValueError("No data received from Binance")
                
            self.logger.info(f"📊 Fetched {len(raw_df):,} raw candles from {raw_df.timestamp.min()} to {raw_df.timestamp.max()}")
            
            # Add metadata columns
            raw_df['pair'] = self.symbol
            raw_df['timeframe'] = self.timeframe
            
            # Feature engineering pipeline (same as build_dataset.py)
            self.logger.info("⚙️  Computing technical indicators...")
            from features.indicators import add_price_indicators
            enriched = add_price_indicators(raw_df.copy(), self.config)
            self.logger.info(f"📈 After indicators: {len(enriched):,} rows")
            
            self.logger.info("🔢 Computing derivative features...")
            from features.derivatives import add_derivative_features
            enriched = add_derivative_features(enriched, self.config)
            self.logger.info(f"🧮 After derivatives: {len(enriched):,} rows")
            
            # External data merge (optional, might be disabled for speed)
            external_enabled = getattr(self.config.external, 'enable_coingecko', False) or \
                             getattr(self.config.external, 'enable_coinmetrics', False) or \
                             getattr(self.config.external, 'enable_dune', False)
            
            if external_enabled:
                self.logger.info("🌐 Merging external data...")
                # Import the merge function from build_dataset
                import sys, os
                pipeline_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pipeline')
                if pipeline_path not in sys.path:
                    sys.path.append(pipeline_path)
                try:
                    # from build_dataset import merge_external  # dinonaktifkan: gunakan versi terpadu atau lewati jika tidak tersedia
                    # enriched = merge_external(enriched, self.config, self.logger)
                    self.logger.warning("merge_external dinonaktifkan di mode realtime untuk kecepatan")
                    self.logger.info(f"🔗 After external merge: {len(enriched):,} rows")
                except ImportError as e:
                    self.logger.warning(f"External data merge unavailable: {e}")
            
            # IMPORTANT: Do NOT add future labels (skip label_future_direction) 
            # but we still need vol_regime as it's used as a feature
            self.logger.info("⚠️  Skipping future labeling (not needed for prediction)")
            
            # Add vol_regime as it's used as a feature by trained models
            vol_col = 'volatility_5'  # Assuming this is the volatility column used
            if vol_col in enriched.columns:
                try:
                    # Calculate vol_regime manually (same logic as label_regime_volatility)
                    low_q, high_q = 0.33, 0.66
                    low = enriched[vol_col].quantile(low_q)
                    high = enriched[vol_col].quantile(high_q)
                    conditions = [enriched[vol_col] < low, enriched[vol_col] > high]
                    enriched['vol_regime'] = np.select(conditions, [0, 2], default=1)
                    self.logger.info("✅ Added vol_regime feature for model compatibility")
                except Exception as e:
                    self.logger.warning(f"Could not calculate vol_regime: {e}")
                    # Add a default vol_regime value (medium volatility regime)
                    enriched['vol_regime'] = 1
                    self.logger.info("⚠️  Using default vol_regime=1 (medium)")
            else:
                # If volatility column doesn't exist, use default
                enriched['vol_regime'] = 1
                self.logger.info("⚠️  volatility_5 column not found, using default vol_regime=1")
            
            # Clean NaN values at the head (from indicator warmup)
            before_clean = len(enriched)
            # Only clean base features, not future labels
            ignore_cols = {"future_return_pct", "direction"}  # Removed vol_regime as it's now a feature
            base_cols = [c for c in enriched.columns if c not in ignore_cols]
            
            if base_cols:
                mask_valid = enriched[base_cols].notna().all(axis=1)
                if mask_valid.any():
                    first_valid_idx = mask_valid.idxmax()
                    enriched = enriched.iloc[first_valid_idx:].copy()
            
            after_clean = len(enriched)
            if before_clean != after_clean:
                self.logger.info(f"🧹 Cleaned {before_clean - after_clean} rows with NaN indicators; final: {after_clean:,} rows")
            
            if enriched.empty:
                raise ValueError("No valid data after feature calculation and cleaning")
            
            # Return only the latest row (for prediction)
            latest_row = enriched.iloc[-1:].copy()
            self.logger.info(f"✅ Real-time features computed successfully (timestamp: {latest_row['timestamp'].iloc[0]})")
            
            return latest_row
            
        except Exception as e:
            self.logger.error(f"Error in fetch_latest_features: {str(e)}")
            
            # Fallback to database if enabled
            pred_config = getattr(self.config, 'prediction', {})
            if pred_config.get('fallback_to_database', True):
                self.logger.warning("🔄 Falling back to database data...")
                fallback_rows = pred_config.get('database_fallback_rows', 300)
                return self.load_latest_database_data(fallback_rows)
            else:
                self.console.print(f"[red]❌ Failed to fetch latest features and fallback disabled: {str(e)}[/red]")
                return None

    def load_latest_database_data(self, history_rows: int = 1) -> Optional[pd.DataFrame]:
        """Load latest processed data from database with optional history for slope calculations.

        Args:
            history_rows: number of most recent rows to load (>=1)
        """
        try:
            db_path = f"data/db/btc_{self.timeframe}.sqlite"

            if not os.path.exists(db_path):
                self.console.print(f"[red]❌ Database not found: {db_path}[/red]")
                return None

            if history_rows < 1:
                history_rows = 1

            conn = sqlite3.connect(db_path)
            query = f"SELECT * FROM features ORDER BY timestamp DESC LIMIT {history_rows}"
            df = pd.read_sql_query(query, conn)
            conn.close()

            if len(df) == 0:
                self.console.print("[red]❌ No data available in database[/red]")
                return None

            # Reverse to chronological order (oldest first) for rolling calculations
            df = df.iloc[::-1].reset_index(drop=True)
            return df

        except Exception as e:
            self.console.print(f"[red]❌ Error loading database data: {str(e)}[/red]")
            self.logger.error(f"Error loading database data: {str(e)}")
            return None
        """Load latest processed data from database with optional history for slope calculations.

        Args:
            history_rows: number of most recent rows to load (>=1)
        """
        try:
            db_path = f"data/db/btc_{self.timeframe}.sqlite"

            if not os.path.exists(db_path):
                self.console.print(f"[red]❌ Database not found: {db_path}[/red]")
                return None

            if history_rows < 1:
                history_rows = 1

            conn = sqlite3.connect(db_path)
            query = f"SELECT * FROM features ORDER BY timestamp DESC LIMIT {history_rows}"
            df = pd.read_sql_query(query, conn)
            conn.close()

            if len(df) == 0:
                self.console.print("[red]❌ No data available in database[/red]")
                return None

            # Reverse to chronological order (oldest first) for rolling calculations
            df = df.iloc[::-1].reset_index(drop=True)
            return df

        except Exception as e:
            self.console.print(f"[red]❌ Error loading database data: {str(e)}[/red]")
            self.logger.error(f"Error loading database data: {str(e)}")
            return None
            
    def analyze_bollinger_position(self, data: pd.DataFrame, live_price: float) -> Dict[str, Any]:
        """Analyze live price position relative to Bollinger Bands"""
        try:
            row = data.iloc[0]
            database_price = row['close']
            
            analysis = {
                'database_price': database_price,
                'live_price': live_price,
                'price_difference_pct': ((live_price - database_price) / database_price) * 100,
                'bb_levels': {},
                'bb_analysis': {},
                'signals': []
            }
            
            # Get BB levels for multiple periods
            for period in [48, 96]:
                bb_upper_2 = row.get(f'bb_{period}_upper_2.0')
                bb_upper_1 = row.get(f'bb_{period}_upper_1.0')
                bb_middle = row.get(f'bb_{period}_middle')
                bb_lower_1 = row.get(f'bb_{period}_lower_1.0')
                bb_lower_2 = row.get(f'bb_{period}_lower_2.0')
                bb_squeeze = row.get(f'bb_{period}_squeeze_flag', 0)
                
                if bb_upper_2 is not None:
                    analysis['bb_levels'][f'bb_{period}'] = {
                        'upper_2': bb_upper_2,
                        'upper_1': bb_upper_1,
                        'middle': bb_middle,
                        'lower_1': bb_lower_1,
                        'lower_2': bb_lower_2,
                        'squeeze': bb_squeeze
                    }
                    
                    # Calculate distances to each level using LIVE price
                    distances = {
                        f'BB{period} Upper(2σ)': abs(((live_price - bb_upper_2) / bb_upper_2) * 100),
                        f'BB{period} Upper(1σ)': abs(((live_price - bb_upper_1) / bb_upper_1) * 100),
                        f'BB{period} Middle': abs(((live_price - bb_middle) / bb_middle) * 100),
                        f'BB{period} Lower(1σ)': abs(((live_price - bb_lower_1) / bb_lower_1) * 100),
                        f'BB{period} Lower(2σ)': abs(((live_price - bb_lower_2) / bb_lower_2) * 100),
                    }
                    
                    # Find closest level
                    closest_level = min(distances.items(), key=lambda x: x[1])
                    
                    # Determine position and signal
                    if live_price > bb_upper_2:
                        position = f'Above BB{period} Upper(2σ) - OVERBOUGHT'
                        signal = 'SELL'
                        color = 'red'
                    elif live_price > bb_upper_1:
                        position = f'Between BB{period} Upper Bands - Strong Bullish'
                        signal = 'CAUTION'
                        color = 'yellow'
                    elif live_price > bb_middle:
                        position = f'Above BB{period} Middle - Bullish'
                        signal = 'BUY'
                        color = 'green'
                    elif live_price > bb_lower_1:
                        position = f'Between BB{period} Lower Bands - Bearish'
                        signal = 'CAUTION'
                        color = 'yellow'
                    elif live_price > bb_lower_2:
                        position = f'Below BB{period} Lower(1σ) - Oversold'
                        signal = 'BUY'
                        color = 'green'
                    else:
                        position = f'Below BB{period} Lower(2σ) - EXTREME OVERSOLD'
                        signal = 'STRONG BUY'
                        color = 'bright_green'
                        
                    analysis['bb_analysis'][f'bb_{period}'] = {
                        'position': position,
                        'signal': signal,
                        'color': color,
                        'closest_level': closest_level[0],
                        'closest_distance': closest_level[1],
                        'squeeze': bb_squeeze,
                        'distances': distances
                    }
                    
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing BB position: {str(e)}")
            return {}
            
    def make_prediction(self, data: pd.DataFrame) -> Tuple[int, np.ndarray, float, Dict[str, Any]]:
        """Make prediction on the database data.

        Returns tuple: (prediction, probabilities, confidence, metadata)
        metadata keys: missing_feature_count, total_expected_features, missing_features
        """
        try:
            # Prepare features
            if self.feature_columns:
                # Strict alignment: keep exact order, fill missing with 0, drop extras
                cols_in_df = set(data.columns)
                missing_features = [c for c in self.feature_columns if c not in cols_in_df]
                extra_features = [c for c in data.columns if c not in self.feature_columns]
                if missing_features:
                    self.console.print(f"[yellow]⚠️  Missing features ({len(missing_features)}): {missing_features[:10]}{'...' if len(missing_features)>10 else ''}[/yellow]")
                if extra_features:
                    if len(extra_features) <= 5:
                        self.console.print(f"[blue]ℹ️  Ignoring extra features: {extra_features}[/blue]")
                    else:
                        self.console.print(f"[blue]ℹ️  Ignoring {len(extra_features)} extra features[/blue]")
                # Build aligned DataFrame in the exact training order
                aligned = {}
                for col in self.feature_columns:
                    if col in data.columns:
                        aligned[col] = data[col].values
                    else:
                        aligned[col] = np.zeros(len(data))
                X = pd.DataFrame(aligned, columns=self.feature_columns)
                # Quick sanity check for counts
                if len(self.feature_columns) != X.shape[1]:
                    self.console.print(f"[red]❌ Feature alignment mismatch: expected {len(self.feature_columns)}, built {X.shape[1]}" )
            else:
                # Use all numeric columns except target-related ones
                exclude_cols = ['target', 'direction', 'future_return_pct', 'timestamp', 'symbol', 'pair', 'timeframe']  # Removed vol_regime as it's now a feature
                feature_cols = [col for col in data.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])]
                X = data[feature_cols]
                missing_features = set()
                
            # Handle any remaining missing values
            X = X.fillna(0)
            
            # Scale the features
            if self.scaler:
                try:
                    X_scaled = self.scaler.transform(X)
                except Exception as e:
                    self.console.print(f"[yellow]⚠️  Scaler mismatch: {e}. Using raw (unscaled) features.[/yellow]")
                    X_scaled = X.values
            else:
                X_scaled = X.values
                
            # Make prediction
            prediction = None
            prediction_proba = None

            model_type = self.model_type.lower()
            if model_type == 'lightgbm':
                import lightgbm as lgb
                # Align to booster feature names if available to avoid name mismatch errors
                booster_feature_names = self.model.feature_name() if hasattr(self.model, 'feature_name') else None
                if booster_feature_names and len(booster_feature_names) > 0:
                    # Use DataFrame order mapping onto scaled matrix
                    name_to_index = {name: idx for idx, name in enumerate(X.columns)}
                    col_arrays = []
                    for fname in booster_feature_names:
                        if fname in name_to_index:
                            col_arrays.append(X_scaled[:, name_to_index[fname]])
                        else:
                            # Feature absent -> zeros (length = n_samples)
                            col_arrays.append(np.zeros(X_scaled.shape[0]))
                    X_ordered = np.column_stack(col_arrays)
                else:
                    X_ordered = X_scaled  # fallback

                # Use numpy array to bypass pandas name checking
                probs = self.model.predict(X_ordered)
                if probs.ndim == 1:  # might be single-class margin -> fabricate distribution
                    # Assume ordered classes 0,1,2 and softmax-like transform not available; fallback heuristic
                    # Use a centered distribution around predicted margin sign
                    pred_idx = int(round(float(probs[0]))) if len(probs) else 1
                    if pred_idx == 0:
                        prediction_proba = np.array([0.8, 0.15, 0.05])
                    elif pred_idx == 2:
                        prediction_proba = np.array([0.05, 0.15, 0.8])
                    else:
                        prediction_proba = np.array([0.2, 0.6, 0.2])
                    prediction = int(np.argmax(prediction_proba))
                else:
                    # probs shape (n_samples, n_classes)
                    prediction_proba = probs[0]
                    prediction = int(np.argmax(prediction_proba))
            elif model_type == 'xgboost':
                import xgboost as xgb
                # Provide feature names if available to prevent mismatch errors
                if self.feature_columns and len(self.feature_columns) == X_scaled.shape[1]:
                    dmatrix = xgb.DMatrix(X_scaled, feature_names=self.feature_columns)
                else:
                    dmatrix = xgb.DMatrix(X_scaled)
                probs = self.model.predict(dmatrix)
                if probs.ndim == 1:
                    # Regression or binary fallback
                    pred_val = probs[0]
                    if pred_val < 0.5:
                        prediction_proba = np.array([0.7, 0.25, 0.05])
                        prediction = 0
                    elif pred_val > 1.5:
                        prediction_proba = np.array([0.05, 0.25, 0.7])
                        prediction = 2
                    else:
                        prediction_proba = np.array([0.2, 0.6, 0.2])
                        prediction = 1
                else:
                    prediction_proba = probs[0]
                    prediction = int(np.argmax(prediction_proba))
            elif model_type == 'catboost':
                # CatBoostClassifier predict_proba returns (n_samples, n_classes)
                if hasattr(self.model, 'predict_proba'):
                    prediction_proba = self.model.predict_proba(X_scaled)[0]
                    prediction = int(np.argmax(prediction_proba))
                else:
                    raw_pred = self.model.predict(X_scaled)
                    pred_idx = int(raw_pred[0]) if hasattr(raw_pred, '__len__') else int(raw_pred)
                    if pred_idx == 0:
                        prediction_proba = np.array([0.8, 0.15, 0.05])
                    elif pred_idx == 2:
                        prediction_proba = np.array([0.05, 0.15, 0.8])
                    else:
                        prediction_proba = np.array([0.2, 0.6, 0.2])
                    prediction = int(np.argmax(prediction_proba))
            else:
                # Fallback to scikit-learn like interface
                predictions = self.model.predict(X_scaled)
                if hasattr(self.model, 'predict_proba'):
                    probas = self.model.predict_proba(X_scaled)
                    prediction_proba = probas[0]
                    prediction = int(np.argmax(prediction_proba))
                else:
                    prediction = int(predictions[0])
                    if prediction == 0:
                        prediction_proba = np.array([0.8, 0.15, 0.05])
                    elif prediction == 2:
                        prediction_proba = np.array([0.05, 0.15, 0.8])
                    else:
                        prediction_proba = np.array([0.2, 0.6, 0.2])
                    
            confidence = float(np.max(prediction_proba))
            
            metadata = {
                'missing_feature_count': len(missing_features),
                'total_expected_features': len(self.feature_columns) if self.feature_columns else X.shape[1],
                'missing_features': sorted(list(missing_features)) if missing_features else []
            }
            return prediction, prediction_proba, confidence, metadata
            
        except Exception as e:
            self.console.print(f"[red]❌ Error making prediction: {str(e)}[/red]")
            self.logger.error(f"Error making prediction: {str(e)}")
            return -1, np.array([0.33, 0.33, 0.33]), 0.33, {'error': str(e), 'missing_feature_count': 0, 'total_expected_features': 0, 'missing_features': []}
            
    def create_enhanced_display(self, latest_data: pd.DataFrame, prediction: int, 
                              prediction_proba: np.ndarray, confidence: float, 
                              bb_analysis: Dict) -> List[Table]:
        """Create enhanced display with live price and BB analysis"""
        
        tables = []
        
        # Main prediction table
        pred_table = Table(title="🔮 ENHANCED Live Prediction Analysis")
        pred_table.add_column("Metric", style="cyan", no_wrap=True)
        pred_table.add_column("Value", style="white")
        pred_table.add_column("Details", style="dim")
        
        # Prices and timing
        from datetime import datetime, timezone
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        data_time = latest_data['timestamp'].iloc[0] if 'timestamp' in latest_data.columns else "Unknown"
        database_price = bb_analysis.get('database_price', 0)
        live_price = bb_analysis.get('live_price', 0)
        price_diff = bb_analysis.get('price_difference_pct', 0)
        
        pred_table.add_row("🕐 Current Time", current_time, "System time")
        pred_table.add_row("📅 Database Time", str(data_time), "Latest data timestamp")
        pred_table.add_row("💾 Database Price", f"${database_price:.2f}", "Price from historical data")
        pred_table.add_row("📡 Live Price", f"${live_price:.2f}", "Current Binance price")
        
        # Price difference
        diff_color = "green" if price_diff > 0 else "red" if price_diff < 0 else "white"
        pred_table.add_row("📈 Price Change", f"[{diff_color}]{price_diff:+.2f}%[/{diff_color}]", "Live vs Database")
        
        # Model prediction
        pred_table.add_row("🤖 Model", f"{self.model_type}", f"Timeframe: {self.timeframe}")
        
        prediction_labels = {0: "📉 SELL", 1: "➡️  HOLD", 2: "📈 BUY"}
        prediction_colors = {0: "red", 1: "yellow", 2: "green"}
        
        pred_text = prediction_labels[prediction]
        pred_table.add_row("🎯 ML Prediction", f"[{prediction_colors[prediction]}]{pred_text}[/{prediction_colors[prediction]}]", f"Class {prediction}")
        
        # Confidence
        confidence_pct = confidence * 100
        confidence_color = "green" if confidence > 0.8 else "yellow" if confidence > 0.6 else "red"
        pred_table.add_row("🎲 Confidence", f"{confidence_pct:.2f}%", f"[{confidence_color}]Model certainty[/{confidence_color}]")
        
        tables.append(pred_table)
        
        # Bollinger Bands Analysis Table
        bb_table = Table(title="📊 Live Price vs Bollinger Bands")
        bb_table.add_column("BB Period", style="cyan")
        bb_table.add_column("Position", style="white")
        bb_table.add_column("Signal", style="white")
        bb_table.add_column("Closest Level", style="dim")
        bb_table.add_column("Distance", style="dim")
        
        for period_key, period_data in bb_analysis.get('bb_analysis', {}).items():
            period = period_key.replace('bb_', '')
            position = period_data.get('position', 'Unknown')
            signal = period_data.get('signal', 'HOLD')
            signal_color = period_data.get('color', 'white')
            closest_level = period_data.get('closest_level', 'N/A')
            closest_distance = period_data.get('closest_distance', 0)
            
            bb_table.add_row(
                f"BB{period}",
                position[:40] + "..." if len(position) > 40 else position,
                f"[{signal_color}]{signal}[/{signal_color}]",
                closest_level.replace(f'BB{period} ', ''),
                f"{closest_distance:.2f}%"
            )
            
        tables.append(bb_table)
        
        # BB Levels Table(s) for all available periods (e.g., 48, 96)
        bb_levels_all = bb_analysis.get('bb_levels', {}) or {}
        # Sort keys like 'bb_48' by numeric period
        def _period_key(k: str) -> int:
            try:
                return int(str(k).split('_')[1])
            except Exception:
                return 0
        for key in sorted(bb_levels_all.keys(), key=_period_key):
            levels = bb_levels_all.get(key) or {}
            period = str(key).replace('bb_', '')
            levels_table = Table(title=f"💹 Bollinger Band Levels (BB{period})")
            levels_table.add_column("Level", style="cyan")
            levels_table.add_column("Price", style="white")
            levels_table.add_column("Live Distance", style="white")
            levels_table.add_column("Type", style="white")

            levels_info = [
                ("Upper (2σ)", levels.get('upper_2', 0), "RESISTANCE"),
                ("Upper (1σ)", levels.get('upper_1', 0), "Resistance"),
                ("Middle", levels.get('middle', 0), "PIVOT"),
                ("Lower (1σ)", levels.get('lower_1', 0), "Support"),
                ("Lower (2σ)", levels.get('lower_2', 0), "SUPPORT"),
            ]

            for level_name, level_price, level_type in levels_info:
                if level_price and level_price > 0:
                    distance_pct = ((live_price - level_price) / level_price) * 100
                    distance_color = "green" if distance_pct > 0 else "red"
                    levels_table.add_row(
                        level_name,
                        f"${level_price:.2f}",
                        f"[{distance_color}]{distance_pct:+.2f}%[/{distance_color}]",
                        level_type
                    )
            tables.append(levels_table)
        
        # Advanced Trading Analytics Table (from add_trading_outputs)
        if 'trading_outputs' in bb_analysis:
            trading_outputs = bb_analysis['trading_outputs']
            
            analytics_table = Table(title="📈 Advanced Trading Analytics")
            analytics_table.add_column("Metric", style="cyan")
            analytics_table.add_column("Value", style="white")
            analytics_table.add_column("Details", style="dim")
            
            # Risk Reward Ratio
            if 'risk_reward_ratio' in trading_outputs:
                rr_ratio = trading_outputs['risk_reward_ratio']
                rr_color = "green" if rr_ratio > 1.5 else "yellow" if rr_ratio > 1.0 else "red"
                analytics_table.add_row(
                    "⚖️  Risk-Reward Ratio", 
                    f"[{rr_color}]1:{rr_ratio:.2f}[/{rr_color}]",
                    "Reward vs Risk potential"
                )
            
            # Expected Return
            if 'expected_return' in trading_outputs:
                exp_return = trading_outputs['expected_return']
                exp_color = "green" if exp_return > 0 else "red"
                analytics_table.add_row(
                    "💰 Expected Return", 
                    f"[{exp_color}]{exp_return:+.2f}%[/{exp_color}]",
                    "Statistical expectation"
                )
            
            # Next Candle Prediction
            if 'next_candle_prediction' in trading_outputs:
                next_pred = trading_outputs['next_candle_prediction']
                next_color = "green" if next_pred > 0 else "red"
                analytics_table.add_row(
                    "🔮 Next Candle", 
                    f"[{next_color}]{next_pred:+.2f}%[/{next_color}]",
                    "Projected price movement"
                )
            
            # Trend Strength
            if 'trend_strength' in trading_outputs:
                trend = trading_outputs['trend_strength']
                trend_value = trend['value']
                trend_state = trend['state']
                
                # Handle NaN values in trend strength
                if np.isnan(trend_value):
                    trend_value = 25.0  # Default value if NaN
                
                trend_color = "green" if trend_state == "Strong" else "yellow" if trend_state == "Moderate" else "red"
                analytics_table.add_row(
                    "📊 Trend Strength", 
                    f"[{trend_color}]{trend_value:.1f} ({trend_state})[/{trend_color}]",
                    "ADX indicator"
                )
            
            # Volume Confirmation
            if 'volume_confirmation' in trading_outputs:
                vol_confirm = trading_outputs['volume_confirmation']
                vol_value = vol_confirm['ratio']
                vol_state = vol_confirm['state']
                vol_color = "green" if vol_state == "High" else "yellow" if vol_state == "Normal" else "red"
                analytics_table.add_row(
                    "📊 Volume Confirmation", 
                    f"[{vol_color}]{vol_value:.2f}x ({vol_state})[/{vol_color}]",
                    "Relative to average"
                )
            
            # Historical Performance
            if 'historical_performance' in trading_outputs and trading_outputs['historical_performance']:
                hist_perf = trading_outputs['historical_performance']
                win_rate = hist_perf.get('win_rate', 0)
                avg_profit = hist_perf.get('avg_profit', 0)
                win_color = "green" if win_rate > 60 else "yellow" if win_rate > 40 else "red"
                analytics_table.add_row(
                    "🏆 Similar Patterns", 
                    f"Win Rate: [{win_color}]{win_rate:.1f}%[/{win_color}]",
                    f"Avg Profit: {avg_profit:+.2f}%"
                )
            
            # Alerts
            if 'alerts' in trading_outputs and trading_outputs['alerts']:
                alerts = trading_outputs['alerts']
                alert_text = " | ".join(alerts)
                analytics_table.add_row(
                    "⚠️  Alerts", 
                    alert_text,
                    "Important market conditions"
                )
                
            tables.append(analytics_table)
        
        return tables
        
    def create_bb_slope_analysis(self, df):
        """Analyze Bollinger Bands slopes for different periods"""
        slope_analysis = {}
        
        try:
            # Calculate slope for different BB periods
            for period in [24, 48, 96]:
                if len(df) >= period + 5:  # Make sure we have enough data
                    slope_analysis[f'bb_{period}'] = analyze_bb_slope(df, period=period)
            
            return slope_analysis
        except Exception as e:
            self.console.print(f"[yellow]⚠️  Slope analysis error: {str(e)}[/yellow]")
            return {}
            
    def create_trading_recommendations(self, prediction: int, confidence: float, bb_analysis: Dict) -> Table:
        """Buat rekomendasi trading lengkap (Bahasa Indonesia) dengan Gate/Trigger/TP-SL/Size/Flip."""
        
        rec_table = Table(title="🎯 TRADING RECOMMENDATIONS & ACTION PLAN")
        rec_table.add_column("Aksi", style="cyan", no_wrap=True, width=16)
        rec_table.add_column("Detail", style="white", width=48)
        rec_table.add_column("Alasan", style="dim", width=42)
        
        live_price = bb_analysis.get('live_price', 0)
        
        # Get BB levels for recommendations
        bb_48_levels = bb_analysis.get('bb_levels', {}).get('bb_48', {})
        
        # Prediction labels and colors
        prediction_labels = {0: "📉 SELL", 1: "➡️  HOLD", 2: "📈 BUY"}
        prediction_colors = {0: "red", 1: "yellow", 2: "green"}
        
        # Main trading signal (akan di-adjust oleh gating R:R jika perlu)
        signal = prediction_labels[prediction]
        signal_color = prediction_colors[prediction]

        # Reward:Risk gating (arah-aware) dan prefer hasil extremes untuk TP/SL
        # rr_min: gunakan override CLI bila ada, lalu config, fallback 1.1
        rr_min = getattr(self, 'rr_min_override', None)
        rr_gate_triggered = False
        multi_context = bb_analysis.get('_multi_result') if isinstance(bb_analysis.get('_multi_result'), dict) else (bb_analysis.get('multi_model') if isinstance(bb_analysis.get('multi_model'), dict) else None)
        up_pct = None; dn_pct = None; rr_long = None; rr_short = None
        if rr_min is None:
            try:
                if hasattr(self.config, 'multi_horizon'):
                    rr_min = self.config.multi_horizon.get('rr_min')
                elif isinstance(self.config, dict):
                    rr_min = self.config.get('multi_horizon', {}).get('rr_min')
            except Exception:
                rr_min = None
        if rr_min is None:
            rr_min = 1.1
        if multi_context:
            up_pct = multi_context.get('upside_pct')
            dn_pct = multi_context.get('downside_pct')
            # Derive directional RR
            if up_pct is not None and dn_pct is not None and up_pct > 0 and dn_pct > 0:
                rr_long = (up_pct / dn_pct)  # good if >= rr_min
                rr_short = (dn_pct / up_pct) # good if >= rr_min
        # Direction-aware gating
        if rr_min and prediction in (0,2) and rr_long is not None and rr_short is not None:
            rr_for_decision = rr_long if prediction == 2 else rr_short
            if rr_for_decision < rr_min:
                rr_gate_triggered = True
                signal = prediction_labels[1]
                signal_color = prediction_colors[1]
        
        reason_extra = f" | R:R below {rr_min}" if 'rr_gate_triggered' in bb_analysis else ""
        base_reason = f"ML Model confidence: {confidence*100:.1f}%" + reason_extra
        rec_table.add_row(
            f"[{signal_color}]🎯 SIGNAL[/{signal_color}]",
            f"[{signal_color}]{signal}[/{signal_color}]",
            base_reason
        )

        # Ringkasan Gate R:R per arah
        long_gate_ok = (rr_long is not None and rr_long >= rr_min)
        short_gate_ok = (rr_short is not None and rr_short >= rr_min)
        rr_long_txt = "-" if rr_long is None else f"{rr_long:.2f}"
        rr_short_txt = "-" if rr_short is None else f"{rr_short:.2f}"
        gate_detail = f"rr_min={rr_min:.2f} | LONG {rr_long_txt} → {'Lolos' if long_gate_ok else 'Terblokir'} | SHORT {rr_short_txt} → {'Lolos' if short_gate_ok else 'Terblokir'}"
        rec_table.add_row("📏 GATE R:R", gate_detail, "Menggunakan estimasi extremes (Up/Down); fallback BB bila tidak tersedia.")

        # Sinyal gabungan ML+BB untuk kebutuhan timing
        try:
            bb_sig_payload = integrate_bb_signals(bb_analysis.get('bb_analysis', {}) or {},
                                                  slope_analysis=bb_analysis.get('slope_analysis'))
        except Exception:
            bb_sig_payload = {'integrated_signal': 'NEUTRAL', 'confidence': 0.5}
        resolved = resolve_signal_conflict(
            bb_sig_payload.get('integrated_signal', 'NEUTRAL'),
            prediction,
            float(confidence or 0.0),
            float(bb_sig_payload.get('confidence', 0.5))
        )

        # Prefer-RR Mode: jika aktif dan ML confidence tidak terlalu tinggi, prioritaskan arah dengan R:R >= rr_min
        prefer_rr_active = bool(getattr(self, 'prefer_rr_mode', False)) and (float(confidence or 0.0) <= float(getattr(self, 'prefer_rr_max_ml_conf', 0.9)))
        prefer_dir = None
        if prefer_rr_active and (rr_long is not None or rr_short is not None):
            if long_gate_ok and not short_gate_ok:
                prefer_dir = 'LONG'
            elif short_gate_ok and not long_gate_ok:
                prefer_dir = 'SHORT'
            elif long_gate_ok and short_gate_ok:
                # pilih yang R:R lebih besar
                try:
                    prefer_dir = 'LONG' if (rr_long or 0) >= (rr_short or 0) else 'SHORT'
                except Exception:
                    prefer_dir = None
        # Tampilkan status mode
        if prefer_rr_active:
            mode_detail = f"Prefer-RR aktif (maks ML conf {getattr(self,'prefer_rr_max_ml_conf',0.9):.2f})"
            mode_reason = "Memihak arah dengan R:R ≥ rr_min hanya jika ada TRIGGER valid dan konteks BB selaras."
            if prefer_dir:
                mode_detail += f" → Prioritas: {prefer_dir}"
            rec_table.add_row("MODE", mode_detail, mode_reason)
        
        # Entry recommendations
        effective_pred = 1 if rr_gate_triggered else prediction

        if effective_pred == 2:  # BUY
            confidence_level = "🔥 HIGH" if confidence > 0.9 else "⚡ MEDIUM" if confidence > 0.7 else "⚠️  LOW"
            rec_table.add_row(
                "📈 ENTRY",
                f"LONG position at ~${live_price:.0f}",
                f"Confidence: {confidence_level}" + (" | R:R Gate" if rr_gate_triggered else "")
            )
            # Position sizing
            if confidence > 0.95:
                position_size = "2-3% of portfolio"; size_reason = "Very high confidence"
            elif confidence > 0.85:
                position_size = "1-2% of portfolio"; size_reason = "High confidence"
            else:
                position_size = "0.5-1% of portfolio"; size_reason = "Moderate confidence"
            rec_table.add_row("💰 POSITION", position_size, size_reason)

            # Risk management (prefer extremes-based TP/SL if available; fallback ke BB)
            if bb_48_levels:
                upper_2 = bb_48_levels.get('upper_2', live_price * 1.03)
                upper_1 = bb_48_levels.get('upper_1', live_price * 1.02)
                middle = bb_48_levels.get('middle', live_price)
                lower_1 = bb_48_levels.get('lower_1', live_price * 0.98)
                lower_2 = bb_48_levels.get('lower_2', live_price * 0.97)
                entry = live_price
                if up_pct is not None and dn_pct is not None and up_pct > 0 and dn_pct > 0:
                    tp = entry * (1 + up_pct/100.0)
                    sl = entry * (1 - dn_pct/100.0)
                    tp_note = "Extremes model (max_high)"; sl_note = "Extremes model (min_low)"
                    rr_ratio = (up_pct / dn_pct) if dn_pct > 0 else None
                else:
                    tp_pct = 2.5; sl_pct = 2.8
                    tp = max(middle, upper_1, upper_2)
                    tp_note = "BB Middle/Upper - resistance" if tp > entry else f"Default +{tp_pct}% from entry"; tp = tp if tp > entry else entry*(1+tp_pct/100)
                    sl = min(lower_1, lower_2)
                    sl_note = "BB Lower(2σ) - strong support" if sl < entry else f"Default -{sl_pct}% from entry"; sl = sl if sl < entry else entry*(1-sl_pct/100)
                    risk = abs(entry - sl); reward = abs(tp - entry); rr_ratio = (reward/risk) if risk>0 else None
                rec_table.add_row("🛡️  STOP LOSS", f"${sl:.0f} ({((sl-entry)/entry*100):+.1f}%)", sl_note)
                rec_table.add_row("🎯 TAKE PROFIT", f"${tp:.0f} ({((tp-entry)/entry*100):+.1f}%)", tp_note)
                if rr_ratio is not None:
                    rec_table.add_row("⚖️  R:R RATIO", f"1:{rr_ratio:.1f}", "Risk vs Reward analysis (LONG)")

        elif effective_pred == 0:  # SELL
            rec_table.add_row(
                "📉 ENTRY",
                f"SHORT position at ~${live_price:.0f}",
                f"Bearish signal ({confidence*100:.1f}%)" + (" | R:R Gate" if rr_gate_triggered else "")
            )
            # Risk management (prefer extremes first)
            if bb_48_levels:
                upper_2 = bb_48_levels.get('upper_2', live_price * 1.03)
                upper_1 = bb_48_levels.get('upper_1', live_price * 1.02)
                middle = bb_48_levels.get('middle', live_price)
                lower_1 = bb_48_levels.get('lower_1', live_price * 0.98)
                lower_2 = bb_48_levels.get('lower_2', live_price * 0.97)
                entry = live_price
                if up_pct is not None and dn_pct is not None and up_pct > 0 and dn_pct > 0:
                    tp = entry * (1 - dn_pct/100.0)
                    sl = entry * (1 + up_pct/100.0)
                    tp_note = "Extremes model (min_low)"; sl_note = "Extremes model (max_high)"
                    rr_ratio = (dn_pct / up_pct) if up_pct > 0 else None
                else:
                    tp_pct = 2.5; sl_pct = 2.8
                    tp = min(middle, lower_1, lower_2)
                    tp_note = "BB Middle/Lower - support" if tp < entry else f"Default -{tp_pct}% from entry"; tp = tp if tp < entry else entry*(1-tp_pct/100)
                    sl = max(upper_1, upper_2)
                    sl_note = "BB Upper(2σ) - strong resistance" if sl > entry else f"Default +{sl_pct}% from entry"; sl = sl if sl > entry else entry*(1+sl_pct/100)
                    risk = abs(entry - sl); reward = abs(entry - tp); rr_ratio = (reward/risk) if risk>0 else None
                rec_table.add_row("🛡️  STOP LOSS", f"${sl:.0f} ({((sl-entry)/entry*100):+.1f}%)", sl_note)
                rec_table.add_row("🎯 TAKE PROFIT", f"${tp:.0f} ({((tp-entry)/entry*100):+.1f}%)", tp_note)
                if rr_ratio is not None:
                    rec_table.add_row("⚖️  R:R RATIO", f"1:{rr_ratio:.1f}", "Risk vs Reward analysis (SHORT)")

        else:  # HOLD
            rec_table.add_row(
                "⏸️  ENTRY",
                "WAIT for clearer signals",
                ("R:R below threshold" if rr_gate_triggered else "Mixed/unclear market condition")
            )
        
        # BB Position context (klasifikasi yang lebih akurat berbasis posisi/level terdekat)
        bb_context = []
        for period_key, period_data in bb_analysis.get('bb_analysis', {}).items():
            pos_text = str(period_data.get('position', '')).lower()
            closest = str(period_data.get('closest_level', '')).lower()
            # Jika ada percent_b gunakan batas umum: <=0.1 Oversold, >=0.9 Overbought
            pb = period_data.get('percent_b')
            label = None
            try:
                if isinstance(pb, (int, float)):
                    if pb <= 0.10:
                        label = 'Oversold'
                    elif pb >= 0.90:
                        label = 'Overbought'
                    elif pb > 0.50:
                        label = 'Bullish (di atas Middle)'
                    elif pb < 0.50:
                        label = 'Bearish (di bawah Middle)'
                    else:
                        label = 'Dekat Middle'
            except Exception:
                label = None
            # Fallback: gunakan nearest level / deskripsi posisi
            if label is None:
                if 'upper' in closest:
                    label = 'Overbought'
                elif 'lower' in closest:
                    label = 'Oversold'
                else:
                    if 'above' in pos_text and 'middle' in pos_text:
                        label = 'Bullish (di atas Middle)'
                    elif 'below' in pos_text and 'middle' in pos_text:
                        label = 'Bearish (di bawah Middle)'
                    else:
                        label = 'Dekat Middle'
            bb_context.append(f"{period_key.upper()}: {label}")
                
        if bb_context:
            rec_table.add_row(
                "📊 BB CONTEXT",
                " | ".join(bb_context),
                "Bollinger Bands market position"
            )
        
        # TIMING berbasis sinyal ter‑resolve + gate
        res_pos = resolved.get('position', 'WAIT')  # LONG/SHORT/WAIT
        # Jika prefer-RR aktif dan ML tidak terlalu yakin, arahkan timing ke prefer_dir bila ada
        if prefer_rr_active and prefer_dir:
            res_pos = prefer_dir
        if res_pos == 'LONG' and (rr_long is None or rr_long >= rr_min):
            timing = "🟢 SIAP PADA TRIGGER LONG"
        elif res_pos == 'SHORT' and (rr_short is None or rr_short >= rr_min):
            timing = "🟢 SIAP PADA TRIGGER SHORT"
        elif res_pos == 'WAIT':
            timing = "⏸️  TUNGGU KONFIRMASI"
        else:
            timing = "🟡 TUNGGU"
        timing_reason = f"Resolved: {resolved.get('signal','-')} | ML {resolved.get('ml_weight',0):.0%} vs BB {resolved.get('bb_weight',0):.0%} | rr_min={rr_min:.2f}"
        rec_table.add_row("⏰ TIMING", timing, timing_reason)

        # TRIGGER (micro‑playbook) dan TP/SL ringkas
        rec_table.add_row("TRIGGER LONG", "Breakout + retest di atas BB48 U2σ; band melebar & volume naik.", "Hindari LONG saat menempel band atas tanpa ekspansi.")
        rec_table.add_row("TRIGGER SHORT", "Rejection/stall di BB48 U1σ/U2σ + struktur mikro bearish.", "SHORT wajar bila RR_SHORT ≥ rr_min dan momentum melemah.")
        # Tampilkan TP/SL ringkas menggunakan extremes terlebih dahulu (fallback BB)
        def _fmt_price(p):
            try:
                return f"${p:,.2f}"
            except Exception:
                return "-"
        # Siapkan level BB48 untuk fallback
        upper_2 = bb_48_levels.get('upper_2', None)
        upper_1 = bb_48_levels.get('upper_1', None)
        middle = bb_48_levels.get('middle', None)
        lower_1 = bb_48_levels.get('lower_1', None)
        lower_2 = bb_48_levels.get('lower_2', None)
        if live_price:
            if up_pct is not None and dn_pct is not None and up_pct > 0 and dn_pct > 0:
                tp_long = live_price * (1 + up_pct/100.0)
                sl_long = live_price * (1 - dn_pct/100.0)
                tp_short = live_price * (1 - dn_pct/100.0)
                sl_short = live_price * (1 + up_pct/100.0)
            else:
                tp_long = upper_2 or upper_1 or (live_price*1.025)
                sl_long = middle or lower_1 or (live_price*0.972)
                tp_short = middle or lower_1 or (live_price*0.975)
                sl_short = upper_2 or upper_1 or (live_price*1.028)
            rec_table.add_row("TP/SL LONG", f"TP: {_fmt_price(tp_long)} | SL: {_fmt_price(sl_long)}", "Prioritas extremes; fallback BB48.")
            rec_table.add_row("TP/SL SHORT", f"TP: {_fmt_price(tp_short)} | SL: {_fmt_price(sl_short)}", "Prioritas extremes; fallback BB48.")

        # Rekomendasi ukuran posisi generik
        rec_table.add_row("UKURAN POSISI", "0.25R saat sinyal konflik; 0.5–1.0R bila ML+BB+R:R selaras.", "Kurangi risiko saat bobot ML mendominasi HOLD atau sinyal BB bertentangan.")

        # Kondisi pembalikan (Flip)
        rec_table.add_row("FLIP LONG", "LONG bila RR_LONG ≥ rr_min, BB → BUY, dan ML → BUY/WEAK BUY.", "Butuh konfluensi arah + edge R:R.")
        rec_table.add_row("FLIP SHORT", "SHORT bila RR_SHORT ≥ rr_min, BB → SELL, dan ML → SELL/WEAK SELL.", "Butuh konfluensi arah + edge R:R.")
        
        # Add trading outputs insights (if available)
        if 'trading_outputs' in bb_analysis:
            trading_outputs = bb_analysis['trading_outputs']
            
            # Add trend strength from ADX
            if 'trend_strength' in trading_outputs:
                trend = trading_outputs['trend_strength']
                trend_state = trend['state']
                trend_value = trend['value']
                
                # Handle NaN values in trend strength
                if np.isnan(trend_value):
                    trend_value = 25.0  # Default value if NaN
                
                rec_table.add_row(
                    "💪 TREND",
                    f"{trend_state} ({trend_value:.1f})",
                    "ADX trend strength analysis"
                )
            
            # Add volume confirmation
            if 'volume_confirmation' in trading_outputs:
                vol_confirm = trading_outputs['volume_confirmation']
                vol_state = vol_confirm['state']
                vol_ratio = vol_confirm['ratio']
                
                rec_table.add_row(
                    "📊 VOLUME",
                    f"{vol_state} ({vol_ratio:.2f}x avg)",
                    "Volume confirmation analysis"
                )
            
            # Add historical performance
            if 'historical_performance' in trading_outputs and trading_outputs['historical_performance']:
                hist_perf = trading_outputs['historical_performance']
                win_rate = hist_perf.get('win_rate', 0)
                
                rec_table.add_row(
                    "🔄 PATTERN",
                    f"Win Rate: {win_rate:.1f}%",
                    "Historical performance of similar patterns"
                )
            
            # Add alerts if any
            if 'alerts' in trading_outputs and trading_outputs['alerts']:
                alerts = trading_outputs['alerts']
                alert_text = " | ".join(alerts)
                
                rec_table.add_row(
                    "⚠️  ALERTS",
                    alert_text,
                    "Important market conditions"
                )
        
        return rec_table
        
    def create_trading_summary(self, prediction: int, confidence: float, bb_analysis: Dict) -> str:
        """Create quick trading summary text"""
        
        # Get all BB signals
        bb_signals = {}
        for period_key, period_data in bb_analysis.get('bb_analysis', {}).items():
            bb_signals[period_key] = period_data.get('signal', 'HOLD')
        
        # Get slope analysis if available
        slope_analysis = bb_analysis.get('slope_analysis', {})
        
        # Integrate BB signals using proper weighting if available
        integrated_bb = {}
        if bb_signals and 'bb_48' in bb_signals:
            try:
                integrated_bb = integrate_bb_signals(bb_analysis.get('bb_analysis', {}), slope_analysis)
            except Exception as e:
                self.console.print(f"[yellow]⚠️  Error integrating BB signals: {str(e)}[/yellow]")
        
        # Resolve conflicts between ML and BB
        ml_signal_text = {0: "SELL", 1: "HOLD", 2: "BUY"}
        ml_signal = ml_signal_text.get(prediction, "UNKNOWN")
        
        bb_confidence = integrated_bb.get('confidence', 0.7)
        bb_signal = integrated_bb.get('integrated_signal', "NEUTRAL")
        
        # Resolve any conflicts between ML and BB signals
        resolved = resolve_signal_conflict(bb_signal, prediction, confidence, bb_confidence)
        
        prediction_labels = {0: "📉 SELL", 1: "➡️  HOLD", 2: "📈 BUY"}
        prediction_colors = {0: "red", 1: "yellow", 2: "green"}
        
        # Use resolved signal instead of just ML prediction
        signal_map = {"BUY": 2, "STRONG BUY": 2, "SELL": 0, "STRONG SELL": 0, "HOLD": 1, "NEUTRAL": 1}
        resolved_prediction = signal_map.get(resolved['signal'], prediction)
        
        signal = prediction_labels.get(resolved_prediction, prediction_labels[prediction])
        signal_color = prediction_colors.get(resolved_prediction, prediction_colors[prediction])
        live_price = bb_analysis.get('live_price', 0)
        
        summary_lines = []
        
        # Main signal
        if resolved['conflict']:
            summary_lines.append(f"[{signal_color}]🎯 PRIMARY SIGNAL: {signal} (ML Conf: {confidence*100:.1f}%, BB Conf: {bb_confidence*100:.1f}%)[/{signal_color}]")
        else:
            summary_lines.append(f"[{signal_color}]🎯 PRIMARY SIGNAL: {signal} (Confidence: {confidence*100:.1f}%)[/{signal_color}]")
        summary_lines.append("")
        
        # Price context
        summary_lines.append(f"💰 Current BTC Price: ${live_price:.2f}")
        
        # BB Analysis summary
        bb_signal_texts = []
        for period_key, period_signal in bb_signals.items():
            if period_signal in ['BUY', 'STRONG BUY']:
                bb_signal_texts.append(f"{period_key.replace('bb_', 'BB').upper()}: {period_signal}")
            elif period_signal in ['SELL', 'STRONG SELL']:
                bb_signal_texts.append(f"{period_key.replace('bb_', 'BB').upper()}: {period_signal}")
        
        if bb_signal_texts:
            summary_lines.append(f"📊 BB Signals: {' | '.join(bb_signal_texts)}")
            
        # Add slope analysis if available
        slope_texts = []
        for period, slope_data in slope_analysis.items():
            trend = slope_data.get('trend', 'FLAT')
            strength = slope_data.get('trend_strength', 'WEAK')
            if trend != 'FLAT':
                slope_texts.append(f"BB{period}: {trend} ({strength})")
        
        if slope_texts:
            summary_lines.append(f"📈 BB Trends: {' | '.join(slope_texts)}")
        
        # Quick action
        # Use resolved signal for recommendation, not just ML
        if resolved['position'] == 'LONG' and bb_confidence > 0.8 and confidence > 0.8:
            summary_lines.append("")
            summary_lines.append("[bright_green]⚡ RECOMMENDED ACTION: Enter LONG position with strong conviction[/bright_green]")
        elif resolved['position'] == 'LONG':
            summary_lines.append("")
            summary_lines.append("[green]⚡ RECOMMENDED ACTION: Consider LONG position with caution[/green]")
        elif resolved['position'] == 'SHORT' and bb_confidence > 0.8 and confidence > 0.8:
            summary_lines.append("")
            summary_lines.append("[bright_red]⚡ RECOMMENDED ACTION: Enter SHORT position with strong conviction[/bright_red]")
        elif resolved['position'] == 'SHORT':
            summary_lines.append("")
            summary_lines.append("[red]⚡ RECOMMENDED ACTION: Consider SHORT position or exit LONG[/red]")
        else:
            summary_lines.append("")
            summary_lines.append("[yellow]⚡ RECOMMENDED ACTION: WAIT for clearer signals[/yellow]")
            
        # Add reasoning
        summary_lines.append("")
        summary_lines.append(f"🧠 Reasoning: {resolved['reasoning']}")
        
        # Add trading outputs if available
        if 'trading_outputs' in bb_analysis:
            trading_outputs = bb_analysis['trading_outputs']
            summary_lines.append("")
            
            # Add trend strength from ADX
            if 'trend_strength' in trading_outputs:
                trend = trading_outputs['trend_strength']
                trend_state = trend['state']
                trend_value = trend['value']
                
                # Handle NaN values
                if np.isnan(trend_value):
                    trend_value = 25.0  # Default value if NaN
                
                trend_color = "green" if trend_state == "Strong" else "yellow" if trend_state == "Moderate" else "red"
                summary_lines.append(f"[{trend_color}]💪 Trend Strength: {trend_state} (ADX: {trend_value:.1f})[/{trend_color}]")
            
            # Add volume confirmation
            if 'volume_confirmation' in trading_outputs:
                vol_confirm = trading_outputs['volume_confirmation']
                vol_state = vol_confirm['state']
                vol_ratio = vol_confirm['ratio']
                vol_color = "green" if vol_state == "High" else "yellow" if vol_state == "Normal" else "red"
                summary_lines.append(f"[{vol_color}]📊 Volume: {vol_state} ({vol_ratio:.2f}x average)[/{vol_color}]")
            
            # Add expected return
            if 'expected_return' in trading_outputs:
                exp_return = trading_outputs['expected_return']
                exp_color = "green" if exp_return > 0 else "red"
                summary_lines.append(f"[{exp_color}]💰 Expected Return: {exp_return:+.2f}%[/{exp_color}]")
            
            # Add alerts if any
            if 'alerts' in trading_outputs and trading_outputs['alerts']:
                alerts = trading_outputs['alerts']
                summary_lines.append("")
                summary_lines.append(f"⚠️  ALERTS: {' | '.join(alerts)}")
        
        return "\n".join(summary_lines)
        
    def run_single_prediction(self, history_rows: int = 1, confidence_threshold: Optional[float] = None, 
                            output_mode: str = 'rich', lookback_days: Optional[int] = None, 
                            lookback_candles: Optional[int] = None) -> bool:
        """Run a single enhanced prediction cycle"""
        try:
            # Check if real-time fetching is enabled
            pred_config = getattr(self.config, 'prediction', {})
            enable_realtime = pred_config.get('enable_realtime_fetch', True)
            
            if enable_realtime:
                # Fetch fresh data and compute features real-time
                self.console.print("[yellow]🔄 Fetching latest data and computing features real-time...[/yellow]")
                latest_data = self.fetch_latest_features(lookback_days=lookback_days, lookback_candles=lookback_candles)
                
                if latest_data is None:
                    return False
                    
                # For slope analysis, we need more history - try to get it or fallback to database
                latest_data_full = latest_data  # For now, single row is sufficient
                self.console.print(f"[green]✅ Real-time features computed (timestamp: {latest_data['timestamp'].iloc[0]})[/green]")
                
            else:
                # Fallback to database (old behavior)
                self.console.print("[yellow]📊 Loading database features (realtime fetch disabled)...[/yellow]")
                latest_data_full = self.load_latest_database_data(history_rows=history_rows)
                latest_data = None if latest_data_full is None else latest_data_full.tail(1).copy()
                
                if latest_data is None:
                    return False
                
            # Get live price
            self.console.print("[yellow]📡 Fetching live price from Binance...[/yellow]")
            live_price = self.get_live_price()
            
            if live_price is None:
                self.console.print("[red]❌ Failed to fetch live price[/red]")
                return False
                
            self.console.print(f"[green]✅ Live price: ${live_price:.2f}[/green]")
            
            # Analyze Bollinger Bands with live price
            self.console.print("[yellow]📊 Analyzing Bollinger Bands position...[/yellow]")
            bb_analysis = self.analyze_bollinger_position(latest_data, live_price)
            
            # Add Bollinger Bands slope analysis
            self.console.print("[yellow]📈 Analyzing Bollinger Bands slopes...[/yellow]")
            slope_analysis = self.create_bb_slope_analysis(latest_data_full) if latest_data_full is not None else {}
            if slope_analysis:
                bb_analysis['slope_analysis'] = slope_analysis
            
            # Make prediction on database features
            self.console.print("[yellow]🧠 Making prediction...[/yellow]")
            prediction, prediction_proba, confidence, pred_meta = self.make_prediction(latest_data)

            # Multi-model inference (jika tersedia latest_data lengkap)
            multi_result = None
            if hasattr(self, 'multi_models') and self.multi_models:
                try:
                    multi_result = self.predict_multi(latest_data)
                except Exception as e:
                    self.logger.warning(f"Multi-model inference gagal: {e}")
            if multi_result:
                # tempel ke bb_analysis agar bisa dipakai gating
                bb_analysis['_multi_result'] = multi_result

            # Apply confidence threshold override if provided
            overridden = False
            if confidence_threshold is not None and confidence < confidence_threshold:
                overridden = True
                original_prediction = prediction
                prediction = 1  # Force HOLD
            else:
                original_prediction = prediction
            
            if prediction == -1:
                return False
            
            # Generate additional trading outputs
            trading_outputs = {}
            if self.talib_available:
                self.console.print("[yellow]📊 Generating advanced trading analytics...[/yellow]")
                trading_outputs = add_trading_outputs(latest_data, live_price, prediction, confidence)
            else:
                self.console.print("[yellow]⚠️  Advanced analytics disabled (TA-Lib not available)[/yellow]")
            
            # Store trading outputs in bb_analysis for display
            bb_analysis['trading_outputs'] = trading_outputs
            
            # Prepare resolved signal and RR gating for simulator and UI coherence
            try:
                bb_sig_payload = integrate_bb_signals(bb_analysis.get('bb_analysis', {}) or {},
                                                      slope_analysis=bb_analysis.get('slope_analysis'))
            except Exception:
                bb_sig_payload = {'integrated_signal': 'NEUTRAL', 'confidence': 0.5}
            resolved = resolve_signal_conflict(
                bb_sig_payload.get('integrated_signal', 'NEUTRAL'),
                prediction,
                float(confidence or 0.0),
                float(bb_sig_payload.get('confidence', 0.5))
            )
            # RR gate values
            rr_min_val = None
            try:
                rr_min_val = self.config.multi_horizon.get('rr_min') if hasattr(self.config, 'multi_horizon') else None
            except Exception:
                rr_min_val = None
            if rr_min_val is None:
                rr_min_val = 1.1
            up_pct = None; dn_pct = None; rr_long = None; rr_short = None
            multi_result = bb_analysis.get('_multi_result')
            if multi_result:
                up_pct = multi_result.get('upside_pct')
                dn_pct = multi_result.get('downside_pct')
                if up_pct is not None and dn_pct is not None and up_pct>0 and dn_pct>0:
                    rr_long = up_pct / dn_pct
                    rr_short = dn_pct / up_pct
            long_ok = (rr_long is None) or (rr_long >= rr_min_val)
            short_ok = (rr_short is None) or (rr_short >= rr_min_val)
            resolved_pos = resolved.get('position', 'WAIT')
            
            # Prepare resolved signal and RR gate values for simulator/UI coherence
            try:
                bb_sig_payload = integrate_bb_signals(bb_analysis.get('bb_analysis', {}) or {},
                                                      slope_analysis=bb_analysis.get('slope_analysis'))
            except Exception:
                bb_sig_payload = {'integrated_signal': 'NEUTRAL', 'confidence': 0.5}
            resolved = resolve_signal_conflict(
                bb_sig_payload.get('integrated_signal', 'NEUTRAL'),
                prediction,
                float(confidence or 0.0),
                float(bb_sig_payload.get('confidence', 0.5))
            )
            rr_min_val = None
            try:
                rr_min_val = self.config.multi_horizon.get('rr_min') if hasattr(self.config, 'multi_horizon') else None
            except Exception:
                rr_min_val = None
            if rr_min_val is None:
                rr_min_val = 1.1
            up_pct = None; dn_pct = None; rr_long = None; rr_short = None
            if multi_result:
                up_pct = multi_result.get('upside_pct')
                dn_pct = multi_result.get('downside_pct')
                if up_pct is not None and dn_pct is not None and up_pct>0 and dn_pct>0:
                    rr_long = up_pct / dn_pct
                    rr_short = dn_pct / up_pct
            long_ok = rr_long is None or rr_long >= rr_min_val
            short_ok = rr_short is None or rr_short >= rr_min_val
            resolved_pos = resolved.get('position', 'WAIT')

            # Display enhanced results
            tables = self.create_enhanced_display(latest_data, prediction, prediction_proba, confidence, bb_analysis)
            trading_rec = self.create_trading_recommendations(prediction, confidence, bb_analysis)
            
            # JSON output mode (no rich tables)
            if output_mode == 'json':
                import datetime as _dt
                import json
                ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
                bb_signals = {k: v.get('signal') for k, v in bb_analysis.get('bb_analysis', {}).items()}
                # Compute directional RR for gating summary
                rr_long_json = None; rr_short_json = None; rr_gate_flag = False
                if multi_result and multi_result.get('upside_pct') is not None and multi_result.get('downside_pct') is not None:
                    up_pct_j = multi_result.get('upside_pct') or 0
                    dn_pct_j = abs(multi_result.get('downside_pct') or 0)
                    if up_pct_j>0 and dn_pct_j>0:
                        rr_long_json = up_pct_j / dn_pct_j
                        rr_short_json = dn_pct_j / up_pct_j
                        rr_min_val = self.config.multi_horizon.get('rr_min',1.2) if hasattr(self.config,'multi_horizon') else 1.2
                        if prediction in (0,2):
                            rr_for_decision = rr_long_json if prediction==2 else rr_short_json
                            rr_gate_flag = (rr_for_decision is not None and rr_for_decision < rr_min_val)
                out_obj = {
                    'timestamp': ts,
                    'timeframe': self.timeframe,
                    'model_type': self.model_type,
                    'prediction': prediction,
                    'prediction_original': original_prediction,
                    'confidence': confidence,
                    'overridden': overridden,
                    'confidence_threshold': confidence_threshold,
                    'prediction_proba': prediction_proba.tolist(),
                    'bb_signals': bb_signals,
                    'missing_feature_count': pred_meta.get('missing_feature_count'),
                    'total_expected_features': pred_meta.get('total_expected_features'),
                    'missing_features': pred_meta.get('missing_features'),
                    'price': bb_analysis.get('live_price'),
                    'price_db': bb_analysis.get('database_price'),
                    'price_diff_pct': bb_analysis.get('price_difference_pct'),
                    'multi_model': multi_result if multi_result else None,
                    'rr_long': rr_long_json,
                    'rr_short': rr_short_json,
                    'rr_gate_triggered': bool(rr_gate_flag)
                }
                print(json.dumps(out_obj, ensure_ascii=False))
                return True

            self.console.clear()
            header_title = "🚀 ENHANCED Live Trading Analysis with Bollinger Bands"
            if overridden:
                header_title += f" (Overridden to HOLD due to confidence<{confidence_threshold})"
            self.console.print(Panel(
                header_title,
                title="System Status"
            ))

            # Feature integrity warning panel
            if pred_meta.get('missing_feature_count', 0) > 0:
                ratio = pred_meta['missing_feature_count'] / max(1, pred_meta.get('total_expected_features', 1))
                style = 'yellow' if ratio <= 0.1 else 'red'
                self.console.print(Panel(
                    f"Missing {pred_meta['missing_feature_count']} / {pred_meta.get('total_expected_features')} features ({ratio*100:.1f}%). Prediction reliability reduced.",
                    title="Feature Integrity Warning",
                    border_style=style
                ))
            
            # Print each table separately
            for table in tables:
                self.console.print(table)
                self.console.print()

            # Tampilkan panel multi-model jika ada
            if multi_result and isinstance(multi_result, dict) and 'targets' in multi_result:
                from rich.table import Table
                mm = multi_result
                mm_table = Table(title="🧠 Multi-Model Targets & Extremes")
                mm_table.add_column("Target", style="cyan")
                mm_table.add_column("Pred/Primary", style="white")
                mm_table.add_column("Extra", style="dim")
                # Direction primary line
                if mm.get('direction_primary'):
                    mm_table.add_row("direction_primary", str(mm['direction_primary']), f"conf={mm.get('direction_confidence'):.3f}")
                # Extremes summary
                if mm.get('upside_pct') is not None and mm.get('downside_pct') is not None:
                    up_pct = mm.get('upside_pct') or 0
                    dn_pct = abs(mm.get('downside_pct') or 0)
                    rr_long = (up_pct/dn_pct) if (up_pct>0 and dn_pct>0) else None
                    rr_short = (dn_pct/up_pct) if (up_pct>0 and dn_pct>0) else None
                    rr_min = self.config.multi_horizon.get('rr_min',1.2)
                    if rr_long is not None:
                        rr_color = 'green' if rr_long>= rr_min else 'yellow'
                        mm_table.add_row("reward_risk_long", f"1:{rr_long:.2f}", f"up={up_pct:.2f}% dn={dn_pct:.2f}%")
                    if rr_short is not None:
                        rr_color = 'green' if rr_short>= rr_min else 'yellow'
                        mm_table.add_row("reward_risk_short", f"1:{rr_short:.2f}", f"up={up_pct:.2f}% dn={dn_pct:.2f}%")
                # Each raw target
                for tgt, info in mm['targets'].items():
                    if info.get('task')=='classification':
                        probs = info.get('probs') or []
                        mm_table.add_row(tgt, str(info.get('prediction')), f"probs={[round(p,3) for p in probs]}")
                    else:
                        mm_table.add_row(tgt, f"{info.get('prediction'):.5f}" if info.get('prediction') is not None else 'None', info.get('task',''))
                self.console.print(mm_table)
                self.console.print()
            
            # Add trading recommendations
            self.console.print(trading_rec)
            self.console.print()
            
            # Add summary panel
            summary_text = self.create_trading_summary(prediction, confidence, bb_analysis)
            # Append multi-model insight (show RR for LONG & SHORT if available)
            if multi_result and multi_result.get('upside_pct') is not None and multi_result.get('downside_pct') is not None:
                up_pct = multi_result.get('upside_pct') or 0
                dn_pct = abs(multi_result.get('downside_pct') or 0)
                rr_long = (up_pct/dn_pct) if (up_pct>0 and dn_pct>0) else None
                rr_short = (dn_pct/up_pct) if (up_pct>0 and dn_pct>0) else None
                rr_min_val = self.config.multi_horizon.get('rr_min', 1.2) if hasattr(self.config,'multi_horizon') else 1.2
                parts = ["Multi-Model RR:"]
                if rr_long is not None:
                    suffix = "" if rr_long>=rr_min_val else f" (below {rr_min_val})"
                    parts.append(f"LONG 1:{rr_long:.2f}")
                    if suffix:
                        parts[-1] += suffix
                if rr_short is not None:
                    suffix = "" if rr_short>=rr_min_val else f" (below {rr_min_val})"
                    parts.append(f"SHORT 1:{rr_short:.2f}")
                    if suffix:
                        parts[-1] += suffix
                parts.append(f"(Up {up_pct:.2f}%, Down {dn_pct:.2f}%)")
                summary_text += "\n" + " ".join(parts)
            self.console.print(Panel(
                summary_text,
                title="⚡ QUICK TRADING SUMMARY",
                border_style="bright_yellow"
            ))
            
            # Paper-trade simulator step (optional, CSV logging)
            if getattr(self, 'paper_trade_log', None):
                if self._paper_engine is None:
                    self._paper_engine = self._PaperTradeEngine(self.paper_trade_log, getattr(self, 'paper_timeout_mins', 120))
                # Compute candidate TP/SL for both directions
                bb_48_levels = (bb_analysis.get('bb_levels') or {}).get('bb_48', {})
                tp_l, sl_l, rr_l = self._compute_tp_sl(live_price, bb_48_levels, up_pct, dn_pct, 'LONG')
                tp_s, sl_s, rr_s = self._compute_tp_sl(live_price, bb_48_levels, up_pct, dn_pct, 'SHORT')
                import datetime as _dt
                now_iso = _dt.datetime.utcnow().isoformat() + 'Z'
                open_trade = self._paper_engine.load_open_trade()
                # Close logic for existing trade
                if open_trade is not None:
                    try:
                        dirn = str(open_trade.get('direction', ''))
                        trade_id = str(open_trade.get('trade_id'))
                        entry_time = pd.to_datetime(open_trade.get('open_time'), utc=True)
                        timeout_minutes = int(getattr(self, 'paper_timeout_mins', 120))
                        deadline = entry_time + pd.Timedelta(minutes=timeout_minutes)
                        now_utc = pd.Timestamp.now(tz='UTC')
                        if dirn == 'LONG':
                            if live_price >= float(open_trade.get('tp_price')):
                                self._paper_engine.close_trade(trade_id, now_iso, live_price, 'TP')
                                self.console.print(f"[green]📗 PAPER: CLOSE LONG #{trade_id} at ${live_price:.2f} (TP)")
                            elif live_price <= float(open_trade.get('sl_price')):
                                self._paper_engine.close_trade(trade_id, now_iso, live_price, 'SL')
                                self.console.print(f"[red]📕 PAPER: CLOSE LONG #{trade_id} at ${live_price:.2f} (SL)")
                            elif now_utc >= deadline:
                                self._paper_engine.close_trade(trade_id, now_iso, live_price, 'TIMEOUT')
                                self.console.print(f"[yellow]⏱️  PAPER: CLOSE LONG #{trade_id} at ${live_price:.2f} (TIMEOUT)")
                        elif dirn == 'SHORT':
                            if live_price <= float(open_trade.get('tp_price')):
                                self._paper_engine.close_trade(trade_id, now_iso, live_price, 'TP')
                                self.console.print(f"[green]📗 PAPER: CLOSE SHORT #{trade_id} at ${live_price:.2f} (TP)")
                            elif live_price >= float(open_trade.get('sl_price')):
                                self._paper_engine.close_trade(trade_id, now_iso, live_price, 'SL')
                                self.console.print(f"[red]📕 PAPER: CLOSE SHORT #{trade_id} at ${live_price:.2f} (SL)")
                            elif now_utc >= deadline:
                                self._paper_engine.close_trade(trade_id, now_iso, live_price, 'TIMEOUT')
                                self.console.print(f"[yellow]⏱️  PAPER: CLOSE SHORT #{trade_id} at ${live_price:.2f} (TIMEOUT)")
                    except Exception as _e:
                        self.logger.warning(f"Paper close check failed: {_e}")

                # Entry logic if no open trade
                open_trade = self._paper_engine.load_open_trade()
                if open_trade is None:
                    # Determine entry direction based on resolved signal + RR gate
                    prefer_rr_active = bool(getattr(self, 'prefer_rr_mode', False)) and (float(confidence or 0.0) <= float(getattr(self, 'prefer_rr_max_ml_conf', 0.9)))
                    prefer_dir = None
                    if prefer_rr_active and (rr_long is not None or rr_short is not None):
                        if long_ok and not short_ok:
                            prefer_dir = 'LONG'
                        elif short_ok and not long_ok:
                            prefer_dir = 'SHORT'
                        elif long_ok and short_ok:
                            prefer_dir = 'LONG' if (rr_long or 0) >= (rr_short or 0) else 'SHORT'
                    # Choose direction
                    dir_to_open = None
                    if resolved_pos == 'LONG' and long_ok:
                        dir_to_open = 'LONG'
                    elif resolved_pos == 'SHORT' and short_ok:
                        dir_to_open = 'SHORT'
                    elif prefer_dir:
                        dir_to_open = prefer_dir
                    # Basic confidence guard
                    if dir_to_open and (confidence >= 0.6 or prefer_rr_active):
                        trade_id = str(int(time.time()))
                        if dir_to_open == 'LONG':
                            tp_price, sl_price, rr_used = tp_l, sl_l, rr_l
                        else:
                            tp_price, sl_price, rr_used = tp_s, sl_s, rr_s
                        open_payload = {
                            'trade_id': trade_id,
                            'open_time': now_iso,
                            'timeframe': self.timeframe,
                            'symbol': self.symbol,
                            'direction': dir_to_open,
                            'entry_price': live_price,
                            'tp_price': tp_price,
                            'sl_price': sl_price,
                            'rr_at_entry': rr_used,
                            'rr_min': rr_min_val,
                            'ml_confidence': confidence,
                            'resolved_pos': resolved_pos,
                            'prefer_rr_active': prefer_rr_active,
                        }
                        self._paper_engine.open_trade(open_payload)
                        self.console.print(f"[cyan]📝 PAPER: OPEN {dir_to_open} #{trade_id} at ${live_price:.2f} | TP ${tp_price:.2f} SL ${sl_price:.2f}")

            # Paper-trade simulator (optional)
            if getattr(self, 'paper_trade_log', None):
                if self._paper_engine is None:
                    self._paper_engine = self._PaperTradeEngine(self.paper_trade_log, getattr(self, 'paper_timeout_mins', 120))
                # Compute candidate TP/SL for both directions
                bb_48_levels = (bb_analysis.get('bb_levels') or {}).get('bb_48', {})
                tp_l, sl_l, rr_l = self._compute_tp_sl(live_price, bb_48_levels, up_pct, dn_pct, 'LONG')
                tp_s, sl_s, rr_s = self._compute_tp_sl(live_price, bb_48_levels, up_pct, dn_pct, 'SHORT')
                import datetime as _dt
                now_iso = _dt.datetime.utcnow().replace(tzinfo=None).isoformat() + 'Z'
                open_trade = self._paper_engine.load_open_trade()
                # Close logic for existing trade
                if open_trade is not None:
                    try:
                        dirn = str(open_trade.get('direction', ''))
                        trade_id = str(open_trade.get('trade_id'))
                        entry_time = pd.to_datetime(open_trade.get('open_time'), utc=True)
                        timeout_minutes = int(getattr(self, 'paper_timeout_mins', 120))
                        deadline = entry_time + pd.Timedelta(minutes=timeout_minutes)
                        if dirn == 'LONG':
                            if live_price >= float(open_trade.get('tp_price')):
                                self._paper_engine.close_trade(trade_id, now_iso, live_price, 'TP')
                                self.console.print(f"[green]📗 PAPER: CLOSE LONG #{trade_id} at ${live_price:.2f} (TP)")
                            elif live_price <= float(open_trade.get('sl_price')):
                                self._paper_engine.close_trade(trade_id, now_iso, live_price, 'SL')
                                self.console.print(f"[red]📕 PAPER: CLOSE LONG #{trade_id} at ${live_price:.2f} (SL)")
                            elif pd.Timestamp.now(tz='UTC') >= deadline:
                                self._paper_engine.close_trade(trade_id, now_iso, live_price, 'TIMEOUT')
                                self.console.print(f"[yellow]⏱️  PAPER: CLOSE LONG #{trade_id} at ${live_price:.2f} (TIMEOUT)")
                        elif dirn == 'SHORT':
                            if live_price <= float(open_trade.get('tp_price')):
                                self._paper_engine.close_trade(trade_id, now_iso, live_price, 'TP')
                                self.console.print(f"[green]📗 PAPER: CLOSE SHORT #{trade_id} at ${live_price:.2f} (TP)")
                            elif live_price >= float(open_trade.get('sl_price')):
                                self._paper_engine.close_trade(trade_id, now_iso, live_price, 'SL')
                                self.console.print(f"[red]📕 PAPER: CLOSE SHORT #{trade_id} at ${live_price:.2f} (SL)")
                            elif pd.Timestamp.utcnow().tz_localize('UTC') >= deadline:
                                self._paper_engine.close_trade(trade_id, now_iso, live_price, 'TIMEOUT')
                                self.console.print(f"[yellow]⏱️  PAPER: CLOSE SHORT #{trade_id} at ${live_price:.2f} (TIMEOUT)")
                    except Exception as _e:
                        self.logger.warning(f"Paper close check failed: {_e}")
                # Refresh open status after potential close
                open_trade = self._paper_engine.load_open_trade()
                if open_trade is None:
                    # Determine entry direction based on resolved signal + RR gate or prefer-RR
                    prefer_rr_active = bool(getattr(self, 'prefer_rr_mode', False)) and (float(confidence or 0.0) <= float(getattr(self, 'prefer_rr_max_ml_conf', 0.9)))
                    prefer_dir = None
                    if prefer_rr_active and (rr_long is not None or rr_short is not None):
                        if long_ok and not short_ok:
                            prefer_dir = 'LONG'
                        elif short_ok and not long_ok:
                            prefer_dir = 'SHORT'
                        elif long_ok and short_ok:
                            prefer_dir = 'LONG' if (rr_long or 0) >= (rr_short or 0) else 'SHORT'
                    dir_to_open = None
                    if resolved_pos == 'LONG' and long_ok:
                        dir_to_open = 'LONG'
                    elif resolved_pos == 'SHORT' and short_ok:
                        dir_to_open = 'SHORT'
                    elif prefer_dir:
                        dir_to_open = prefer_dir
                    # Basic confidence guard
                    if dir_to_open and (confidence >= 0.6 or prefer_rr_active):
                        trade_id = str(int(time.time()))
                        if dir_to_open == 'LONG':
                            tp_price, sl_price, rr_used = tp_l, sl_l, rr_l
                        else:
                            tp_price, sl_price, rr_used = tp_s, sl_s, rr_s
                        open_payload = {
                            'trade_id': trade_id,
                            'open_time': now_iso,
                            'timeframe': self.timeframe,
                            'symbol': self.symbol,
                            'direction': dir_to_open,
                            'entry_price': live_price,
                            'tp_price': tp_price,
                            'sl_price': sl_price,
                            'rr_at_entry': rr_used,
                            'rr_min': rr_min_val,
                            'ml_confidence': confidence,
                            'resolved_pos': resolved_pos,
                            'prefer_rr_active': prefer_rr_active,
                        }
                        self._paper_engine.open_trade(open_payload)
                        self.console.print(f"[cyan]📝 PAPER: OPEN {dir_to_open} #{trade_id} at ${live_price:.2f} | TP ${tp_price:.2f} SL ${sl_price:.2f}")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ Error in enhanced prediction cycle: {str(e)}[/red]")
            self.logger.error(f"Error in enhanced prediction cycle: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Enhanced Live Crypto Prediction with Bollinger Bands')
    parser.add_argument('--timeframe', choices=['5m', '15m', '30m', '1h', '2h', '4h', '6h'], default='15m',
                       help='Trading timeframe (multi-timeframe support)')
    parser.add_argument('--model', choices=['auto', 'CatBoost', 'XGBoost', 'LightGBM'], default='LightGBM',
                       help='ML model to use (auto = prefer LightGBM > XGBoost > CatBoost)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Update interval in seconds')
    parser.add_argument('--history-rows', type=int, default=300,
                       help='Number of recent rows to load for context/slope analysis')
    parser.add_argument('--confidence-threshold', type=float, default=None,
                       help='If set, predictions below this confidence are forced to HOLD')
    parser.add_argument('--output', choices=['rich', 'json'], default='rich',
                       help='Output mode: rich tables or machine-readable JSON (single mode prints one JSON object)')
    parser.add_argument('--rr-min', type=float, default=None,
                       help='Override reward:risk minimum threshold used for gating (overrides config.multi_horizon.rr_min)')
    parser.add_argument('--market', choices=['spot','future','futures-usdm','usdm'], default=None,
                       help='Override market type (spot atau USD-M futures). Akan menimpa config.data.market_type')
    parser.add_argument('--prefer-rr-mode', action='store_true',
                       help='Prioritaskan arah dengan R:R >= rr_min saat ML masih HOLD dan confidence <= prefer-rr-max-ml-conf')
    parser.add_argument('--prefer-rr-max-ml-conf', type=float, default=0.9,
                       help='Prefer-RR hanya aktif jika ML confidence <= nilai ini (default 0.9)')
    # Paper trade logging & simulator ringan
    parser.add_argument('--paper-trade-log', type=str, default=None,
                       help='Path CSV untuk mencatat sinyal dan simulasi trade ringan (entry via TRIGGER + TP/SL).')
    parser.add_argument('--paper-timeout-mins', type=int, default=120,
                       help='Timeout trade simulasi (menit). Jika tidak kena TP/SL hingga waktu ini, trade ditutup di harga terakhir.')
    parser.add_argument('--single', action='store_true',
                       help='Run single prediction only')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuous predictions (opposite of --single)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    
    # Real-time data fetching arguments
    parser.add_argument('--realtime-fetch', action='store_true',
                       help='Enable real-time data fetching from Binance (override config)')
    parser.add_argument('--no-realtime-fetch', action='store_true',
                       help='Disable real-time fetching, use database only (override config)')
    parser.add_argument('--lookback-days', type=int, default=None,
                       help='Days of historical data to fetch for feature calculation (override config)')
    parser.add_argument('--lookback-candles', type=int, default=None,
                       help='Number of candles to fetch (overrides lookback-days and config)')
                       
    args = parser.parse_args()
    
    # Initialize enhanced prediction system
    system = EnhancedPredictionSystem(
        timeframe=args.timeframe,
        model_type=args.model,
        config_path=args.config
    )
    
    # Apply CLI overrides to system config
    if hasattr(system.config, 'prediction'):
        if args.realtime_fetch:
            system.config.prediction.enable_realtime_fetch = True
        elif args.no_realtime_fetch:
            system.config.prediction.enable_realtime_fetch = False
            
        if args.lookback_days is not None:
            system.config.prediction.lookback_days = args.lookback_days
        
        if args.lookback_candles is not None:
            system.config.prediction.lookback_candles = args.lookback_candles

    # Override market type via CLI
    if args.market is not None:
        try:
            # set di config
            if hasattr(system.config, 'data') and isinstance(system.config.data, dict):
                system.config.data['market_type'] = args.market
            elif hasattr(system.config, 'data'):
                setattr(system.config.data, 'market_type', args.market)
        except Exception:
            pass
        # set di instance
        try:
            system.market_type = args.market
            system._set_binance_api_base()
        except Exception:
            pass

    # Allow CLI override of R:R threshold for gating
    try:
        mh = getattr(system.config, 'multi_horizon', None)
        if args.rr_min is not None:
            if mh is None:
                # Create minimal structure if missing
                try:
                    system.config.multi_horizon = {'rr_min': args.rr_min}
                except Exception:
                    pass
            else:
                try:
                    # dict-like
                    mh['rr_min'] = args.rr_min
                except Exception:
                    try:
                        # attribute-like
                        setattr(mh, 'rr_min', args.rr_min)
                    except Exception:
                        pass
    except Exception:
        pass

    # Apply Prefer-RR mode flags to system instance
    try:
        setattr(system, 'prefer_rr_mode', bool(args.prefer_rr_mode))
        setattr(system, 'prefer_rr_max_ml_conf', float(args.prefer_rr_max_ml_conf))
    except Exception:
        pass

    # Apply paper trade options
    try:
        setattr(system, 'paper_trade_log', args.paper_trade_log)
        setattr(system, 'paper_timeout_mins', int(args.paper_timeout_mins))
    except Exception:
        # Backward compatibility if attr name typo; set correctly
        try:
            setattr(system, 'paper_trade_log', args.paper_trade_log)
            setattr(system, 'paper_timeout_mins', int(getattr(args, 'paper_timeout_mins', 120)))
        except Exception:
            pass
    
    if not system.model:
        print("❌ Failed to load model. Exiting...")
        sys.exit(1)
        
    # Run prediction
    # Determine execution mode (continuous has precedence if both flags provided)
    if args.continuous and args.single:
        print("[i] Both --single and --continuous provided; running in continuous mode.")

    if args.continuous or not args.single:
        # Continuous mode (default if neither single nor continuous specified)
        try:
            print(f"🔄 Starting continuous mode (updates every {args.interval}s)")
            print("Press Ctrl+C to stop...")
            print()
            
            while True:
                success = system.run_single_prediction(history_rows=args.history_rows,
                                                       confidence_threshold=args.confidence_threshold,
                                                       output_mode=args.output,
                                                       lookback_days=args.lookback_days,
                                                       lookback_candles=args.lookback_candles)
                
                if success:
                    print(f"\n✅ Prediction completed. Next update in {args.interval}s")
                else:
                    print(f"\n❌ Prediction failed. Retrying in {args.interval}s")
                
                # Show countdown
                remaining = args.interval
                while remaining > 0:
                    step = min(10, remaining)
                    print(f"⏱️  Next update in {remaining}s...", end='\r')
                    time.sleep(step)
                    remaining -= step
                print(" " * 40, end='\r')  # Clear countdown
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping enhanced prediction system...")
    else:
        system.run_single_prediction(history_rows=args.history_rows,
                                     confidence_threshold=args.confidence_threshold,
                                     output_mode=args.output,
                                     lookback_days=args.lookback_days,
                                     lookback_candles=args.lookback_candles)

if __name__ == "__main__":
    main()