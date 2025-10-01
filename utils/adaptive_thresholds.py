"""
Adaptive Threshold Auto-Calculation Module

Menghitung multipliers optimal berdasarkan data volatility historis
untuk adaptive threshold system dalam multi-timeframe trading.

Author: AI Assistant
Created: 2025-09-29
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

from utils.path_utils import generate_db_path, resolve_dataset_base_dir, resolve_feature_table


class AdaptiveThresholdCalculator:
    """
    Kelas untuk menghitung optimal threshold multipliers berdasarkan 
    volatility patterns dari historical market data.
    """
    
    def __init__(
        self,
        pair: str = "BTCUSDT",
        config=None,
        db_base_path: Optional[str] = None,
        cache_file: Optional[str] = None,
        feature_table: Optional[str] = None,
    ):
        self.pair = pair
        self.config = config
        self.db_base_path = resolve_dataset_base_dir(config, override=db_base_path)
        self.feature_table = feature_table or resolve_feature_table(config)
        # Generate pair-specific cache file if not provided
        if cache_file is None:
            pair_clean = pair.replace('USDT', '').lower()
            cache_file = f"data/adaptive_multipliers_cache_{pair_clean}.json"
        self.cache_file = cache_file
        self.volatility_cache = {}
        
    def load_timeframe_data(self, pair: str, timeframe: str, lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Load historical data untuk timeframe tertentu dari database
        
        Args:
            pair: Trading pair seperti 'BTCUSDT', 'ETHUSDT'
            timeframe: Timeframe seperti '5m', '1h', '4h'
            lookback_days: Jumlah hari historical data yang dimuat
            
        Returns:
            DataFrame dengan OHLCV data atau None jika gagal
        """
        try:
            # Generate database path using shared helper
            db_path = generate_db_path(pair, timeframe, base_dir=self.db_base_path)
            
            if not os.path.exists(db_path):
                logger.debug(f"Database tidak ditemukan: {db_path}")
                return None
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d')
            
            conn = sqlite3.connect(db_path)
            query = f"""
                SELECT timestamp, open, high, low, close, volume 
                FROM {self.feature_table}
                WHERE timestamp >= ? 
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(cutoff_str,), parse_dates=['timestamp'])
            conn.close()
            
            if df.empty:
                logger.warning(f"Tidak ada data untuk {timeframe} dalam {lookback_days} hari terakhir")
                return None
                
            logger.info(f"Loaded {len(df)} rows data untuk {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data untuk {timeframe}: {e}")
            return None
    
    def calculate_volatility_metrics(self, df: pd.DataFrame, method: str = "rolling_std", 
                                   window: int = 20, annualize: bool = True) -> Dict[str, float]:
        """
        Hitung berbagai volatility metrics dari price data
        
        Args:
            df: DataFrame dengan kolom OHLCV
            method: Method perhitungan ('rolling_std', 'atr', 'parkinson')  
            window: Rolling window size
            annualize: Whether to annualize volatility
            
        Returns:
            Dict dengan volatility metrics
        """
        try:
            if df.empty or 'close' not in df.columns:
                return {}
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            if method == "rolling_std":
                # Standard deviation of returns
                vol = returns.rolling(window=window).std()
                if annualize:
                    # Approximate annualization (varies by timeframe)
                    periods_per_year = self._get_periods_per_year(window)
                    vol = vol * np.sqrt(periods_per_year)
                    
            elif method == "atr":
                # Average True Range based volatility
                if all(col in df.columns for col in ['high', 'low', 'close']):
                    tr1 = df['high'] - df['low']
                    tr2 = np.abs(df['high'] - df['close'].shift())
                    tr3 = np.abs(df['low'] - df['close'].shift())
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    vol = true_range.rolling(window=window).mean() / df['close']
                else:
                    logger.warning("ATR method membutuhkan kolom high, low, close")
                    return {}
                    
            elif method == "parkinson":
                # Parkinson volatility estimator (high-low based)  
                if all(col in df.columns for col in ['high', 'low']):
                    log_hl = np.log(df['high'] / df['low'])
                    vol = np.sqrt(log_hl.rolling(window=window).mean() / (4 * np.log(2)))
                    if annualize:
                        periods_per_year = self._get_periods_per_year(window)
                        vol = vol * np.sqrt(periods_per_year)
                else:
                    logger.warning("Parkinson method membutuhkan kolom high, low")
                    return {}
            else:
                logger.error(f"Unknown volatility method: {method}")
                return {}
            
            # Calculate metrics
            vol_clean = vol.dropna()
            if vol_clean.empty:
                return {}
            
            metrics = {
                'mean': float(vol_clean.mean()),
                'median': float(vol_clean.median()),
                'std': float(vol_clean.std()),
                'recent': float(vol_clean.iloc[-10:].mean()),  # Last 10 periods average
                'percentile_25': float(vol_clean.quantile(0.25)),
                'percentile_75': float(vol_clean.quantile(0.75)),
                'count': len(vol_clean)
            }
            
            logger.debug(f"Volatility metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return {}
    
    def _get_periods_per_year(self, window: int) -> int:
        """
        Estimate periods per year untuk annualization
        (Approximation - actual values depend on timeframe)
        """
        # This is a rough approximation - could be made more precise
        # by parsing timeframe string
        return 252  # Trading days per year (default)
    
    def calculate_optimal_multipliers(self, pair: str, timeframes: List[str], base_timeframe: str = "1h",
                                    lookback_days: int = 30, method: str = "rolling_std",
                                    window: int = 20, min_multiplier: float = 0.2,
                                    max_multiplier: float = 2.5, balance_factor: float = 0.5) -> Dict[str, float]:
        """
        Hitung optimal multipliers berdasarkan volatility ratios dengan balanced approach
        
        Args:
            pair: Trading pair seperti 'BTCUSDT', 'ETHUSDT'
            timeframes: List timeframes untuk dihitung
            base_timeframe: Reference timeframe (multiplier = 1.0)
            lookback_days: Historical data lookback
            method: Volatility calculation method
            window: Rolling window
            min_multiplier: Minimum multiplier value
            max_multiplier: Maximum multiplier value
            balance_factor: Balance between inverse vol (0.0) and timeframe scaling (1.0)
            
        Returns:
            Dict dengan multipliers per timeframe
        """
        try:
            logger.info(f"Menghitung optimal multipliers untuk {pair} {len(timeframes)} timeframes")
            
            volatilities = {}
            
            # Calculate volatility untuk setiap timeframe
            for tf in timeframes:
                df = self.load_timeframe_data(pair, tf, lookback_days)
                if df is not None:
                    vol_metrics = self.calculate_volatility_metrics(df, method, window)
                    if vol_metrics and 'recent' in vol_metrics:
                        # Gunakan recent volatility (weighted average last 10 periods)
                        volatilities[tf] = vol_metrics['recent']
                        logger.info(f"Volatility {tf}: {vol_metrics['recent']:.6f}")
                    else:
                        logger.warning(f"Gagal menghitung volatility untuk {tf}")
                else:
                    logger.warning(f"Gagal memuat data untuk {tf}")
            
            if not volatilities:
                logger.error("Tidak ada volatility data yang berhasil dihitung")
                return {}
            
            # Pastikan base_timeframe ada
            if base_timeframe not in volatilities:
                logger.error(f"Base timeframe {base_timeframe} tidak ditemukan dalam data")
                return {}
            
            base_volatility = volatilities[base_timeframe]
            multipliers = {}
            
            # Define timeframe order dan scaling yang lebih reasonable
            tf_order = {'5m': 0, '15m': 1, '30m': 2, '1h': 3, '2h': 4, '4h': 5, '6h': 6, '12h': 7, '1d': 8}
            base_order = tf_order.get(base_timeframe, 3)  # Default to 1h position
            
            # Calculate multipliers dengan hybrid approach
            for tf in timeframes:
                if tf in volatilities:
                    # Component 1: Inverse volatility (statistical approach)
                    vol_component = base_volatility / volatilities[tf]
                    
                    # Component 2: More reasonable timeframe scaling
                    tf_order_val = tf_order.get(tf, base_order)
                    order_diff = base_order - tf_order_val
                    
                    # Use linear scaling instead of exponential, with dampening
                    # Shorter timeframes get moderate increase, longer get moderate decrease
                    if order_diff > 0:  # Shorter timeframe than base
                        scale_component = 1.0 + (order_diff * 0.15)  # 15% per step up
                    else:  # Longer timeframe than base  
                        scale_component = 1.0 + (order_diff * 0.10)  # 10% per step down
                    
                    # Bound the scaling component reasonably
                    scale_component = max(0.5, min(scale_component, 1.5))
                    
                    # Hybrid combination with bounds checking
                    raw_multiplier = (
                        (1 - balance_factor) * vol_component + 
                        balance_factor * scale_component
                    )
                    
                    # Apply conservative clipping
                    multiplier = np.clip(raw_multiplier, min_multiplier, max_multiplier)
                    multipliers[tf] = round(float(multiplier), 3)
                    
                    logger.info(f"Multiplier {tf}: {multiplier:.3f} "
                               f"(vol: {vol_component:.3f}, scale: {scale_component:.3f}, raw: {raw_multiplier:.3f})")
                else:
                    logger.warning(f"Tidak dapat menghitung multiplier untuk {tf}")
            
            # Normalize base_timeframe ke 1.0 (reference)
            if base_timeframe in multipliers:
                base_mult = multipliers[base_timeframe]
                for tf in multipliers:
                    multipliers[tf] = round(multipliers[tf] / base_mult, 3)
                
                logger.info("Multipliers dinormalisasi dengan base_timeframe = 1.0")
            
            return multipliers
            
        except Exception as e:
            logger.error(f"Error calculating optimal multipliers: {e}")
            return {}
    
    def smooth_multipliers(self, new_multipliers: Dict[str, float], 
                          old_multipliers: Dict[str, float], 
                          smoothing_factor: float = 0.7) -> Dict[str, float]:
        """
        Smooth multipliers dengan EMA untuk stability
        
        Args:
            new_multipliers: Newly calculated multipliers
            old_multipliers: Previous multipliers
            smoothing_factor: EMA weight untuk old values (0.7 = 70% old, 30% new)
            
        Returns:
            Smoothed multipliers
        """
        if not old_multipliers:
            return new_multipliers
        
        smoothed = {}
        for tf in new_multipliers:
            if tf in old_multipliers:
                # EMA smoothing
                smoothed[tf] = round(
                    smoothing_factor * old_multipliers[tf] + 
                    (1 - smoothing_factor) * new_multipliers[tf], 
                    3
                )
            else:
                smoothed[tf] = new_multipliers[tf]
        
        logger.info("Applied EMA smoothing to multipliers")
        return smoothed
    
    def save_multipliers_cache(self, multipliers: Dict[str, float], 
                              metadata: Optional[Dict] = None):
        """
        Save calculated multipliers ke cache file untuk future use
        
        Args:
            multipliers: Calculated multipliers
            metadata: Additional metadata (calculation method, timestamp, etc.)
        """
        try:
            cache_data = {
                'multipliers': multipliers,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Multipliers cache saved to {self.cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving multipliers cache: {e}")
    
    def load_multipliers_cache(self, max_age_hours: int = 24) -> Optional[Dict[str, float]]:
        """
        Load cached multipliers jika masih valid
        
        Args:
            max_age_hours: Maximum age dalam hours untuk consider cache valid
            
        Returns:
            Cached multipliers atau None jika expired/tidak ada
        """
        try:
            if not os.path.exists(self.cache_file):
                return None
            
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check age
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                logger.info(f"Cache expired ({age_hours:.1f}h > {max_age_hours}h)")
                return None
            
            logger.info(f"Loaded cached multipliers (age: {age_hours:.1f}h)")
            return cache_data.get('multipliers', {})
            
        except Exception as e:
            logger.error(f"Error loading multipliers cache: {e}")
            return None


def get_adaptive_multipliers(config, pair: str, timeframes: List[str], force_recalculate: bool = False) -> Dict[str, float]:
    """
    Main function untuk mendapatkan adaptive multipliers
    
    Args:
        config: Configuration object
        pair: Trading pair seperti 'BTCUSDT', 'ETHUSDT'
        timeframes: List of timeframes
        force_recalculate: Force recalculation even if cache valid
        
    Returns:
        Dict dengan multipliers per timeframe
    """
    try:
        # Check configuration
        if not hasattr(config, 'target') or 'adaptive_thresholds' not in config.target:
            logger.warning("Adaptive thresholds tidak dikonfigurasi, menggunakan defaults")
            return {}
        
        adaptive_cfg = config.target['adaptive_thresholds']
        
        # Check if auto-calculation enabled
        method = adaptive_cfg.get('method', 'manual')
        if method == 'manual':
            logger.info("Using manual multipliers dari config")
            return adaptive_cfg.get('manual_multipliers', {})
        
        if method not in ['auto_volatility', 'auto_distribution', 'hybrid']:
            logger.warning(f"Unknown method: {method}, fallback ke manual")
            return adaptive_cfg.get('manual_multipliers', {})
        
        # Auto-calculation parameters
        lookback_days = adaptive_cfg.get('lookback_days', 30)
        base_timeframe = adaptive_cfg.get('base_timeframe', '1h')
        auto_update = adaptive_cfg.get('auto_update', True)

        calc = AdaptiveThresholdCalculator(pair=pair, config=config)
        
        # Try cache jika auto_update enabled
        cached_multipliers = None
        if auto_update and not force_recalculate:
            cache_hours = adaptive_cfg.get('update_frequency_days', 7) * 24
            cached_multipliers = calc.load_multipliers_cache(max_age_hours=cache_hours)
        
        if cached_multipliers:
            logger.info("Using cached adaptive multipliers")
            return cached_multipliers
        
        # Calculate new multipliers
        logger.info("Calculating new adaptive multipliers...")
        
        if method == 'auto_volatility':
            vol_cfg = adaptive_cfg.get('volatility_calc', {})
            new_multipliers = calc.calculate_optimal_multipliers(
                pair=pair,
                timeframes=timeframes,
                base_timeframe=base_timeframe,
                lookback_days=lookback_days,
                method=vol_cfg.get('method', 'rolling_std'),
                window=vol_cfg.get('window', 20),
                min_multiplier=vol_cfg.get('min_multiplier', 0.2),
                max_multiplier=vol_cfg.get('max_multiplier', 2.5),
                balance_factor=vol_cfg.get('balance_factor', 0.3)
            )
            
            # If auto-calculation failed (no databases exist), fall back to manual
            if not new_multipliers:
                logger.warning("Auto-calculation gagal (databases tidak tersedia), fallback ke manual multipliers")
                return adaptive_cfg.get('manual_multipliers', {})
            
            # Apply smoothing if we have previous values
            smoothing_factor = vol_cfg.get('smoothing_factor', 0.7)
            if cached_multipliers:
                new_multipliers = calc.smooth_multipliers(
                    new_multipliers, cached_multipliers, smoothing_factor
                )
        
        elif method == 'hybrid':
            # Hybrid: Start with manual values, adjust with volatility data
            logger.info("Using hybrid approach: manual base + volatility adjustments")
            
            manual_multipliers = adaptive_cfg.get('manual_multipliers', {})
            vol_cfg = adaptive_cfg.get('volatility_calc', {})
            
            # Get auto-calculated multipliers for comparison
            auto_multipliers = calc.calculate_optimal_multipliers(
                pair=pair,
                timeframes=timeframes,
                base_timeframe=base_timeframe,
                lookback_days=lookback_days,
                method=vol_cfg.get('method', 'rolling_std'),
                window=vol_cfg.get('window', 20),
                min_multiplier=0.1,  # Wider range for auto component
                max_multiplier=3.0
            )
            
            # If auto-calculation failed, use pure manual values 
            if not auto_multipliers:
                logger.info("Auto-calculation gagal untuk hybrid, menggunakan manual multipliers")
                return adaptive_cfg.get('manual_multipliers', {})
            
            # Hybrid blend: 70% manual experience + 30% data-driven adjustments
            hybrid_weight = vol_cfg.get('hybrid_weight', 0.3)  # Weight for auto component
            new_multipliers = {}
            
            for tf in timeframes:
                if tf in manual_multipliers:
                    manual_val = manual_multipliers[tf]
                    auto_val = auto_multipliers.get(tf, manual_val)
                    
                    # Blend with constraints to prevent extreme deviations
                    blended = (1 - hybrid_weight) * manual_val + hybrid_weight * auto_val
                    
                    # Constraint: limit deviation from manual values
                    max_deviation = manual_val * 0.5  # Max 50% deviation
                    blended = max(manual_val - max_deviation, 
                                min(manual_val + max_deviation, blended))
                    
                    new_multipliers[tf] = round(blended, 3)
                    
                    logger.info(f"Hybrid {tf}: manual={manual_val:.3f}, auto={auto_val:.3f}, "
                               f"blended={blended:.3f}")
                else:
                    # Use auto value if no manual value available
                    new_multipliers[tf] = auto_multipliers.get(tf, 1.0)

        else:  # auto_distribution method (future implementation)
            logger.warning("auto_distribution method belum diimplementasi, fallback ke manual")
            return adaptive_cfg.get('manual_multipliers', {})
        
        # Fallback ke manual jika calculation gagal
        if not new_multipliers:
            logger.warning("Auto-calculation gagal, menggunakan manual multipliers")
            return adaptive_cfg.get('manual_multipliers', {})
        
        # Save cache
        if auto_update:
            metadata = {
                'method': method,
                'lookback_days': lookback_days,
                'base_timeframe': base_timeframe,
                'timeframes': timeframes
            }
            calc.save_multipliers_cache(new_multipliers, metadata)
        
        logger.info(f"Auto-calculated multipliers: {new_multipliers}")
        return new_multipliers
        
    except Exception as e:
        logger.error(f"Error dalam get_adaptive_multipliers: {e}")
        # Emergency fallback
        return {'5m': 0.3, '15m': 0.6, '1h': 1.0, '2h': 1.4, '4h': 2.0, '6h': 2.5}