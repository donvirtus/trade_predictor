import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def label_future_direction(df: pd.DataFrame, horizon: int, sideways_threshold_pct: float = 1.0) -> pd.DataFrame:
    future_price = df['close'].shift(-horizon)
    pct_change = (future_price - df['close']) / df['close'] * 100
    cond_up = pct_change > sideways_threshold_pct
    cond_down = pct_change < -sideways_threshold_pct
    new_cols = {}
    new_cols['direction'] = np.select([cond_down, cond_up], [0,2], default=1)
    new_cols['future_return_pct'] = pct_change
    return df.assign(**new_cols)


def label_regime_volatility(df: pd.DataFrame, vol_col: str, low_q: float = 0.33, high_q: float = 0.66) -> pd.DataFrame:
    if vol_col not in df.columns:
        return df
    low = df[vol_col].quantile(low_q)
    high = df[vol_col].quantile(high_q)
    new_cols = {}
    new_cols['vol_regime'] = np.select([df[vol_col] < low, df[vol_col] > high], [0,2], default=1)
    return df.assign(**new_cols)


def label_multi_horizon_directions(df: pd.DataFrame, horizons, sideways_threshold_pct: float = 1.0, 
                                   base_col: str = 'close', timeframe: str = None, cfg = None, 
                                   pair: str = 'BTCUSDT') -> pd.DataFrame:
    """Add direction labels for multiple horizons without overwriting existing 'direction'.

    Creates columns:
      - future_return_pct_h{H}
      - direction_h{H} (0=DOWN,1=SIDEWAYS,2=UP)

    Parameters
    ----------
    df : pd.DataFrame
    horizons : Iterable[int]
        List of integer horizons (positive) to label.
    sideways_threshold_pct : float
        Threshold in percent (e.g. 1.0 means ±1%).
    base_col : str
        Column representing the reference price (default: 'close').
    timeframe : str, optional  
        Current timeframe for adaptive threshold adjustment (e.g. '5m', '1h', '4h')
    cfg : optional
        Configuration object containing adaptive_thresholds settings
    """
    if base_col not in df.columns:
        return df
    
    # ✅ ADAPTIVE THRESHOLD berdasarkan timeframe dengan auto-calculation support
    effective_threshold = sideways_threshold_pct
    
    if cfg and hasattr(cfg, 'target') and 'adaptive_thresholds' in cfg.target:
        adaptive_cfg = cfg.target['adaptive_thresholds']
        
        if adaptive_cfg.get('enabled', False) and timeframe:
            try:
                # Import auto-calculation utility
                from utils.adaptive_thresholds import get_adaptive_multipliers
                
                # Get all configured timeframes for multiplier calculation
                all_timeframes = [timeframe]  # Start with current timeframe
                if 'manual_multipliers' in adaptive_cfg:
                    all_timeframes.extend(adaptive_cfg['manual_multipliers'].keys())
                all_timeframes = list(set(all_timeframes))  # Remove duplicates
                
                # Get optimal multipliers (auto-calculated or manual)
                multipliers = get_adaptive_multipliers(cfg, pair, all_timeframes)
                
                if isinstance(multipliers, dict) and timeframe in multipliers:
                    base_threshold = sideways_threshold_pct
                    multiplier = multipliers[timeframe]
                    effective_threshold = base_threshold * multiplier
                    
                    logger.info(f"🎯 Adaptive threshold for {timeframe}: {effective_threshold:.2f}% "
                               f"(base: {base_threshold}%, multiplier: {multiplier}x)")
                else:
                    logger.warning(f"No multiplier found for timeframe {timeframe}, using base threshold")
                    
            except ImportError as e:
                logger.warning(f"Cannot import adaptive_thresholds utility: {e}")
            except Exception as e:
                logger.error(f"Error calculating adaptive threshold: {e}")
    
    new_cols = {}
    for h in sorted(set(int(h) for h in horizons if h > 0)):
        future_price = df[base_col].shift(-h)
        pct_change = (future_price - df[base_col]) / df[base_col] * 100.0
        cond_up = pct_change > effective_threshold  # ✅ Gunakan adaptive threshold
        cond_down = pct_change < -effective_threshold  # ✅ Gunakan adaptive threshold
        dir_col = f'direction_h{h}'
        ret_col = f'future_return_pct_h{h}'
        new_cols[dir_col] = np.select([cond_down, cond_up], [0, 2], default=1)
        new_cols[ret_col] = pct_change
    if not new_cols:
        return df
    return df.assign(**new_cols)


def label_extremes_horizon(df: pd.DataFrame, horizon: int, base_col: str = 'close', high_col: str = 'high', low_col: str = 'low',
                           include_time_to: bool = False) -> pd.DataFrame:
    """Label future extremes (maximum favorable excursion & maximum adverse excursion) within a horizon.

    Adds columns:
      - future_max_high_h{H}: (max(high_{t+1..t+H}) - close_t)/close_t
      - future_min_low_h{H}: (min(low_{t+1..t+H}) - close_t)/close_t
      - (optional) time_to_max_high_h{H}, time_to_min_low_h{H}: first offset index (1..H)

    Notes:
      - Last H rows will have NaN due to insufficient lookahead; caller can trim using an existing rule.
    """
    if any(c not in df.columns for c in [base_col, high_col, low_col]):
        return df
    H = int(horizon)
    if H <= 0:
        return df
    # Build forward-shifted matrices
    high_shifts = [df[high_col].shift(-k) for k in range(1, H + 1)]
    low_shifts = [df[low_col].shift(-k) for k in range(1, H + 1)]
    high_mat = pd.concat(high_shifts, axis=1)
    low_mat = pd.concat(low_shifts, axis=1)
    max_high = high_mat.max(axis=1)
    min_low = low_mat.min(axis=1)
    base_price = df[base_col]
    max_ret = (max_high - base_price) / base_price
    min_ret = (min_low - base_price) / base_price
    new_cols = {
        f'future_max_high_h{H}': max_ret,
        f'future_min_low_h{H}': min_ret
    }
    if include_time_to:
        # Argmax / argmin (first occurrence) +1 to convert 0-based to step count
        time_to_max = high_mat.values.argmax(axis=1) + 1
        time_to_min = low_mat.values.argmin(axis=1) + 1
        # Where all NaN (tail), set NaN
        nan_mask = high_mat.isna().all(axis=1)
        time_to_max = pd.Series(time_to_max, index=df.index).where(~nan_mask, other=pd.NA)
        nan_mask_low = low_mat.isna().all(axis=1)
        time_to_min = pd.Series(time_to_min, index=df.index).where(~nan_mask_low, other=pd.NA)
        new_cols[f'time_to_max_high_h{H}'] = time_to_max
        new_cols[f'time_to_min_low_h{H}'] = time_to_min
    return df.assign(**new_cols)

