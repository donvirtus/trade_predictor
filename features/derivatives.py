import pandas as pd
import numpy as np
from typing import Any, Dict, Iterable


def add_derivative_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    new_cols: Dict[str, pd.Series] = {}
    # Original lagged features
    for lag in cfg.lagged_periods:
        new_cols[f'close_lag_{lag}'] = df['close'].shift(lag)
        for p in cfg.ma_periods:
            ma_col = f'ma_{p}'
            if ma_col in df.columns:
                new_cols[f'{ma_col}_lag_{lag}'] = df[ma_col].shift(lag)
        rsi_col = f'rsi_{cfg.rsi_period}'
        if rsi_col in df.columns:
            new_cols[f'{rsi_col}_lag_{lag}'] = df[rsi_col].shift(lag)
    # Original ratio features
    for p in cfg.ma_periods:
        ma_col = f'ma_{p}'
        if ma_col in df.columns:
            new_cols[f'close_to_{ma_col}'] = _safe_ratio(df['close'], df[ma_col])
    if 'vwap' in df.columns:
        new_cols['close_to_vwap'] = _safe_ratio(df['close'], df['vwap'])
    for period in cfg.bb_periods:
        mid = f'bb_{period}_middle'
        if mid in df.columns:
            new_cols[f'close_to_{mid}'] = _safe_ratio(df['close'], df[mid])
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        new_cols['macd_diff'] = df['macd'] - df['macd_signal']

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Event-aware MA ↔︎ Bollinger features
    df = add_ma_bb_crossover_features(df, cfg)
    # Detailed Bollinger slope / expansion diagnostics
    df = add_bb_slope_features(df, cfg)
    return df


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    result = numerator.divide(denom)
    return result.replace([np.inf, -np.inf], np.nan)


def _sanitize_dev(dev: float) -> str:
    key = str(dev).replace('-', 'm')
    return key.replace('.', '_')


def _calculate_linear_regression_slope(series: pd.Series, window: int) -> pd.Series:
    """Calculate linear regression slope over rolling window for smoother trend detection"""
    def lr_slope(y):
        if len(y) < 2 or y.isna().all():
            return np.nan
        x = np.arange(len(y))
        valid_mask = ~y.isna()
        if valid_mask.sum() < 2:
            return np.nan
        x_valid, y_valid = x[valid_mask], y[valid_mask]
        if len(x_valid) < 2:
            return np.nan
        # Linear regression: slope = (n*Σxy - ΣxΣy) / (n*Σx² - (Σx)²)
        n = len(x_valid)
        sum_x, sum_y = x_valid.sum(), y_valid.sum()
        sum_xy = (x_valid * y_valid).sum()
        sum_x2 = (x_valid ** 2).sum()
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return np.nan
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    return series.rolling(window=window, min_periods=2).apply(lr_slope, raw=False)


def _compute_bars_since(trigger: pd.Series, limit: int | None = None) -> pd.Series:
    if trigger.empty:
        return trigger.copy()
    trig = trigger.fillna(0).astype(int)
    indices = np.arange(len(trig), dtype=float)
    hit_idx = np.where(trig.values > 0, indices, np.nan)
    last_hit = pd.Series(hit_idx, index=trig.index).ffill().values
    bars = pd.Series(indices - last_hit, index=trig.index)
    mask = np.isnan(last_hit)
    if limit is not None:
        bars = bars.clip(upper=limit)
        bars[mask] = float(limit)
    else:
        bars[mask] = np.nan
    return bars


def _bool_to_int(series: pd.Series) -> pd.Series:
    return series.astype(int).astype('Int8').fillna(0)


def _get_cross_config(cfg) -> Dict[str, Any]:
    cross_cfg = getattr(cfg, 'ma_bb_cross', {}) or {}
    if not isinstance(cross_cfg, dict):
        cross_cfg = dict(cross_cfg)
    return cross_cfg


def add_ma_bb_crossover_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    cross_cfg = _get_cross_config(cfg)
    if not cross_cfg.get('enabled', True):
        return df

    lookbacks: Iterable[int] = cross_cfg.get('lookbacks', [1, 3, 5]) or []
    lookbacks = [lb for lb in lookbacks if isinstance(lb, int) and lb > 0]
    distance_mode = str(cross_cfg.get('distance_mode', 'ratio')).lower()
    include_middle = bool(cross_cfg.get('include_middle', True))
    include_percent_b = bool(cross_cfg.get('include_percent_b', True))
    since_limit = cross_cfg.get('since_cross_limit')
    since_limit = int(since_limit) if isinstance(since_limit, (int, float)) and since_limit > 0 else None
    slope_window = max(1, int(cross_cfg.get('slope_window', 5)))
    slope_threshold = float(cross_cfg.get('slope_threshold', 0.001))

    ma_periods = getattr(cfg, 'ma_periods', []) or []
    bb_periods = getattr(cfg, 'bb_periods', []) or []
    bb_devs = getattr(cfg, 'bb_devs', []) or []
    if not ma_periods or not bb_periods or not bb_devs:
        return df

    period_levels: Dict[int, Dict[str, pd.Series]] = {}
    period_slopes: Dict[int, Dict[str, pd.Series]] = {}
    period_meta: Dict[int, Dict[str, Any]] = {}

    for bb_period in bb_periods:
        level_map: Dict[str, pd.Series] = {}
        meta_map: Dict[str, Any] = {}
        middle_col = f'bb_{bb_period}_middle'
        if include_middle and middle_col in df.columns:
            level_map['middle'] = df[middle_col]
            meta_map['middle'] = {'position': 'middle', 'dev': None, 'dev_key': 'middle'}
        for dev in bb_devs:
            dev_key = _sanitize_dev(dev)
            upper_col = f'bb_{bb_period}_upper_{dev}'
            lower_col = f'bb_{bb_period}_lower_{dev}'
            if upper_col in df.columns:
                level_map[f'upper_{dev_key}'] = df[upper_col]
                meta_map[f'upper_{dev_key}'] = {'position': 'upper', 'dev': dev, 'dev_key': dev_key}
            if lower_col in df.columns:
                level_map[f'lower_{dev_key}'] = df[lower_col]
                meta_map[f'lower_{dev_key}'] = {'position': 'lower', 'dev': dev, 'dev_key': dev_key}
        if not level_map:
            continue
        period_levels[bb_period] = level_map
        # ✅ NORMALIZED BB SLOPES (scale-invariant)
        period_slopes[bb_period] = {
            level_name: _safe_ratio(series - series.shift(slope_window), series.shift(slope_window).abs()) * 100
            for level_name, series in level_map.items()
        }
        period_meta[bb_period] = meta_map

    if not period_levels:
        return df

    new_cols: Dict[str, pd.Series] = {}

    for ma_period in ma_periods:
        ma_col = f'ma_{ma_period}'
        if ma_col not in df.columns:
            continue
        ma_series = df[ma_col]
        ma_prev = ma_series.shift(1)
        
        # ✅ NORMALIZED SLOPE CALCULATIONS (scale-invariant)
        ma_base = ma_series.shift(slope_window)
        ma_slope_pct = _safe_ratio(ma_series - ma_base, ma_base.abs()) * 100  # Percentage change
        ma_slope_raw = (ma_series - ma_base) / slope_window  # Raw for backward compatibility
        
        # Linear regression slope for smoother trend detection
        ma_slope_lr = _calculate_linear_regression_slope(ma_series, slope_window)
        slope_pct = ma_slope_pct  # Use percentage-based as primary

        new_cols[f'{ma_col}_slope_pct_w{slope_window}'] = ma_slope_pct  # Primary: percentage-based
        new_cols[f'{ma_col}_slope_raw_w{slope_window}'] = ma_slope_raw  # Backward compatibility
        new_cols[f'{ma_col}_slope_lr_w{slope_window}'] = ma_slope_lr    # Linear regression trend

        # ✅ PERCENTAGE-BASED SLOPE THRESHOLDS (more meaningful)
        pct_threshold = slope_threshold * 100  # Convert to percentage
        ma_up = (ma_slope_pct > pct_threshold).fillna(False)
        ma_down = (ma_slope_pct < -pct_threshold).fillna(False)

        for bb_period, level_map in period_levels.items():
            meta_map = period_meta.get(bb_period, {})
            for level_name, level_series in level_map.items():
                level_prev = level_series.shift(1)
                level_slope = period_slopes[bb_period][level_name]
                meta = meta_map.get(level_name, {})

                base_label = f'{ma_col}_bb_{bb_period}_{level_name}'

                if distance_mode in ('ratio', 'both'):
                    new_cols[f'{base_label}_ratio'] = _safe_ratio(ma_series, level_series)
                    new_cols[f'{base_label}_offset_ratio'] = _safe_ratio(ma_series - level_series, level_series.abs())
                if distance_mode in ('delta', 'both'):
                    new_cols[f'{base_label}_delta'] = ma_series - level_series
                
                # ✅ BB-NORMALIZED DISTANCE (seperti %B tapi untuk MA position)
                # Implementasi logika user: "jarak relatif MA ke band"
                if meta.get('dev') is not None:
                    dev_value = meta['dev']
                    dev_key = meta['dev_key']
                    # Cari upper dan lower bounds untuk normalisasi
                    upper_bound_col = f'bb_{bb_period}_upper_{dev_value}'
                    lower_bound_col = f'bb_{bb_period}_lower_{dev_value}'
                    if upper_bound_col in df.columns and lower_bound_col in df.columns:
                        upper_bound = df[upper_bound_col]
                        lower_bound = df[lower_bound_col]
                        # MA position dalam BB range (0-1 scale, seperti %B)
                        ma_bb_position = _safe_ratio(ma_series - lower_bound, upper_bound - lower_bound)
                        new_cols[f'{base_label}_bb_position'] = ma_bb_position
                        
                        # Distance to target levels (sesuai logika user)
                        middle_col = f'bb_{bb_period}_middle'
                        if middle_col in df.columns:
                            middle_series = df[middle_col]
                            # Jarak ke middle band (normalized)
                            dist_to_middle_norm = _safe_ratio(ma_series - middle_series, upper_bound - lower_bound)
                            new_cols[f'{base_label}_dist_to_middle_norm'] = dist_to_middle_norm
                
                # ✅ MIDDLE BAND NORMALIZED FEATURES (untuk level_name == 'middle')
                elif meta.get('position') == 'middle':
                    # Untuk middle band, gunakan widest BB deviation untuk normalisasi
                    widest_dev = max(bb_devs) if bb_devs else 2.0
                    upper_bound_col = f'bb_{bb_period}_upper_{widest_dev}'
                    lower_bound_col = f'bb_{bb_period}_lower_{widest_dev}'
                    if upper_bound_col in df.columns and lower_bound_col in df.columns:
                        upper_bound = df[upper_bound_col]
                        lower_bound = df[lower_bound_col]
                        # MA position relative to widest BB range
                        ma_bb_position = _safe_ratio(ma_series - lower_bound, upper_bound - lower_bound)
                        new_cols[f'{base_label}_bb_position_wide'] = ma_bb_position
                        
                        # Distance from MA to middle (should be 0 for perfect alignment)
                        dist_to_middle_norm = _safe_ratio(ma_series - level_series, upper_bound - lower_bound)
                        new_cols[f'{base_label}_ma_middle_deviation'] = dist_to_middle_norm

                ma_ge = (ma_series >= level_series).fillna(False)
                ma_le = (ma_series <= level_series).fillna(False)
                prev_lt = (ma_prev < level_prev).fillna(False)
                prev_gt = (ma_prev > level_prev).fillna(False)

                cross_up = _bool_to_int(ma_ge & prev_lt)
                cross_down = _bool_to_int(ma_le & prev_gt)

                cross_base = f'{base_label}_cross'
                new_cols[f'{cross_base}_up'] = cross_up
                new_cols[f'{cross_base}_down'] = cross_down

                for lb in lookbacks:
                    roll_params = dict(window=lb, min_periods=1)
                    new_cols[f'{cross_base}_up_recent_{lb}'] = cross_up.rolling(**roll_params).max()
                    new_cols[f'{cross_base}_down_recent_{lb}'] = cross_down.rolling(**roll_params).max()

                if since_limit is not None:
                    new_cols[f'{cross_base}_up_bars_since'] = _compute_bars_since(cross_up, since_limit)
                    new_cols[f'{cross_base}_down_bars_since'] = _compute_bars_since(cross_down, since_limit)
                else:
                    new_cols[f'{cross_base}_up_bars_since'] = _compute_bars_since(cross_up)
                    new_cols[f'{cross_base}_down_bars_since'] = _compute_bars_since(cross_down)

                # Momentum around crossing
                momentum = (ma_series - ma_prev) - (level_series - level_prev)
                new_cols[f'{cross_base}_momentum'] = momentum

                # Slope alignment vs divergence (menggunakan percentage slopes)  
                level_up = (level_slope > pct_threshold).fillna(False)
                level_down = (level_slope < -pct_threshold).fillna(False)
                alignment = pd.Series(
                    np.select(
                        [ma_up & level_up, ma_down & level_down],
                        [1, -1],
                        default=0
                    ),
                    index=df.index
                )
                divergence = pd.Series(
                    np.select(
                        [ma_up & level_down, ma_down & level_up],
                        [1, -1],
                        default=0
                    ),
                    index=df.index
                )
                new_cols[f'{base_label}_slope_alignment'] = alignment
                new_cols[f'{base_label}_slope_divergence'] = divergence
                new_cols[f'{base_label}_slope_diff'] = ma_slope_pct - level_slope
                
                # ✅ SEQUENTIAL CROSSING LOGIC (sesuai request user)
                # Implementasi: \"MA 4 dari posisi 'dekat dengan BB 24 deviasi +2', mengarah turun, menembus deviasi +1\"
                if meta.get('position') and meta.get('dev') is not None:
                    dev_value = meta['dev']
                    position = meta['position']  # 'upper', 'lower', 'middle'
                    
                    # Crossing direction flags dengan slope context
                    cross_with_slope_up = cross_up & ma_up  # MA naik DAN crossing up
                    cross_with_slope_down = cross_down & ma_down  # MA turun DAN crossing down
                    cross_against_slope_up = cross_up & ma_down  # MA turun TAPI crossing up (weak signal)
                    cross_against_slope_down = cross_down & ma_up  # MA naik TAPI crossing down (weak signal)
                    
                    new_cols[f'{cross_base}_slope_confirmed_up'] = _bool_to_int(cross_with_slope_up)
                    new_cols[f'{cross_base}_slope_confirmed_down'] = _bool_to_int(cross_with_slope_down)
                    new_cols[f'{cross_base}_slope_against_up'] = _bool_to_int(cross_against_slope_up)
                    new_cols[f'{cross_base}_slope_against_down'] = _bool_to_int(cross_against_slope_down)
                    
                    # Probabilistic target prediction (sesuai logika user)
                    # \"crossing dari +1 turun, probabilitas 70% capai -1\"
                    if position == 'upper' and dev_value == 1.0:  # Dari upper +1
                        # Target: middle band probability
                        target_prob_middle = cross_down & ma_down  # Confirmed downward
                        new_cols[f'{cross_base}_target_middle_prob'] = _bool_to_int(target_prob_middle)
                    elif position == 'lower' and dev_value == 1.0:  # Dari lower -1  
                        # Target: middle band probability
                        target_prob_middle = cross_up & ma_up  # Confirmed upward
                        new_cols[f'{cross_base}_target_middle_prob'] = _bool_to_int(target_prob_middle)

                if include_percent_b and meta.get('dev') is not None:
                    dev_value = meta['dev']
                    percent_col = f'bb_{bb_period}_percent_b_{dev_value}'
                    if percent_col in df.columns:
                        percent_key = _sanitize_dev(dev_value)
                        new_cols[f'{cross_base}_percentb_{percent_key}'] = df[percent_col]

    if not new_cols:
        return df

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_bb_slope_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    slope_cfg = getattr(cfg, 'bb_slope', {}) or {}
    if not isinstance(slope_cfg, dict):
        slope_cfg = dict(slope_cfg)
    if not slope_cfg.get('enabled', True):
        return df

    window = max(1, int(slope_cfg.get('window', 5)))
    normalize = bool(slope_cfg.get('normalize', True))
    threshold = float(slope_cfg.get('threshold', 0.001))
    expansion_thr = float(slope_cfg.get('expansion_threshold', threshold))

    bb_periods = getattr(cfg, 'bb_periods', []) or []
    bb_devs = getattr(cfg, 'bb_devs', []) or []
    if not bb_periods or not bb_devs:
        return df

    new_cols: Dict[str, pd.Series] = {}

    for period in bb_periods:
        middle_col = f'bb_{period}_middle'
        if middle_col not in df.columns:
            continue
        middle_series = df[middle_col]
        
        # ✅ NORMALIZED BB MIDDLE SLOPE (percentage-based, scale-invariant)
        middle_base = middle_series.shift(window)
        middle_slope_pct = _safe_ratio(middle_series - middle_base, middle_base.abs()) * 100
        middle_slope_raw = (middle_series - middle_base) / window  # Backward compatibility
        middle_slope_lr = _calculate_linear_regression_slope(middle_series, window)
        
        new_cols[f'bb_{period}_middle_slope_pct'] = middle_slope_pct  # Primary: percentage-based
        new_cols[f'bb_{period}_middle_slope_raw'] = middle_slope_raw  # Backward compatibility
        new_cols[f'bb_{period}_middle_slope_lr'] = middle_slope_lr    # Linear regression trend
        
        if normalize:
            new_cols[f'bb_{period}_middle_slope_norm'] = middle_slope_pct  # Same as pct for middle
            
        # ✅ PERCENTAGE-BASED THRESHOLD (more meaningful than raw values)
        pct_threshold = threshold * 100
        new_cols[f'bb_{period}_middle_slope_dir'] = np.select(
            [middle_slope_pct > pct_threshold, middle_slope_pct < -pct_threshold],
            [1, -1],
            default=0
        )

        for dev in bb_devs:
            dev_key = _sanitize_dev(dev)
            upper_col = f'bb_{period}_upper_{dev}'
            lower_col = f'bb_{period}_lower_{dev}'

            upper_slope_series = None
            lower_slope_series = None

            if upper_col in df.columns:
                upper_series = df[upper_col]
                # ✅ NORMALIZED UPPER BAND SLOPE
                upper_base = upper_series.shift(window)
                upper_slope_pct = _safe_ratio(upper_series - upper_base, upper_base.abs()) * 100
                upper_slope_raw = (upper_series - upper_base) / window
                
                new_cols[f'bb_{period}_upper_{dev_key}_slope_pct'] = upper_slope_pct
                new_cols[f'bb_{period}_upper_{dev_key}_slope_raw'] = upper_slope_raw
                upper_slope_series = upper_slope_pct  # Use percentage as primary
                
                if normalize:
                    new_cols[f'bb_{period}_upper_{dev_key}_slope_norm'] = upper_slope_pct

            if lower_col in df.columns:
                lower_series = df[lower_col]
                # ✅ NORMALIZED LOWER BAND SLOPE  
                lower_base = lower_series.shift(window)
                lower_slope_pct = _safe_ratio(lower_series - lower_base, lower_base.abs()) * 100
                lower_slope_raw = (lower_series - lower_base) / window
                
                new_cols[f'bb_{period}_lower_{dev_key}_slope_pct'] = lower_slope_pct
                new_cols[f'bb_{period}_lower_{dev_key}_slope_raw'] = lower_slope_raw
                lower_slope_series = lower_slope_pct  # Use percentage as primary
                
                if normalize:
                    new_cols[f'bb_{period}_lower_{dev_key}_slope_norm'] = lower_slope_pct

            if upper_slope_series is not None and lower_slope_series is not None:
                upper_series = df[upper_col]
                lower_series = df[lower_col]
                # ✅ NORMALIZED BANDWIDTH CALCULATIONS
                bandwidth = upper_series - lower_series
                bandwidth_label = f'bb_{period}_bandwidth_{dev_key}'
                if normalize:
                    # Bandwidth as percentage of middle price (scale-invariant)
                    new_cols[bandwidth_label] = _safe_ratio(bandwidth, middle_series.abs()) * 100
                else:
                    new_cols[bandwidth_label] = bandwidth

                # ✅ BANDWIDTH SLOPE (percentage-based expansion/contraction)
                bandwidth_base = bandwidth.shift(window)
                bw_slope_pct = _safe_ratio(bandwidth - bandwidth_base, bandwidth_base.abs()) * 100
                bw_slope_raw = (bandwidth - bandwidth_base) / window
                
                new_cols[f'{bandwidth_label}_slope_pct'] = bw_slope_pct  # Primary
                new_cols[f'{bandwidth_label}_slope_raw'] = bw_slope_raw  # Backup
                
                if normalize:
                    new_cols[f'{bandwidth_label}_slope_norm'] = bw_slope_pct

                # ✅ EXPANSION FLAGS dengan percentage threshold
                pct_expansion_thr = expansion_thr * 100
                new_cols[f'{bandwidth_label}_expansion_flag'] = np.select(
                    [bw_slope_pct > pct_expansion_thr, bw_slope_pct < -pct_expansion_thr],
                    [1, -1],
                    default=0
                )

                squeeze_window = max(window * 4, 10)
                rolling_q = bandwidth.rolling(squeeze_window, min_periods=1).quantile(0.2)
                new_cols[f'{bandwidth_label}_squeeze_flag'] = (bandwidth <= rolling_q).astype(int)

                new_cols[f'bb_{period}_upper_lower_{dev_key}_slope_diff'] = upper_slope_series - lower_slope_series

    if not new_cols:
        return df

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
