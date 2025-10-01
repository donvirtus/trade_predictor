import os, sqlite3, sys
import pandas as pd
from datetime import datetime, timezone
import time
from typing import List, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

# Allow direct execution: add project root to sys.path if modules not found
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.logger_config import get_logger

from utils.config import load_config
from utils.path_utils import (
    generate_db_path,
    resolve_dataset_base_dir,
    resolve_feature_table,
    resolve_metadata_table,
)
from data.binance_fetch import init_exchange, normalize_symbol, fetch_ohlcv, fetch_ohlcv_history, _last_closed_candle
from data.continuity import detect_gaps, backfill_gaps, extend_tail, validate_continuity
from data.external.coingecko import fetch_coingecko_snapshot
from data.external.coinmetrics import fetch_coinmetrics
from data.external.dune import fetch_dune_query
from features.indicators import add_price_indicators
from features.derivatives import add_derivative_features
from features.targets import (
    label_future_direction,
    label_regime_volatility,
    label_multi_horizon_directions,
    label_extremes_horizon,
)


def merge_external(df: pd.DataFrame, cfg, logger):
    df['date'] = df['timestamp'].dt.date
    frames = []
    if cfg.external.get('enable_coingecko', False):
        cg = fetch_coingecko_snapshot(cfg.external.get('coin_id','bitcoin'))
        frames.append(cg)
    if cfg.external.get('enable_coinmetrics', False):
        cm = fetch_coinmetrics(asset=cfg.external.get('coinmetrics_asset','btc'), metrics=tuple(cfg.external.get('coinmetrics_metrics',["AdrActCnt"])) )
        frames.append(cm)
    # Dune (placeholder)
    if cfg.external.get('enable_dune', False):
        for q in cfg.external.get('dune_query_ids', []):
            dq = fetch_dune_query(q)
            frames.append(dq)
    if not frames:
        return df
    ext = frames[0]
    for add in frames[1:]:
        if add is not None and not add.empty:
            ext = ext.merge(add, on='date', how='outer')
    ext.sort_values('date', inplace=True)
    merged = df.merge(ext, on='date', how='left')
    merged.sort_values('timestamp', inplace=True)
    merged.ffill(inplace=True)
    return merged


def save_to_sqlite(df: pd.DataFrame, db_path: str, table: str, mode: str = 'replace'):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists=mode, index=False)


def _compute_indicators_chunk(df_chunk, cfg, chunk_id):
    """Compute technical indicators for a data chunk - for parallel processing"""
    try:
        from features.indicators import add_price_indicators
        # logger.debug(f"Processing indicators chunk {chunk_id} with {len(df_chunk)} rows")
        return add_price_indicators(df_chunk.copy(), cfg), chunk_id
    except Exception as e:
        print(f"Error in indicators chunk {chunk_id}: {e}")
        return df_chunk, chunk_id

def _compute_derivatives_chunk(df_chunk, cfg, chunk_id):
    """Compute derivatives for a data chunk - for parallel processing"""
    try:
        from features.derivatives import add_derivative_features
        # logger.debug(f"Processing derivatives chunk {chunk_id} with {len(df_chunk)} rows")
        return add_derivative_features(df_chunk.copy(), cfg), chunk_id
    except Exception as e:
        print(f"Error in derivatives chunk {chunk_id}: {e}")
        return df_chunk, chunk_id


def fetch_and_process_timeframe(
    ex,
    cfg,
    pair: str,
    timeframe: str,
    logger,
    dataset_base_dir: str,
    feature_table: str,
    workers: Optional[int] = None
) -> pd.DataFrame:
    """Fetch and process data for a specific pair-timeframe combination"""
    start_time = time.time()
    logger.info(f"🚀 Starting {pair} {timeframe} processing pipeline...")
    
    sym = normalize_symbol(ex, pair)
    
    # Determine fetch method
    fetch_cfg = cfg.fetch if hasattr(cfg, 'fetch') else {}
    use_history = False
    if fetch_cfg:
        if fetch_cfg.get('years') or fetch_cfg.get('start') or fetch_cfg.get('end') or fetch_cfg.get('incremental', False):
            use_history = True
    # Force full rebuild overrides incremental/history heuristics
    force_full = getattr(cfg, 'force_full_rebuild', True)
    retain_history = getattr(cfg, 'retain_full_history', False)
    if force_full:
        # We'll still use the history fetcher but with incremental disabled and start derived from months/years
        use_history = True
    
    # Check for existing data if incremental mode
    existing_max = None
    effective_incremental = bool(fetch_cfg.get('incremental', False)) and not force_full
    db_path = generate_db_path(pair, timeframe, base_dir=dataset_base_dir)
    if use_history and effective_incremental and os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                q = f"SELECT max(timestamp) FROM {feature_table}"
                res = conn.execute(q).fetchone()[0]
                if res:
                    existing_max = pd.to_datetime(res, utc=True)
                    logger.info(f"Found existing data up to {existing_max}")
        except Exception as e:
            logger.warning(f"Could not read existing data: {e}")
    
    # Fetch raw data
    if use_history:
        years = fetch_cfg.get('years')
        start_iso = fetch_cfg.get('start')
        end_iso = fetch_cfg.get('end')
        start_dt = pd.to_datetime(start_iso, utc=True) if start_iso else None
        end_dt = pd.to_datetime(end_iso, utc=True) if end_iso else None
        limit = int(fetch_cfg.get('limit', 1000))
        tail_latest_iters = int(fetch_cfg.get('tail_latest_iters', 4))
        overlap_factor = int(fetch_cfg.get('overlap_factor', 50))
        gap_tol = float(fetch_cfg.get('gap_tolerance_factor', 1.5))
        incremental = effective_incremental
        
        # Use cfg.months as fallback if years is None
        months_fallback = getattr(cfg, 'months', 12)
        raw = fetch_ohlcv_history(
            ex, sym, timeframe,
            start_dt=start_dt,
            end_dt=end_dt,
            years=years,
            months=months_fallback,
            limit=limit,
            tail_latest_iters=tail_latest_iters,
            overlap_factor=overlap_factor,
            gap_tolerance_factor=gap_tol,
            incremental=incremental,
            existing_max=existing_max,
            logger=logger
        )
    else:
        raw = fetch_ohlcv(ex, sym, timeframe, months=cfg.months)
    
    if raw.empty:
        logger.warning(f"❌ Empty data for {pair} {timeframe}")
        return pd.DataFrame()
    
    logger.info(f"📊 Raw data: {len(raw):,} candles from {raw.timestamp.min()} to {raw.timestamp.max()}")
    
    # Continuity enhancement
    fetch_cfg = cfg.fetch if hasattr(cfg, 'fetch') else {}
    gap_passes = int(fetch_cfg.get('gap_backfill_passes', 2)) if fetch_cfg else 2
    for _ in range(gap_passes):
        gaps = detect_gaps(raw, timeframe, gap_tolerance_factor=float(fetch_cfg.get('gap_tolerance_factor', 1.5)) if fetch_cfg else 1.5)
        if not gaps:
            break
        raw = backfill_gaps(ex, sym, timeframe, raw, limit=int(fetch_cfg.get('limit', 1000)) if fetch_cfg else 1000, gap_list=gaps, overlap_factor=5)
    
    # Tail extension
    raw = extend_tail(ex, sym, timeframe, raw, limit=int(fetch_cfg.get('limit', 1000)) if fetch_cfg else 1000, passes=3)
    report = validate_continuity(raw, timeframe)
    if not report.get('continuous', True):
        logger.warning(f"Continuity not perfect {pair} {timeframe}: gaps={report['gap_count']}")
    
    # Add metadata
    raw['pair'] = pair
    raw['timeframe'] = timeframe
    
    # Feature engineering dengan parallel processing
    logger.info(f"⚙️  Adding technical indicators for {len(raw)} rows...")
    
    # Determine optimal worker count dan chunk size untuk parallel processing
    if workers is not None:
        n_workers = max(1, min(workers, mp.cpu_count()))  # Validate custom workers
        logger.info(f"🎛️  Using custom worker count: {n_workers} (requested: {workers}, available CPUs: {mp.cpu_count()})")
    else:
        # Auto-detect: gunakan semua CPU cores untuk dataset besar, tapi limit untuk kecil
        total_cpus = mp.cpu_count()
        if len(raw) > 10000:
            n_workers = total_cpus  # Gunakan semua cores untuk dataset besar
        elif len(raw) > 5000:
            n_workers = max(2, total_cpus // 2)  # Setengah cores untuk dataset sedang
        else:
            n_workers = min(4, total_cpus)  # Conservative untuk dataset kecil
        logger.info(f"🔧 Auto-detected workers: {n_workers}/{total_cpus} CPUs (dataset size: {len(raw)} rows)")
    
    chunk_size = max(500, len(raw) // (n_workers * 3))  # Lebih kecil untuk better load balancing
    
    if len(raw) > chunk_size * 2 and n_workers > 1:
        logger.info(f"🚀 Using parallel processing: {n_workers} workers, chunk size: {chunk_size}")
        
        # Split dataframe into chunks for parallel processing
        chunks = [raw.iloc[i:i+chunk_size].copy() for i in range(0, len(raw), chunk_size)]
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_chunk = {executor.submit(_compute_indicators_chunk, chunk, cfg, i): i 
                             for i, chunk in enumerate(chunks)}
            
            processed_chunks = [None] * len(chunks)
            for future in as_completed(future_to_chunk):
                chunk_result, chunk_id = future.result()
                processed_chunks[chunk_id] = chunk_result
        
        # Combine results
        enriched = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"✅ Parallel technical indicators completed. Columns: {len(enriched.columns)}")
    else:
        # Fall back to single-threaded processing for small datasets
        enriched = add_price_indicators(raw.copy(), cfg)
        logger.info(f"✅ Technical indicators added (single-threaded). Columns: {len(enriched.columns)}")
    
    logger.info(f"📈 After indicators: {len(enriched):,} rows")
    
    logger.info(f"🔢 Adding derivative features for {len(enriched)} rows...")
    
    if len(enriched) > chunk_size * 2 and n_workers > 1:
        logger.info(f"🚀 Using parallel processing for derivatives: {n_workers} workers")
        
        # Split dataframe into chunks for parallel processing
        chunks = [enriched.iloc[i:i+chunk_size].copy() for i in range(0, len(enriched), chunk_size)]
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_chunk = {executor.submit(_compute_derivatives_chunk, chunk, cfg, i): i 
                             for i, chunk in enumerate(chunks)}
            
            processed_chunks = [None] * len(chunks)
            for future in as_completed(future_to_chunk):
                chunk_result, chunk_id = future.result()
                processed_chunks[chunk_id] = chunk_result
        
        # Combine results
        enriched = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"✅ Parallel derivatives completed. Final columns: {len(enriched.columns)}")
    else:
        # Fall back to single-threaded processing for small datasets
        enriched = add_derivative_features(enriched, cfg)
        logger.info(f"✅ Derivatives added (single-threaded). Final columns: {len(enriched.columns)}")
    
    logger.info(f"🧮 After derivatives: {len(enriched):,} rows")
    
    logger.info(f"🌐 Merging external data...")
    enriched = merge_external(enriched, cfg, logger)
    logger.info(f"🔗 After external merge: {len(enriched):,} rows")
    
    # Labels (primary direction + multi-horizon + extremes)
    base_horizon = cfg.target.get('horizon', 20)
    
    # ✅ NEW: Support multiple threshold modes
    threshold_mode = cfg.target.get('threshold_mode', 'manual')
    logger.info(f"🎯 Creating base direction label (horizon={base_horizon}, mode={threshold_mode})...")
    
    # Determine effective threshold based on mode
    effective_sideways_thr = 1.0  # Default fallback
    
    if threshold_mode == 'auto_balanced':
        try:
            # Load auto-balanced thresholds
            auto_config_path = cfg.target.get('auto_balanced_config', 'data/auto_balanced_thresholds.json')
            if os.path.exists(auto_config_path):
                import json
                with open(auto_config_path, 'r') as f:
                    auto_config = json.load(f)
                
                # Look for specific pair+timeframe combination
                source_key = f"{pair.replace('USDT', '').upper()} {timeframe}"
                if source_key in auto_config.get('individual_results', {}):
                    effective_sideways_thr = auto_config['individual_results'][source_key]['optimal_threshold']
                    logger.info(f"🎯 Auto-balanced threshold for {timeframe}: {effective_sideways_thr:.3f}% (from {source_key})")
                else:
                    # Use average recommendation
                    effective_sideways_thr = auto_config.get('recommendation', {}).get('threshold', 0.545)
                    logger.info(f"🎯 Auto-balanced threshold for {timeframe}: {effective_sideways_thr:.3f}% (average fallback)")
            else:
                # Use fallback from config
                effective_sideways_thr = cfg.target.get('auto_balance', {}).get('fallback_threshold', 0.545)
                logger.warning(f"Auto-balance config not found, using fallback: {effective_sideways_thr:.3f}%")
                
        except Exception as e:
            logger.error(f"Error loading auto-balanced threshold: {e}")
            effective_sideways_thr = cfg.target.get('auto_balance', {}).get('fallback_threshold', 0.545)
    
    elif threshold_mode == 'adaptive':
        # Original adaptive threshold logic
        sideways_thr = cfg.target.get('sideways_threshold_pct', 1.0)
        effective_sideways_thr = sideways_thr
        if hasattr(cfg, 'target') and 'adaptive_thresholds' in cfg.target:
            adaptive_cfg = cfg.target['adaptive_thresholds']
            if adaptive_cfg.get('enabled', False):
                try:
                    # Import auto-calculation utility 
                    from utils.adaptive_thresholds import get_adaptive_multipliers
                    
                    # Check if current timeframe database exists before trying adaptive calculation
                    db_path = generate_db_path(pair, timeframe, base_dir=dataset_base_dir)
                    if not os.path.exists(db_path):
                        logger.debug(f"Database {db_path} belum ada, menggunakan base threshold untuk {timeframe}")
                        effective_sideways_thr = sideways_thr
                    else:
                        # Get all configured timeframes for multiplier calculation
                        all_timeframes = [timeframe]
                        if 'manual_multipliers' in adaptive_cfg:
                            all_timeframes.extend(adaptive_cfg['manual_multipliers'].keys())
                        all_timeframes = list(set(all_timeframes))  # Remove duplicates
                        
                        # Get optimal multipliers (auto-calculated or manual)
                        multipliers = get_adaptive_multipliers(cfg, pair, all_timeframes)
                        
                        if isinstance(multipliers, dict) and timeframe in multipliers:
                            multiplier = multipliers[timeframe]
                            effective_sideways_thr = sideways_thr * multiplier
                            logger.info(f"🎯 Adaptive threshold for {timeframe}: {effective_sideways_thr:.2f}% "
                                       f"(base: {sideways_thr}%, multiplier: {multiplier}x)")
                        else:
                            logger.debug(f"No adaptive multiplier found for {timeframe}, using base threshold")
                        
                except ImportError as e:
                    logger.warning(f"Cannot import adaptive_thresholds utility: {e}")
                except Exception as e:
                    logger.debug(f"Error calculating adaptive threshold: {e}, using base threshold")
    
    else:  # manual mode
        effective_sideways_thr = cfg.target.get('sideways_threshold_pct', 1.0)
        logger.info(f"🎯 Manual threshold for {timeframe}: {effective_sideways_thr:.3f}%")
    
    # Multi-horizon direction labels (configurable list; fallback to [1,5,base_horizon])
    mh_cfg = getattr(cfg, 'multi_horizon', {}) if hasattr(cfg, 'multi_horizon') else {}
    horizons = mh_cfg.get('horizons', [1, 5, base_horizon])
    logger.info(f"🎯 Adding multi-horizon direction labels: {horizons}")
    enriched = label_multi_horizon_directions(enriched, horizons, effective_sideways_thr, timeframe=timeframe, cfg=cfg, pair=pair)
    
    # Only create legacy 'direction' if base_horizon is NOT in multi_horizon list
    if base_horizon not in horizons:
        logger.info(f"🎯 Creating legacy direction label (horizon={base_horizon}) - not covered by multi-horizon")
        enriched = label_future_direction(enriched, base_horizon, effective_sideways_thr)
    else:
        logger.info(f"🎯 Skipping legacy direction label - already covered by direction_h{base_horizon} in multi-horizon")

    # Extremes (only for largest horizon to start)
    if mh_cfg.get('enable_extremes', True):
        target_extreme_h = mh_cfg.get('extremes_horizon', base_horizon)
        logger.info(f"📈 Adding extremes for horizon={target_extreme_h} (max high / min low)...")
        enriched = label_extremes_horizon(enriched, target_extreme_h, include_time_to=mh_cfg.get('include_time_to', False))

    # Volatility regime (unchanged)
    vol_col = f'volatility_{cfg.volatility_period}'
    enriched = label_regime_volatility(enriched, vol_col)
    logger.info(f"🏷️  After labeling: {len(enriched):,} rows (columns now: {len(enriched.columns)})")

    # === IMPROVED NaN handling policy ===
    # Remove rows with NaN in key technical indicators that models depend on
    before_clean = len(enriched)
    
    # Define core indicators that MUST be valid (more aggressive than before)
    core_indicators = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
    
    # Add key technical indicators that models typically use
    key_technical = []
    for col in enriched.columns:
        # RSI (critical momentum indicator)
        if col.startswith('rsi_'):
            key_technical.append(col)
        # MACD (critical trend indicator)  
        elif col in ['macd', 'macd_signal', 'macd_histogram']:
            key_technical.append(col)
        # Basic moving averages (trend following)
        elif col.startswith(('sma_', 'ema_')) and any(p in col for p in ['_4', '_20']):
            key_technical.append(col)
        # Basic Bollinger Bands (volatility)
        elif col.startswith('bb_24_') and any(x in col for x in ['middle', 'upper_2.0', 'lower_2.0']):
            key_technical.append(col)
        # BB72 indicators (require longer warm-up) - CRITICAL for quality data
        elif col.startswith('bb_72_') and any(x in col for x in ['middle', 'upper_1.0', 'lower_1.0']):
            key_technical.append(col)
    
    # Combine core + key technical indicators (include ALL BB72 indicators)
    essential_cols = list(core_indicators) + key_technical  # Use ALL technical indicators
    existing_essential = [c for c in essential_cols if c in enriched.columns]
    
    if existing_essential:
        # Find first row where ALL essential indicators are valid (very aggressive)
        mask_all_valid = enriched[existing_essential].notna().all(axis=1)
        
        if mask_all_valid.any():
            first_valid_pos = mask_all_valid.idxmax()
            enriched = enriched.loc[first_valid_pos:].copy()
            trimmed_head = before_clean - len(enriched)
            logger.info(f"🔧 Trimmed {trimmed_head} head rows with NaN in essential indicators (including BB72)")
        else:
            # If no row has ALL indicators valid, be more flexible - just require core + BB24
            logger.warning(f"⚠️  No rows with all indicators valid! Falling back to core+BB24 trimming")
            fallback_cols = [c for c in existing_essential if not c.startswith('bb_72_')]
            if fallback_cols:
                core_mask = enriched[fallback_cols].notna().all(axis=1)
                if core_mask.any():
                    first_valid_pos = core_mask.idxmax()
                    enriched = enriched.loc[first_valid_pos:].copy()
                    trimmed_head = before_clean - len(enriched)
                    logger.info(f"🔧 Trimmed {trimmed_head} head rows (core+BB24 only)")
                else:
                    logger.error(f"❌ No valid rows found even for core indicators!")
            else:
                logger.error(f"❌ No valid fallback columns found!")
    
    # Trim tail: remove rows without target labels (if they exist)
    base_target_cols = ['future_return_pct', 'direction']
    multi_h_targets = [
        c for c in enriched.columns
        if c.startswith(('direction_h', 'future_return_pct_h'))
    ]
    extreme_targets = [
        c for c in enriched.columns
        if c.startswith(('future_max_high_h', 'future_min_low_h', 'time_to_max_high_h', 'time_to_min_low_h'))
    ]

    target_cols = base_target_cols + multi_h_targets + extreme_targets
    existing_targets = [c for c in target_cols if c in enriched.columns]
    
    if existing_targets:
        before_tail = len(enriched)
        enriched = enriched.dropna(subset=existing_targets).copy()
        trimmed_tail = before_tail - len(enriched)
        if trimmed_tail > 0:
            logger.info(f"🎯 Trimmed {trimmed_tail} tail rows without target labels (including multi-horizon/extremes)")
            report_after_trim = validate_continuity(enriched, timeframe)
            if not report_after_trim.get('continuous', True):
                logger.warning(
                    f"⚠️  Post-trim continuity check detected gaps for {pair} {timeframe}: {report_after_trim}"
                )
        else:
            logger.info("🎯 No tail trimming needed; all horizon labels populated")
    
    after_clean = len(enriched)
    dropped_rows = before_clean - after_clean
    
    logger.info(f"🧹 Conservative trimming: {before_clean:,} → {after_clean:,} rows (-{dropped_rows:,})")
    
    # Processing time
    processing_time = time.time() - start_time
    logger.info(f"⏱️  Processing completed in {processing_time:.1f}s")
    
    # Target distribution
    if 'direction' in enriched.columns:
        target_dist = enriched['direction'].value_counts().sort_index()
        total = len(enriched)
        logger.info(f"📊 Target distribution:")
        for direction, count in target_dist.items():
            direction_name = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}.get(direction, f'CLASS_{direction}')
            pct = (count / total) * 100
            logger.info(f"   {direction_name}: {count:,} ({pct:.1f}%)")
    
    logger.info(f"✅ {pair} {timeframe} processing completed successfully!")
    return enriched


def build_separate_timeframes(
    cfg_path: str = 'config/config.yaml',
    base_output_dir: Optional[str] = None,
    target_timeframe: str = None,
    target_pairs: List[str] = None,
    workers: Optional[int] = None,
):
    """Build datasets with separate databases per timeframe (IMPROVED APPROACH)"""
    logger = get_logger('pipeline')
    cfg = load_config(cfg_path)
    # Apply runtime CLI overrides without modifying YAML
    try:
        import argparse as _argparse
        _parser = _argparse.ArgumentParser(add_help=False)
        _parser.add_argument('--months', type=int, default=None)
        _parser.add_argument('--force-full-rebuild', dest='force_full_rebuild', action='store_true')
        _parser.add_argument('--no-force-full-rebuild', dest='force_full_rebuild', action='store_false')
        _parser.set_defaults(force_full_rebuild=None)
        _parser.add_argument('--retain-full-history', dest='retain_full_history', action='store_true')
        _parser.add_argument('--no-retain-full-history', dest='retain_full_history', action='store_false')
        _parser.set_defaults(retain_full_history=None)
        _known, _ = _parser.parse_known_args()
        if _known.months is not None:
            cfg.months = _known.months
        if _known.force_full_rebuild is not None:
            cfg.force_full_rebuild = _known.force_full_rebuild
        if _known.retain_full_history is not None:
            cfg.retain_full_history = _known.retain_full_history
    except Exception:
        pass
    # Resolve storage destinations and table names from configuration
    dataset_base_dir = resolve_dataset_base_dir(cfg, override=base_output_dir)
    feature_table = resolve_feature_table(cfg)
    metadata_table = resolve_metadata_table(cfg)

    # Flags available across the function (after overrides)
    force_full_flag = getattr(cfg, 'force_full_rebuild', True)
    retain_history_flag = getattr(cfg, 'retain_full_history', False)
    
    # Get market type from config
    market_type = cfg._extra.get('data', {}).get('market_type', 'future')
    ex = init_exchange(market_type)
    
    # Filter timeframes if specific timeframe requested
    timeframes_to_process = cfg.timeframes
    if target_timeframe:
        if target_timeframe in cfg.timeframes:
            timeframes_to_process = [target_timeframe]
            logger.info(f"Processing only timeframe: {target_timeframe}")
        else:
            logger.error(f"Requested timeframe '{target_timeframe}' not found in config. Available: {cfg.timeframes}")
            return None
    
    # Filter pairs if specific pairs requested
    pairs_to_process = cfg.pairs
    if target_pairs:
        # Convert to uppercase with USDT suffix for consistency
        normalized_pairs = []
        for pair in target_pairs:
            pair_upper = pair.upper()
            if not pair_upper.endswith('USDT'):
                pair_upper += 'USDT'
            normalized_pairs.append(pair_upper)
        pairs_to_process = normalized_pairs
        logger.info(f"Pairs override: {target_pairs} → {pairs_to_process}")
    
    logger.info(f"Starting separate timeframe collection to {dataset_base_dir}/ (market: {market_type})")
    logger.info(f"Timeframes to process: {timeframes_to_process}")
    logger.info(f"Pairs to process: {pairs_to_process}")
    
    processed_count = 0
    
    for pair in pairs_to_process:
        for tf in timeframes_to_process:
            try:
                # Process this pair-timeframe combination
                df = fetch_and_process_timeframe(
                    ex,
                    cfg,
                    pair,
                    tf,
                    logger,
                    dataset_base_dir,
                    feature_table,
                    workers,
                )
                
                if df.empty:
                    logger.warning(f"Skipping empty dataset for {pair} {tf}")
                    continue
                
                # Generate timeframe-specific database path
                db_path = generate_db_path(pair, tf, base_dir=dataset_base_dir)
                
                # Determine save mode - force_full_rebuild always uses replace
                fetch_cfg = cfg.fetch if hasattr(cfg, 'fetch') else {}
                is_incremental = bool(fetch_cfg.get('incremental', False)) and not force_full_flag
                save_mode = 'replace' if force_full_flag else ('append' if is_incremental and os.path.exists(db_path) else 'replace')
                
                logger.info(f"Saving to {db_path} in {save_mode} mode")

                # ==== Data hygiene before persist ====
                # 1. Ensure timestamp is datetime (UTC) and sorted
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df = df.sort_values('timestamp')

                # 2. Keep only fully closed candles (drop any future / partial)
                from data.binance_fetch import _last_closed_candle
                last_closed = _last_closed_candle(timeframe=tf)
                df = df[df['timestamp'] <= last_closed].copy()

                # 3. Deduplicate on timestamp (keep last occurrence after feature calc)
                before_dups = len(df)
                df = df.drop_duplicates(subset=['timestamp'], keep='last')
                removed = before_dups - len(df)
                if removed:
                    logger.info(f"🧽 Removed {removed} duplicate rows for {pair} {tf}")

                # 4. Merge with existing DB (if exists) to preserve previously collected rows
                #    CRITICAL: Skip merge if force_full_rebuild is active!
                merged_df = df
                retain_history = retain_history_flag
                
                # Only merge if NOT force rebuilding
                if not force_full_flag and os.path.exists(db_path):
                    try:
                        with sqlite3.connect(db_path) as conn:
                            old = pd.read_sql_query(f"SELECT * FROM {feature_table}", conn, parse_dates=['timestamp'])
                        if not old.empty:
                            merged_df = pd.concat([old, df], ignore_index=True)
                            before_merge = len(merged_df)
                            merged_df = merged_df.drop_duplicates(subset=['timestamp'], keep='last')
                            merged_df = merged_df.sort_values('timestamp')
                            logger.info(f"📚 Merged with existing: old={len(old)}, new={len(df)}, unique={len(merged_df)}")
                    except Exception as e:
                        logger.warning(f"Failed to merge existing history for {pair} {tf}: {e}")
                elif force_full_flag:
                    logger.info(f"🔥 Force full rebuild - skipping merge with existing data for {pair} {tf}")

                # 5. Optional rolling retention window based on config.months to prevent unbounded growth
                final_df = merged_df
                if hasattr(cfg, 'months') and cfg.months and not retain_history:
                    lookback_days = int(cfg.months) * 30
                    cutoff = last_closed - pd.Timedelta(days=lookback_days)
                    before_trim = len(final_df)
                    final_df = final_df[final_df['timestamp'] >= cutoff]
                    trimmed = before_trim - len(final_df)
                    if trimmed:
                        logger.info(f"✂️  Trimmed {trimmed} old rows beyond {lookback_days}d window for {pair} {tf}")

                # 6. Re-check ordering
                final_df = final_df.sort_values('timestamp')

                # 7. Conservative NaN handling: only remove rows with NaN in critical columns
                # Keep rows with NaN only in optional external data or complex indicator columns
                critical_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                # Add direction labels if they exist
                if 'direction' in final_df.columns:
                    critical_cols.append('direction')
                
                # Add only basic moving averages as critical (not all indicators)
                basic_mas = [c for c in final_df.columns if c.startswith(('sma_20', 'ema_20', 'sma_50'))]
                critical_cols.extend(basic_mas[:2])  # Only first 2 basic MAs
                
                # Only check critical columns that actually exist
                existing_critical = [c for c in critical_cols if c in final_df.columns]
                before_nan = len(final_df)
                
                if existing_critical:
                    final_df = final_df.dropna(subset=existing_critical)
                    removed_nan = before_nan - len(final_df)
                    if removed_nan:
                        logger.info(f"🧽 Conservative NaN purge removed {removed_nan} rows with missing critical data for {pair} {tf}")
                else:
                    logger.warning(f"⚠️  No critical columns found for NaN checking in {pair} {tf}")

                # Save to timeframe-specific database
                save_to_sqlite(final_df, db_path, feature_table, mode=save_mode)
                logger.info(f"💾 Saved using {save_mode} mode for {pair} {tf}")
                
                # Save metadata
                meta = pd.DataFrame([{
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'pair': pair,
                    'timeframe': tf,
                    'rows': len(final_df),
                    'start_time': final_df['timestamp'].min().isoformat(),
                    'end_time': final_df['timestamp'].max().isoformat(),
                    'duration_days': (final_df['timestamp'].max() - final_df['timestamp'].min()).days,
                    'config_path': cfg_path
                }])
                save_to_sqlite(meta, db_path, metadata_table, mode='append')
                
                logger.info(f"✅ Saved {pair} {tf}: {len(final_df):,} rows to {db_path}")
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {pair} {tf}: {e}")
                continue
    
    if processed_count == 0:
        logger.error("No datasets were successfully processed!")
        return None
    
    logger.info(f"🎉 Successfully processed {processed_count} timeframe datasets")
    
    # Summary report
    logger.info("=== SUMMARY REPORT ===")
    for pair in pairs_to_process:
        for tf in timeframes_to_process:
            db_path = generate_db_path(pair, tf, base_dir=dataset_base_dir)
            if os.path.exists(db_path):
                size_mb = os.path.getsize(db_path) / 1024 / 1024
                with sqlite3.connect(db_path) as conn:
                    try:
                        count = conn.execute(f"SELECT COUNT(*) FROM {feature_table}").fetchone()[0]
                        logger.info(f"  {pair} {tf}: {count:,} rows, {size_mb:.1f} MB")
                    except:
                        logger.info(f"  {pair} {tf}: Database created but could not read")
    
    return processed_count


# Legacy function for backward compatibility
def build(cfg_path: str = 'config/config.yaml', output_db: str = 'data/db/preprocessed.sqlite'):
    """Legacy mixed-timeframe approach (for backward compatibility)"""
    logger = get_logger('pipeline')
    logger.warning("Using legacy mixed-timeframe approach. Consider using build_separate_timeframes() instead.")
    
    cfg = load_config(cfg_path)
    feature_table = resolve_feature_table(cfg)
    metadata_table = resolve_metadata_table(cfg)
    output_db_path = cfg.output.get('sqlite_db', output_db) if hasattr(cfg, 'output') else output_db

    # Get market type from config
    market_type = cfg._extra.get('data', {}).get('market_type', 'future')
    ex = init_exchange(market_type)
    all_frames = []
    
    for pair in cfg.pairs:
        sym = normalize_symbol(ex, pair)
        for tf in cfg.timeframes:
            logger.info(f"Fetching {pair} {tf}")
            fetch_cfg = cfg.fetch if hasattr(cfg, 'fetch') else {}
            use_history = False
            if fetch_cfg:
                if fetch_cfg.get('years') or fetch_cfg.get('start') or fetch_cfg.get('end') or fetch_cfg.get('incremental', False):
                    use_history = True
            
            existing_max = None
            if use_history and fetch_cfg.get('incremental', False):
                if os.path.exists(output_db_path):
                    try:
                        with sqlite3.connect(output_db_path) as conn:
                            q = f"SELECT max(timestamp) FROM {feature_table} WHERE pair='{pair}' AND timeframe='{tf}'"
                            res = conn.execute(q).fetchone()[0]
                            if res:
                                existing_max = pd.to_datetime(res, utc=True)
                    except Exception:
                        pass
            
            if use_history:
                years = fetch_cfg.get('years')
                start_iso = fetch_cfg.get('start')
                end_iso = fetch_cfg.get('end')
                start_dt = pd.to_datetime(start_iso, utc=True) if start_iso else None
                end_dt = pd.to_datetime(end_iso, utc=True) if end_iso else None
                limit = int(fetch_cfg.get('limit', 1000))
                tail_latest_iters = int(fetch_cfg.get('tail_latest_iters', 4))
                overlap_factor = int(fetch_cfg.get('overlap_factor', 50))
                gap_tol = float(fetch_cfg.get('gap_tolerance_factor', 1.5))
                incremental = bool(fetch_cfg.get('incremental', False))
                
                raw = fetch_ohlcv_history(
                    ex, sym, tf,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    years=years,
                    limit=limit,
                    tail_latest_iters=tail_latest_iters,
                    overlap_factor=overlap_factor,
                    gap_tolerance_factor=gap_tol,
                    incremental=incremental,
                    existing_max=existing_max
                )
            else:
                raw = fetch_ohlcv(ex, sym, tf, months=cfg.months)
            
            if raw.empty:
                logger.warning(f"Empty data for {pair} {tf}")
                continue
            
            # Continuity enhancement loop
            fetch_cfg = cfg.fetch if hasattr(cfg, 'fetch') else {}
            gap_passes = int(fetch_cfg.get('gap_backfill_passes', 2)) if fetch_cfg else 2
            for _ in range(gap_passes):
                gaps = detect_gaps(raw, tf, gap_tolerance_factor=float(fetch_cfg.get('gap_tolerance_factor', 1.5)) if fetch_cfg else 1.5)
                if not gaps:
                    break
                raw = backfill_gaps(ex, sym, tf, raw, limit=int(fetch_cfg.get('limit', 1000)) if fetch_cfg else 1000, gap_list=gaps, overlap_factor=5)
            
            raw = extend_tail(ex, sym, tf, raw, limit=int(fetch_cfg.get('limit', 1000)) if fetch_cfg else 1000, passes=3)
            report = validate_continuity(raw, tf)
            if not report.get('continuous', True):
                logger.warning(f"Continuity not perfect {pair} {tf}: gaps={report['gap_count']}")
            
            raw['pair']=pair; raw['timeframe']=tf
            enriched = add_price_indicators(raw.copy(), cfg)
            enriched = add_derivative_features(enriched, cfg)
            enriched = merge_external(enriched, cfg, logger)
            
            # Legacy function: create basic direction label (no multi-horizon support)
            logger.warning("⚠️  Using legacy mixed-timeframe approach - only basic direction label created")
            enriched = label_future_direction(enriched, cfg.target.get('horizon', 20), cfg.target.get('sideways_threshold_pct', 1.0))
            vol_col = f'volatility_{cfg.volatility_period}'
            enriched = label_regime_volatility(enriched, vol_col)
            # enriched.dropna(inplace=True)
            all_frames.append(enriched)
    
    if not all_frames:
        logger.error("No data fetched.")
        return None
    
    final = pd.concat(all_frames, ignore_index=True)
    save_to_sqlite(final, output_db_path, feature_table)
    meta = pd.DataFrame([{ 'created_at': datetime.now(timezone.utc).isoformat(), 'pairs': str(cfg.pairs), 'timeframes': str(cfg.timeframes)}])
    save_to_sqlite(meta, output_db_path, metadata_table, mode='append')
    logger.info(f"Saved dataset rows={len(final)} to {output_db_path}")
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build cryptocurrency trading datasets with technical indicators',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/build_dataset.py                    # Use default config.yaml
  python pipeline/build_dataset.py --config custom.yaml   # Use custom config
  python pipeline/build_dataset.py --output-dir data/custom  # Custom output directory
  python pipeline/build_dataset.py --timeframe 15m   # Process only 15m timeframe
  python pipeline/build_dataset.py --timeframe 1h --months 6  # Process only 1h with 6 months data
  python pipeline/build_dataset.py --pairs btcusdt ethusdt dogeusdt  # Process specific pairs
  python pipeline/build_dataset.py --pairs btcusdt --timeframe 1h    # Single pair, single timeframe
  python pipeline/build_dataset.py --workers 8       # Use 8 parallel workers (max CPU utilization)
  python pipeline/build_dataset.py --workers 1       # Force single-threaded processing
  python pipeline/build_dataset.py --pairs btcusdt --timeframe 15m --workers 4  # Custom workers
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration YAML file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str, 
        default=None,
        help='Output directory override for timeframe databases (defaults to config output settings)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default=None,
        help='Process only specific timeframe (e.g., 15m, 1h, 4h). If not specified, processes all timeframes from config.'
    )
    
    parser.add_argument(
        '--pairs',
        type=str,
        nargs='+',
        default=None,
        help='Process specific trading pairs (e.g., btcusdt ethusdt dogeusdt). If not specified, uses pairs from config.'
    )
    
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Use legacy mixed-timeframe approach instead of separate databases'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers for processing (default: auto-detect, max: all CPU cores)'
    )
    # Runtime overrides (do not modify YAML file)
    parser.add_argument(
        '--months', type=int, default=None,
        help='Override months lookback window (int). Does not modify YAML.'
    )
    parser.add_argument(
        '--force-full-rebuild', dest='force_full_rebuild', action='store_true',
        help='Force full rebuild (override). Does not modify YAML.'
    )
    parser.add_argument(
        '--no-force-full-rebuild', dest='force_full_rebuild', action='store_false',
        help='Disable force full rebuild (override). Does not modify YAML.'
    )
    parser.set_defaults(force_full_rebuild=None)
    parser.add_argument(
        '--retain-full-history', dest='retain_full_history', action='store_true',
        help='Retain old history beyond months window (override). Does not modify YAML.'
    )
    parser.add_argument(
        '--no-retain-full-history', dest='retain_full_history', action='store_false',
        help='Do not retain old history (override). Does not modify YAML.'
    )
    parser.set_defaults(retain_full_history=None)
    
    args = parser.parse_args()
    
    if args.legacy:
        # Use legacy approach
        result = build(args.config)
    else:
        # Use new separate approach by default, applying runtime overrides
        # We pass overrides through environment by re-parsing inside builder; just call normally here
        result = build_separate_timeframes(args.config, args.output_dir, args.timeframe, target_pairs=args.pairs, workers=args.workers)
    
    if result is None or result == 0:
        print("Failed to build datasets")
        exit(1)
    else:
        print(f"Successfully built {result} timeframe datasets")