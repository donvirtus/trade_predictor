import os, sqlite3, sys
import pandas as pd
from datetime import datetime, timezone
import time
from typing import List, Optional
import argparse

# Allow direct execution: add project root to sys.path if modules not found
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.logging import get_logger

from utils.config import load_config
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


def generate_timeframe_db_path(pair: str, timeframe: str, base_dir: str = 'data/db') -> str:
    """Generate database path for specific pair and timeframe"""
    # Convert BTCUSDT → btc, 5m → 5m
    pair_clean = pair.replace('USDT', '').lower()
    filename = f"{pair_clean}_{timeframe}.sqlite"
    return os.path.join(base_dir, filename)


def fetch_and_process_timeframe(ex, cfg, pair: str, timeframe: str, logger) -> pd.DataFrame:
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
    if use_history and effective_incremental:
        db_path = generate_timeframe_db_path(pair, timeframe)
        if os.path.exists(db_path):
            try:
                with sqlite3.connect(db_path) as conn:
                    q = f"SELECT max(timestamp) FROM features"
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
    
    # Feature engineering
    logger.info(f"⚙️  Adding technical indicators...")
    enriched = add_price_indicators(raw.copy(), cfg)
    logger.info(f"📈 After indicators: {len(enriched):,} rows")
    
    logger.info(f"🔢 Adding derivative features...")
    enriched = add_derivative_features(enriched, cfg)
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
                        logger.warning(f"No adaptive multiplier found for {timeframe}, using base threshold")
                        
                except ImportError as e:
                    logger.warning(f"Cannot import adaptive_thresholds utility: {e}")
                except Exception as e:
                    logger.error(f"Error calculating adaptive threshold: {e}")
    
    else:  # manual mode
        effective_sideways_thr = cfg.target.get('sideways_threshold_pct', 1.0)
        logger.info(f"🎯 Manual threshold for {timeframe}: {effective_sideways_thr:.3f}%")
    
    enriched = label_future_direction(enriched, base_horizon, effective_sideways_thr)

    # Multi-horizon direction labels (configurable list; fallback to [1,5,base_horizon])
    mh_cfg = getattr(cfg, 'multi_horizon', {}) if hasattr(cfg, 'multi_horizon') else {}
    horizons = mh_cfg.get('horizons', [1, 5, base_horizon])
    logger.info(f"🎯 Adding multi-horizon direction labels: {horizons}")
    enriched = label_multi_horizon_directions(enriched, horizons, effective_sideways_thr, timeframe=timeframe, cfg=cfg, pair=pair)

    # Extremes (only for largest horizon to start)
    if mh_cfg.get('enable_extremes', True):
        target_extreme_h = mh_cfg.get('extremes_horizon', base_horizon)
        logger.info(f"📈 Adding extremes for horizon={target_extreme_h} (max high / min low)...")
        enriched = label_extremes_horizon(enriched, target_extreme_h, include_time_to=mh_cfg.get('include_time_to', False))

    # Volatility regime (unchanged)
    vol_col = f'volatility_{cfg.volatility_period}'
    enriched = label_regime_volatility(enriched, vol_col)
    logger.info(f"🏷️  After labeling: {len(enriched):,} rows (columns now: {len(enriched.columns)})")

    # === Smart NaN handling policy ===
    # - Hapus baris NaN di awal hanya untuk kolom critical
    # - Preserve data yang valid walau ada beberapa NaN di features optional
    before_clean = len(enriched)
    
    # Define critical columns that must not have NaN (more conservative)
    critical_cols = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
    
    # Add only a few truly essential indicators (not all)
    basic_indicators = [c for c in enriched.columns if any(c.startswith(x) for x in ['sma_', 'ema_']) and any(p in c for p in ['_20', '_50'])]
    if basic_indicators:
        critical_cols.update(basic_indicators[:2])  # Only add first 2 basic moving averages
    
    # Filter to only existing columns
    existing_critical = [c for c in critical_cols if c in enriched.columns]
    
    # Trim head: remove rows where critical columns have NaN (more conservative)
    if existing_critical:
        # Use a less strict approach - only require OHLCV + one basic indicator
        ohlcv_cols = [c for c in existing_critical if c in {'open', 'high', 'low', 'close', 'volume', 'timestamp'}]
        
        # If we have basic indicators, only require OHLCV + at least one indicator
        if len(existing_critical) > len(ohlcv_cols):
            # More flexible: OHLCV must be valid, but allow some indicator NaNs
            mask_ohlcv_valid = enriched[ohlcv_cols].notna().all(axis=1)
            indicator_cols = [c for c in existing_critical if c not in ohlcv_cols]
            
            if indicator_cols:
                # At least one indicator should be valid (not all)
                mask_some_indicators = enriched[indicator_cols].notna().any(axis=1)
                mask_valid_essential = mask_ohlcv_valid & mask_some_indicators
            else:
                mask_valid_essential = mask_ohlcv_valid
        else:
            # Only OHLCV, require all valid
            mask_valid_essential = enriched[existing_critical].notna().all(axis=1)
        
        if mask_valid_essential.any():
            first_valid_pos = mask_valid_essential.idxmax()
            enriched = enriched.loc[first_valid_pos:].copy()
            logger.info(f"🔧 Trimmed head to first valid critical data row (conservative)")
    
    # Trim tail: only remove rows without target labels (if they exist)
    target_cols = ['future_return_pct', 'direction']
    existing_targets = [c for c in target_cols if c in enriched.columns]
    
    if existing_targets:
        enriched = enriched.dropna(subset=existing_targets).copy()
        logger.info(f"🎯 Trimmed tail rows without target labels")
    
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


def build_separate_timeframes(cfg_path: str = 'config/config.yaml', base_output_dir: str = 'data/db', target_timeframe: str = None, target_pairs: List[str] = None):
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
    
    logger.info(f"Starting separate timeframe collection to {base_output_dir}/ (market: {market_type})")
    logger.info(f"Timeframes to process: {timeframes_to_process}")
    logger.info(f"Pairs to process: {pairs_to_process}")
    
    processed_count = 0
    
    for pair in pairs_to_process:
        for tf in timeframes_to_process:
            try:
                # Process this pair-timeframe combination
                df = fetch_and_process_timeframe(ex, cfg, pair, tf, logger)
                
                if df.empty:
                    logger.warning(f"Skipping empty dataset for {pair} {tf}")
                    continue
                
                # Generate timeframe-specific database path
                db_path = generate_timeframe_db_path(pair, tf, base_output_dir)
                
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
                            old = pd.read_sql_query("SELECT * FROM features", conn, parse_dates=['timestamp'])
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
                save_to_sqlite(final_df, db_path, 'features', mode=save_mode)
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
                save_to_sqlite(meta, db_path, 'metadata', mode='append')
                
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
            db_path = generate_timeframe_db_path(pair, tf, base_output_dir)
            if os.path.exists(db_path):
                size_mb = os.path.getsize(db_path) / 1024 / 1024
                with sqlite3.connect(db_path) as conn:
                    try:
                        count = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
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
                out_db = cfg.output.get('sqlite_db', 'data/db/preprocessed.sqlite')
                if os.path.exists(out_db):
                    try:
                        with sqlite3.connect(out_db) as conn:
                            q = f"SELECT max(timestamp) FROM features WHERE pair='{pair}' AND timeframe='{tf}'"
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
            enriched = label_future_direction(enriched, cfg.target.get('horizon', 20), cfg.target.get('sideways_threshold_pct', 1.0))
            vol_col = f'volatility_{cfg.volatility_period}'
            enriched = label_regime_volatility(enriched, vol_col)
            # enriched.dropna(inplace=True)
            all_frames.append(enriched)
    
    if not all_frames:
        logger.error("No data fetched.")
        return None
    
    final = pd.concat(all_frames, ignore_index=True)
    save_to_sqlite(final, output_db, 'features')
    meta = pd.DataFrame([{ 'created_at': datetime.now(timezone.utc).isoformat(), 'pairs': str(cfg.pairs), 'timeframes': str(cfg.timeframes)}])
    save_to_sqlite(meta, output_db, 'metadata', mode='append')
    logger.info(f"Saved dataset rows={len(final)} to {output_db}")
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
        default='data/db',
        help='Output directory for database files (default: data/db)'
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
        result = build_separate_timeframes(args.config, args.output_dir, args.timeframe, target_pairs=args.pairs)
    
    if result is None or result == 0:
        print("Failed to build datasets")
        exit(1)
    else:
        print(f"Successfully built {result} timeframe datasets")