from datetime import datetime, timedelta, timezone
import time
import pandas as pd
import ccxt
from tqdm import tqdm
import logging

TIMEFRAME_MIN = {'1m':1,'5m':5,'15m':15,'30m':30,'1h':60,'2h':120,'4h':240,'6h':360,'1d':1440}

def init_exchange(market_type: str = 'future'):
    """Initialize Binance exchange with market type selection"""
    if market_type not in ['spot', 'future']:
        raise ValueError(f"Unsupported market_type: {market_type}. Use 'spot' or 'future'")
    
    ex = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': market_type}
    })
    ex.load_markets()
    return ex

def normalize_symbol(ex, symbol: str) -> str:
    if symbol in ex.markets:
        return symbol
    s = symbol.replace('_','/') if '_' in symbol else symbol
    if s in ex.markets:
        return s
    if not s.endswith('/USDT') and s.endswith('USDT'):
        candidate = s.replace('USDT','/USDT')
        if candidate in ex.markets:
            return candidate
    raise ValueError(f"Symbol not supported: {symbol}")

def fetch_ohlcv(ex, symbol: str, timeframe: str, months: int = 6, limit: int = 1000) -> pd.DataFrame:
    if timeframe not in TIMEFRAME_MIN:
        raise ValueError(f"Unsupported timeframe {timeframe}")
    now_utc = datetime.now(timezone.utc)
    # Define the target last closed candle boundary (exclude the still-forming current interval)
    tf_minutes = TIMEFRAME_MIN[timeframe]
    target_last_closed = now_utc - timedelta(minutes=now_utc.minute % tf_minutes,
                                             seconds=now_utc.second,
                                             microseconds=now_utc.microsecond)
    target_last_closed -= timedelta(minutes=tf_minutes)  # ensure we only go up to the previous fully closed candle

    start = target_last_closed - timedelta(days=months*30)
    since = int(start.timestamp()*1000)
    all_rows = []
    # Primary historical pagination loop (backfill)
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows += batch
        last_batch_ts = batch[-1][0]
        # Advance cursor by one timeframe to avoid duplicating last candle next fetch
        since = last_batch_ts + tf_minutes*60*1000
        # Stop backfill if we've reached our target window end (target_last_closed)
        if last_batch_ts >= int(target_last_closed.timestamp()*1000):
            break
        time.sleep(ex.rateLimit/1000)

    # Iterative forward fill: ensure we have every candle up to target_last_closed
    # Some exchanges may deliver slightly delayed recent candles; we refetch the trailing window until caught up
    safety_iters = 5  # prevent infinite loops
    while safety_iters > 0 and all_rows:
        safety_iters -= 1
        df_tmp = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
        df_tmp.sort_values('ts', inplace=True)
        last_have = df_tmp.ts.iloc[-1]
        if last_have >= int(target_last_closed.timestamp()*1000):
            break  # already have target
        # Fetch from last_have + one tf to target
        forward_since = last_have + tf_minutes*60*1000
        try:
            forward_batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=forward_since, limit=limit)
        except Exception:
            forward_batch = []
        if not forward_batch:
            # If no data returned, wait briefly then retry next iteration
            time.sleep(min(2, ex.rateLimit/1000 + 0.2))
            continue
        all_rows += forward_batch
        time.sleep(ex.rateLimit/1000)

    # Final tail consolidation: one more compact refetch of last ~2*limit candles to capture any late adjustments
    if all_rows:
        last_start = max(all_rows[-1][0] - (tf_minutes*60*1000*limit*2), 0)
        try:
            tail = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=last_start, limit=limit)
            if tail:
                all_rows += tail
        except Exception:
            pass
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
    df['timestamp'] = pd.to_datetime(df.ts, unit='ms', utc=True)
    df.drop_duplicates('timestamp', inplace=True)
    df.sort_values('timestamp', inplace=True)
    # Trim any candles beyond target_last_closed (if exchange returned partial forming candle)
    df = df[df['timestamp'] <= target_last_closed]
    # Optional continuity check (no gap > timeframe) – could log if needed
    # diffs = df['timestamp'].diff().dropna().dt.total_seconds()/60
    # if (diffs > TIMEFRAME_MIN[timeframe]*1.5).any():
    #     print(f"[warn] Gap detected in {symbol} {timeframe}")
    return df[['timestamp','open','high','low','close','volume']]


def _last_closed_candle(timeframe: str) -> datetime:
    """Return the END boundary of the last fully closed candle (exclude current forming)."""
    now_utc = datetime.now(timezone.utc)
    tf_minutes = TIMEFRAME_MIN[timeframe]
    aligned = now_utc - timedelta(minutes=now_utc.minute % tf_minutes,
                                  seconds=now_utc.second,
                                  microseconds=now_utc.microsecond)
    # Subtract one full timeframe so we only reference a fully closed candle boundary
    return aligned - timedelta(minutes=tf_minutes)


def fetch_ohlcv_history(
    ex,
    symbol: str,
    timeframe: str,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    years: int | None = None,
    months: int | None = None,
    limit: int = 1000,
    tail_latest_iters: int = 4,
    overlap_factor: int = 50,
    gap_tolerance_factor: float = 1.5,
    incremental: bool = False,
    existing_max: datetime | None = None,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """Robust multi-year historical fetch with pagination, tail sync, overlap refinement, and gap checks.

    Parameters mirror config.fetch.* keys. existing_max digunakan untuk mode incremental.
    """
    if timeframe not in TIMEFRAME_MIN:
        raise ValueError(f"Unsupported timeframe {timeframe}")
    tf_min = TIMEFRAME_MIN[timeframe]
    tf_ms = tf_min * 60 * 1000
    last_closed = _last_closed_candle(timeframe)
    # Resolve end_dt
    if end_dt is None:
        end_dt = last_closed
    else:
        # Trim end if melewati last_closed
        if end_dt > last_closed:
            end_dt = last_closed
    # Resolve start_dt
    if start_dt is None:
        if years is not None:
            start_dt = end_dt - timedelta(days=years*365)
        elif months is not None:
            start_dt = end_dt - timedelta(days=months*30)
        else:
            # fallback 1 year
            start_dt = end_dt - timedelta(days=365)
    if incremental and existing_max is not None:
        # Mulai dari candle setelah existing_max
        start_dt = existing_max + timedelta(minutes=tf_min)
        if start_dt > end_dt:
            # Tidak ada yang perlu diambil
            return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
    
    cursor = int(start_dt.timestamp()*1000)
    end_ms = int(end_dt.timestamp()*1000)
    rows: list[list|tuple] = []
    empty_streak = 0
    calls = 0
    
    # Calculate total expected requests for progress bar
    total_duration_ms = end_ms - cursor
    expected_requests = max(1, total_duration_ms // (limit * tf_ms))
    
    if logger:
        logger.info(f"📊 Fetching {symbol} {timeframe}: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        logger.info(f"⏱️  Expected ~{expected_requests} API calls")
    
    with tqdm(total=expected_requests, desc=f"Fetching {symbol} {timeframe}", unit="req", disable=logger is None) as pbar:
        while cursor <= end_ms:
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
            except Exception:
                batch = []
            calls += 1
            
            if not batch:
                empty_streak += 1
                if empty_streak >= 3:
                    if logger:
                        logger.warning(f"⚠️  Stopping after {empty_streak} empty responses")
                    break
                time.sleep(min(2, ex.rateLimit/1000 + 0.2))
                continue
                
            empty_streak = 0
            rows += batch
            last_ts = batch[-1][0]
            
            # Update progress bar
            progress = min(calls, expected_requests)
            pbar.n = progress
            pbar.set_postfix({"Rows": len(rows), "Last": datetime.fromtimestamp(last_ts/1000).strftime('%m-%d %H:%M')})
            pbar.refresh()
            
            if last_ts >= end_ms:
                break
                
            # pagination advance
            cursor = last_ts + tf_ms
            time.sleep(ex.rateLimit/1000)

    if not rows:
        return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])

    if logger:
        logger.info(f"🔄 Post-processing {len(rows)} raw candles...")

    # Tail sync: several latest pulls without since
    if logger:
        logger.info(f"🎯 Tail sync: {tail_latest_iters} iterations")
    for i in range(tail_latest_iters):
        try:
            latest = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception:
            latest = []
        if latest:
            rows += latest
        time.sleep(ex.rateLimit/1000)

    # Overlap refinement
    for _ in range(2):
        if not rows:
            break
        max_ts = max(r[0] for r in rows)
        overlap_since = max_ts - tf_ms * overlap_factor
        if overlap_since < 0:
            overlap_since = 0
        try:
            overlap_batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=overlap_since, limit=limit)
        except Exception:
            overlap_batch = []
        if overlap_batch:
            rows += overlap_batch
        time.sleep(ex.rateLimit/1000)

    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df['timestamp'] = pd.to_datetime(df.ts, unit='ms', utc=True)
    df.drop_duplicates('timestamp', inplace=True)
    df.sort_values('timestamp', inplace=True)
    # Trim beyond end_dt
    df = df[df['timestamp'] <= end_dt]
    # Gap validation
    diffs = df['timestamp'].diff().dropna().dt.total_seconds() / 60
    max_allowed = tf_min * gap_tolerance_factor
    gaps_detected = (diffs > max_allowed).sum()
    
    if gaps_detected > 0:
        if logger:
            logger.warning(f"⚠️  {gaps_detected} gaps detected in {symbol} {timeframe} (max gap: {diffs.max():.1f}m)")
        else:
            print(f"[warn] {gaps_detected} gaps detected in {symbol} {timeframe}: largest diff={diffs.max():.1f}m")
    elif logger:
        logger.info(f"✅ No significant gaps detected in {symbol} {timeframe}")
    
    final_rows = len(df)
    if logger:
        logger.info(f"📈 Final dataset: {final_rows:,} candles from {df.timestamp.min()} to {df.timestamp.max()}")
    
    return df[['timestamp','open','high','low','close','volume']]
