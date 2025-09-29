import pandas as pd
from datetime import timedelta
from typing import List, Tuple
from ccxt.base.exchange import Exchange

def parse_timeframe_to_ms(timeframe: str) -> int:
    if timeframe.endswith('m'):
        return int(timeframe[:-1]) * 60 * 1000
    if timeframe.endswith('h'):
        return int(timeframe[:-1]) * 3600 * 1000
    raise ValueError(f"Unsupported timeframe: {timeframe}")

def detect_gaps(df: pd.DataFrame, timeframe: str, gap_tolerance_factor: float = 1.5) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if df.empty or len(df) < 2:
        return []
    df = df.sort_values('timestamp').reset_index(drop=True)
    diffs = df['timestamp'].diff(periods=1)[1:]
    interval_ms = parse_timeframe_to_ms(timeframe)
    threshold_ms = interval_ms * gap_tolerance_factor
    gaps = []
    for i in range(1, len(df)):
        diff_ms = (df['timestamp'].iloc[i] - df['timestamp'].iloc[i-1]).total_seconds() * 1000
        if diff_ms > threshold_ms:
            gaps.append((df['timestamp'].iloc[i-1], df['timestamp'].iloc[i]))
    return gaps

def backfill_gaps(ex: Exchange, symbol: str, timeframe: str, df: pd.DataFrame, limit: int, gap_list: List[Tuple[pd.Timestamp, pd.Timestamp]], overlap_factor: int = 5) -> pd.DataFrame:
    interval_ms = parse_timeframe_to_ms(timeframe)
    for start, end in gap_list:
        fetch_since = int(start.timestamp() * 1000 - overlap_factor * interval_ms)
        new_data = []
        current_since = fetch_since
        while True:
            batch_list = ex.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            if not batch_list:
                break
            batch = pd.DataFrame(batch_list, columns=["ts","open","high","low","close","volume"])
            batch['timestamp'] = pd.to_datetime(batch['ts'], unit='ms', utc=True)
            batch = batch[['timestamp','open','high','low','close','volume']]
            new_data.append(batch)
            last_ts = batch['timestamp'].max()
            if last_ts >= end:
                break
            current_since = int(last_ts.timestamp() * 1000) + 1
        if new_data:
            new_df = pd.concat(new_data)
            df = pd.concat([df, new_df]).drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    return df

def extend_tail(ex: Exchange, symbol: str, timeframe: str, df: pd.DataFrame, limit: int, passes: int = 3) -> pd.DataFrame:
    interval_ms = parse_timeframe_to_ms(timeframe)
    for _ in range(passes):
        if df.empty:
            break
        max_ts = df['timestamp'].max()
        since = int(max_ts.timestamp() * 1000) + interval_ms
        batch_list = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch_list:
            break
        new = pd.DataFrame(batch_list, columns=["ts","open","high","low","close","volume"])
        new['timestamp'] = pd.to_datetime(new['ts'], unit='ms', utc=True)
        new = new[['timestamp','open','high','low','close','volume']]
        if new.empty:
            break
        df = pd.concat([df, new]).drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    return df

def validate_continuity(df: pd.DataFrame, timeframe: str) -> dict:
    gaps = detect_gaps(df, timeframe)
    return {
        'continuous': len(gaps) == 0,
        'gap_count': len(gaps),
        'gaps': gaps if gaps else None
    }