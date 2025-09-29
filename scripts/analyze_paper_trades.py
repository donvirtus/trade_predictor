#!/usr/bin/env python3
"""
Analyze paper-trade CSV logs produced by scripts/predict_enhanced.py.

Usage:
  python3 scripts/analyze_paper_trades.py --log data/logs/paper_trades_5m.csv [--from "2025-09-01"] [--to "2025-09-27"]

Metrics:
- total_open: jumlah baris OPEN
- total_close: jumlah baris CLOSE
- open_trades: trade OPEN tanpa CLOSE
- closed_trades: trade dengan CLOSE
- win_rate: persentase CLOSE dengan pnl_pct > 0
- avg_pnl_pct: rata-rata pnl_pct (CLOSE)
- avg_hold_minutes: rata-rata durasi dari OPEN ke CLOSE (menit)
- by_direction: metrik terpisah LONG/SHORT
- outcomes: distribusi outcome (TP/SL/TIMEOUT)

Notes:
- Script toleran pada kolom yang hilang. Akan memberi peringatan bila format tak lengkap.
"""
import argparse
import os
import sys
import pandas as pd
from datetime import datetime
from collections import defaultdict


def load_csv(log_path: str) -> pd.DataFrame:
    if not os.path.exists(log_path):
        print(f"❌ Log not found: {log_path}")
        sys.exit(1)
    df = pd.read_csv(log_path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df


def filter_date_range(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    # Parse ISO timestamps from open_time/close_time when present
    def to_ts(val):
        try:
            return pd.to_datetime(val, utc=True)
        except Exception:
            return pd.NaT
    if 'open_time' in df.columns:
        df['open_time_ts'] = df['open_time'].apply(to_ts)
    if 'close_time' in df.columns:
        df['close_time_ts'] = df['close_time'].apply(to_ts)

    if start:
        start_ts = pd.to_datetime(start, utc=True)
        df = df[(df.get('open_time_ts', pd.NaT) >= start_ts) | (df.get('close_time_ts', pd.NaT) >= start_ts)]
    if end:
        end_ts = pd.to_datetime(end, utc=True)
        df = df[(df.get('open_time_ts', pd.NaT) <= end_ts) | (df.get('close_time_ts', pd.NaT) <= end_ts)]
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    # Split
    opens = df[df['event'] == 'OPEN'].copy() if 'event' in df.columns else pd.DataFrame()
    closes = df[df['event'] == 'CLOSE'].copy() if 'event' in df.columns else pd.DataFrame()

    # Index by trade_id
    opens_map = {str(r.trade_id): r for r in opens.itertuples(index=False)}
    closes_map = {str(r.trade_id): r for r in closes.itertuples(index=False)}

    # Pairing
    closed_ids = set(opens_map.keys()) & set(closes_map.keys())
    open_only_ids = set(opens_map.keys()) - set(closes_map.keys())

    # Durations and pnl
    durations = []
    pnls = []
    by_dir = defaultdict(lambda: {'count':0, 'wins':0, 'pnl_sum':0.0, 'durations': []})
    outcomes = defaultdict(int)

    for tid in closed_ids:
        o = opens_map[tid]
        c = closes_map[tid]
        try:
            ot = pd.to_datetime(o.open_time, utc=True)
            ct = pd.to_datetime(c.close_time, utc=True)
        except Exception:
            ot = pd.NaT; ct = pd.NaT
        dur_min = float((ct - ot).total_seconds()/60.0) if (ot is not pd.NaT and ct is not pd.NaT) else None
        pnl = float(c.pnl_pct) if hasattr(c, 'pnl_pct') and pd.notna(c.pnl_pct) else None
        dirn = str(getattr(o, 'direction', 'UNKNOWN')).upper()
        outc = str(getattr(c, 'outcome', 'UNKNOWN')).upper()

        if pnl is not None:
            pnls.append(pnl)
        if dur_min is not None:
            durations.append(dur_min)
        if dirn:
            by_dir[dirn]['count'] += 1
            if pnl is not None:
                if pnl > 0:
                    by_dir[dirn]['wins'] += 1
                by_dir[dirn]['pnl_sum'] += pnl
            if dur_min is not None:
                by_dir[dirn]['durations'].append(dur_min)
        outcomes[outc] += 1

    total_closed = len(closed_ids)
    total_open_only = len(open_only_ids)
    win_rate = (sum(1 for p in pnls if p is not None and p > 0) / total_closed * 100.0) if total_closed > 0 else 0.0
    avg_pnl = (sum(p for p in pnls if p is not None) / len([p for p in pnls if p is not None])) if pnls else 0.0
    avg_hold = (sum(durations)/len(durations)) if durations else 0.0

    # Direction breakdown
    dir_report = {}
    for d, agg in by_dir.items():
        cnt = agg['count']
        wins = agg['wins']
        wr = (wins/cnt*100.0) if cnt>0 else 0.0
        avgp = (agg['pnl_sum']/cnt) if cnt>0 else 0.0
        avgh = (sum(agg['durations'])/len(agg['durations'])) if agg['durations'] else 0.0
        dir_report[d] = {
            'trades': cnt,
            'win_rate_pct': wr,
            'avg_pnl_pct': avgp,
            'avg_hold_minutes': avgh
        }

    return {
        'total_open': len(opens_map),
        'total_close': len(closes_map),
        'open_trades': total_open_only,
        'closed_trades': total_closed,
        'win_rate_pct': win_rate,
        'avg_pnl_pct': avg_pnl,
        'avg_hold_minutes': avg_hold,
        'by_direction': dir_report,
        'outcomes': dict(outcomes),
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze paper trade CSV logs')
    parser.add_argument('--log', required=True, help='Path to CSV log')
    parser.add_argument('--from', dest='start', default=None, help='Start datetime (ISO)')
    parser.add_argument('--to', dest='end', default=None, help='End datetime (ISO)')
    args = parser.parse_args()

    df = load_csv(args.log)
    if args.start or args.end:
        df = filter_date_range(df, args.start, args.end)

    if df.empty:
        print('No data in selected range.')
        return

    metrics = compute_metrics(df)

    print('=== PAPER TRADE SUMMARY ===')
    print(f"Total OPEN: {metrics['total_open']}")
    print(f"Total CLOSE: {metrics['total_close']}")
    print(f"Open trades (still running): {metrics['open_trades']}")
    print(f"Closed trades: {metrics['closed_trades']}")
    print(f"Win rate: {metrics['win_rate_pct']:.2f}%")
    print(f"Avg PnL % (closed): {metrics['avg_pnl_pct']:.3f}")
    print(f"Avg hold (min): {metrics['avg_hold_minutes']:.1f}")
    print('--- Outcomes ---')
    for k,v in metrics['outcomes'].items():
        print(f"  {k}: {v}")
    print('--- By Direction ---')
    for d, rep in metrics['by_direction'].items():
        print(f"  {d}: trades={rep['trades']} | win_rate={rep['win_rate_pct']:.2f}% | avg_pnl={rep['avg_pnl_pct']:.3f}% | avg_hold={rep['avg_hold_minutes']:.1f}m")

if __name__ == '__main__':
    main()
