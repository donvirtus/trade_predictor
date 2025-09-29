import os
import sqlite3
import argparse
import pandas as pd

def summarize_db(db_path: str):
    if not os.path.exists(db_path):
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query("SELECT timestamp FROM features", conn, parse_dates=['timestamp'])
            if df.empty:
                return {"rows": 0}
            rows = len(df)
            min_ts = df['timestamp'].min()
            max_ts = df['timestamp'].max()
            dup = df.duplicated(subset=['timestamp']).sum()
            return {"rows": rows, "min": str(min_ts), "max": str(max_ts), "dups": int(dup)}
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Compare SQLite DB summaries (features table)")
    parser.add_argument('--dir-a', required=True, help='First directory containing timeframe DBs')
    parser.add_argument('--dir-b', required=True, help='Second directory containing timeframe DBs')
    parser.add_argument('--pairs', nargs='*', default=['BTCUSDT'], help='Pairs to check (default: BTCUSDT)')
    parser.add_argument('--timeframes', nargs='*', default=['5m','15m','30m','1h','2h','4h','6h'], help='Timeframes to check')
    args = parser.parse_args()

    def db_name(pair, tf):
        return f"{pair.replace('USDT','').lower()}_{tf}.sqlite"

    print("=== DB Summary Comparison ===")
    for pair in args.pairs:
        for tf in args.timeframes:
            name = db_name(pair, tf)
            a_path = os.path.join(args.dir_a, name)
            b_path = os.path.join(args.dir_b, name)
            a = summarize_db(a_path)
            b = summarize_db(b_path)
            print(f"\n[{pair} {tf}]\nA: {a_path}\n   {a}\nB: {b_path}\n   {b}")

if __name__ == '__main__':
    main()
