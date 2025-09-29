#!/usr/bin/env python3
"""
Script untuk verifikasi jumlah data setelah pipeline selesai.
Memastikan data yang diambil sesuai dengan estimasi setelah dikurangi dropna().

Usage:
    python verify_data_count.py --help
    python verify_data_count.py --config config/config.yaml
"""

import sqlite3
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import yaml

def load_config(config_path: str = 'config/config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def calculate_expected_data(config):
    """Hitung estimasi data yang seharusnya ada setelah dropna()"""
    # Ambil parameter dari config
    months = config.get('months', 12)
    bb_periods = config.get('bb_periods', [96])
    horizon = config.get('targets', {}).get('horizon', 20)

    # Gunakan BB period terbesar untuk estimasi konservatif
    bb_period = max(bb_periods) if bb_periods else 96

    days = months * 30.44  # rata-rata bulan
    hours_per_day = 24

    # Untuk 5m timeframe
    candle_per_hour_5m = 60 // 5  # 12 candle per jam
    total_candle_5m = int(days * hours_per_day * candle_per_hour_5m)
    final_5m = total_candle_5m - bb_period - horizon

    # Untuk 15m timeframe
    candle_per_hour_15m = 60 // 15  # 4 candle per jam
    total_candle_15m = int(days * hours_per_day * candle_per_hour_15m)
    final_15m = total_candle_15m - bb_period - horizon

    return {
        '5m': {'total_raw': total_candle_5m, 'final_expected': final_5m},
        '15m': {'total_raw': total_candle_15m, 'final_expected': final_15m}
    }
    """Hitung estimasi data yang seharusnya ada setelah dropna()"""
    days = months * 30.44  # rata-rata bulan
    hours_per_day = 24

    # Untuk 5m timeframe
    candle_per_hour_5m = 60 // 5  # 12 candle per jam
    total_candle_5m = int(days * hours_per_day * candle_per_hour_5m)
    final_5m = total_candle_5m - bb_period - horizon

    # Untuk 15m timeframe
    candle_per_hour_15m = 60 // 15  # 4 candle per jam
    total_candle_15m = int(days * hours_per_day * candle_per_hour_15m)
    final_15m = total_candle_15m - bb_period - horizon

    return {
        '5m': {'total_raw': total_candle_5m, 'final_expected': final_5m},
        '15m': {'total_raw': total_candle_15m, 'final_expected': final_15m}
    }

def check_actual_data(db_dir: str = 'data/db'):
    """Cek data aktual di database"""
    results = {}

    for tf in ['5m', '15m']:
        db_path = os.path.join(db_dir, f'btc_{tf}.sqlite')
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                count = conn.execute('SELECT COUNT(*) FROM features').fetchone()[0]

                # Get time range
                result = conn.execute('SELECT MIN(timestamp), MAX(timestamp) FROM features').fetchone()
                if result[0] and result[1]:
                    start_ts = datetime.fromisoformat(result[0].replace('Z', '+00:00'))
                    end_ts = datetime.fromisoformat(result[1].replace('Z', '+00:00'))
                    days_diff = (end_ts - start_ts).days

                    results[tf] = {
                        'count': count,
                        'days': days_diff,
                        'start': result[0],
                        'end': result[1]
                    }
                conn.close()
            except Exception as e:
                print(f"Error checking {tf}: {e}")
                results[tf] = None
        else:
            results[tf] = None

    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Verifikasi jumlah data pipeline setelah build_dataset.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python verify_data_count.py                    # Gunakan config default
  python verify_data_count.py --config config/config.yaml
  python verify_data_count.py --help
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path ke file konfigurasi YAML (default: config/config.yaml)'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    print("=== VERIFIKASI JUMLAH DATA PIPELINE ===\n")

    # Load config
    config = load_config(args.config)
    if not config:
        print(f"❌ Gagal memuat config: {args.config}")
        return

    # Ambil parameter dari config
    months = config.get('months', 12)
    bb_periods = config.get('bb_periods', [96])
    horizon = config.get('targets', {}).get('horizon', 20)
    bb_period = max(bb_periods) if bb_periods else 96

    # Hitung estimasi
    expected = calculate_expected_data(config)

    print(f"KONFIGURASI: {args.config}")
    print("PARAMETER DARI CONFIG:")
    print(f"  Months: {months}")
    print(f"  BB periods: {bb_periods} (max: {bb_period})")
    print(f"  Horizon: {horizon}\n")

    print("ESTIMASI DATA YANG SEHARUSNYA ADA:")
    for tf, data in expected.items():
        print(f"  {tf} timeframe:")
        print(f"    Total candle mentah: {data['total_raw']:,}")
        print(f"    Final setelah dropna: {data['final_expected']:,}")
    print()

    # Cek data aktual
    actual = check_actual_data()

    print("DATA AKTUAl DI DATABASE:")
    for tf, data in actual.items():
        if data:
            print(f"  {tf} timeframe:")
            print(f"    Total rows: {data['count']:,}")
            print(f"    Time range: {data['days']} days")
            print(f"    From: {data['start']}")
            print(f"    To: {data['end']}")

            # Bandingkan dengan estimasi
            expected_count = expected[tf]['final_expected']
            actual_count = data['count']
            diff = actual_count - expected_count
            pct_diff = (diff / expected_count) * 100 if expected_count > 0 else 0

            print(f"    Expected: {expected_count:,}")
            print(f"    Difference: {diff:,} ({pct_diff:+.1f}%)")

            if abs(pct_diff) < 5:  # Toleransi 5%
                print("    ✅ Status: OK (dalam toleransi)")
            else:
                print("    ⚠️  Status: PERLU DIPERIKSA")
        else:
            print(f"  {tf} timeframe: Database tidak ditemukan")
        print()

if __name__ == "__main__":
    main()