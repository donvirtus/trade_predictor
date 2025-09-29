#!/usr/bin/env python3
"""
Entry point to collect and preprocess datasets into separate SQLite files per timeframe.

Usage examples:
  python3 scripts/collect_preprocess.py --config config/config.yaml
  python3 scripts/collect_preprocess.py --config config/config.yaml --years 1 --no-force-full-rebuild
  python3 scripts/collect_preprocess.py --config config/config.yaml --months 13 --force-full-rebuild --no-retain-full-history

This wraps pipeline.build_dataset.build_separate_timeframes.
"""
import argparse
import sys

from pipeline.build_dataset import build_separate_timeframes

def main():
    parser = argparse.ArgumentParser(description='Collect & preprocess OHLCV + features per timeframe (separate SQLite files)')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to YAML config')
    parser.add_argument('--base-output', type=str, default='data/db', help='Base output directory for SQLite files')
    # Pass-through overrides recognized inside build_separate_timeframes
    parser.add_argument('--months', type=int, default=None, help='Override months of history (fallback when years/start not provided)')
    parser.add_argument('--years', type=int, default=None, help='Prefer historical window in years (overrides months when set)')
    parser.add_argument('--force-full-rebuild', dest='force_full_rebuild', action='store_true', help='Force full rebuild (disables incremental)')
    parser.add_argument('--no-force-full-rebuild', dest='force_full_rebuild', action='store_false', help='Do not force full rebuild')
    parser.set_defaults(force_full_rebuild=None)
    parser.add_argument('--retain-full-history', dest='retain_full_history', action='store_true', help='Keep all history, do not trim to months window')
    parser.add_argument('--no-retain-full-history', dest='retain_full_history', action='store_false', help='Trim history to months window')
    parser.set_defaults(retain_full_history=None)

    args, unknown = parser.parse_known_args()

    # Delegate to build function. It internally inspects sys.argv for overrides like --months, --force-full-rebuild, etc.
    # We pass cfg path and output dir explicitly.
    build_separate_timeframes(cfg_path=args.config, base_output_dir=args.base_output)

if __name__ == '__main__':
    main()
