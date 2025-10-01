"""
Utility functions untuk path dan naming conventions yang konsisten
untuk support multiple trading pairs dalam project.

Created: 2025-09-29
Author: AI Assistant
"""

import os
from typing import Dict, Optional


def normalize_pair_name(pair: str) -> str:
    """
    Normalize trading pair name untuk naming consistency
    
    Args:
        pair: Trading pair seperti 'BTCUSDT', 'ETHUSDT', 'BTCUSD'
    
    Returns:
        Normalized pair name (e.g., 'btc', 'eth')
    """
    # Remove common quote currencies and convert to lowercase
    pair_clean = pair.upper()
    for quote in ['USDT', 'USDC', 'USD', 'BTC', 'ETH']:
        if pair_clean.endswith(quote):
            pair_clean = pair_clean[:-len(quote)]
            break
    
    return pair_clean.lower()


def generate_db_path(pair: str, timeframe: str, base_dir: str = 'data/db') -> str:
    """
    Generate database file path untuk specific pair and timeframe
    
    Args:
        pair: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
        timeframe: Timeframe (e.g., '5m', '1h', '4h')
        base_dir: Base directory untuk database files
    
    Returns:
        Complete path to database file
    """
    pair_clean = normalize_pair_name(pair)
    filename = f"{pair_clean}_{timeframe}.sqlite"
    return os.path.join(base_dir, filename)


def resolve_dataset_base_dir(config=None, override: Optional[str] = None, default: str = 'data/db') -> str:
    """
    Determine dataset base directory from override, config, or default value.

    Args:
        config: Configuration object with optional output section
        override: Explicit directory override (e.g., CLI argument)
        default: Fallback directory when nothing else provided

    Returns:
        Base directory path for per-timeframe SQLite databases
    """
    if override:
        return override

    if config is not None:
        output_cfg = getattr(config, 'output', {}) or {}
        base_dir = output_cfg.get('base_dir')
        if base_dir:
            return base_dir

        sqlite_path = output_cfg.get('sqlite_db')
        if sqlite_path:
            derived_dir = os.path.dirname(sqlite_path)
            if derived_dir:
                return derived_dir

    return default


def resolve_feature_table(config=None, default: str = 'features') -> str:
    """Resolve feature table name from config output section."""
    if config is not None:
        output_cfg = getattr(config, 'output', {}) or {}
        table_name = output_cfg.get('feature_table')
        if table_name:
            return table_name
    return default


def resolve_metadata_table(config=None, default: str = 'metadata') -> str:
    """Resolve metadata table name from config output section."""
    if config is not None:
        output_cfg = getattr(config, 'output', {}) or {}
        table_name = output_cfg.get('metadata_table')
        if table_name:
            return table_name
    return default


def generate_model_filename(pair: str, timeframe: str, model_type: str, 
                           target: str = 'direction', timestamp: str = None,
                           extension: str = None) -> str:
    """
    Generate model filename dengan consistent naming convention
    
    Args:
        pair: Trading pair
        timeframe: Timeframe
        model_type: Model type ('lightgbm', 'xgboost', 'catboost', etc.)
        target: Target type ('direction', 'direction_h5', etc.)
        timestamp: Optional timestamp untuk versioning
        extension: File extension (auto-detected if None)
    
    Returns:
        Model filename
    """
    pair_clean = normalize_pair_name(pair)
    
    # Auto-detect extension based on model type
    if extension is None:
        ext_map = {
            'lightgbm': '.txt',
            'xgboost': '.json', 
            'catboost': '.cbm',
            'scaler': '.joblib'
        }
        extension = ext_map.get(model_type.lower(), '.joblib')
    
    # Build filename
    if timestamp:
        filename = f"{pair_clean}_{timeframe}_{model_type.lower()}_{target}_{timestamp}{extension}"
    else:
        filename = f"{pair_clean}_{timeframe}_{model_type.lower()}_{target}{extension}"
    
    return filename


def generate_metadata_filename(pair: str, timeframe: str, model_type: str,
                              target: str = 'direction') -> str:
    """
    Generate metadata filename untuk model
    
    Args:
        pair: Trading pair
        timeframe: Timeframe
        model_type: Model type
        target: Target type
    
    Returns:
        Metadata filename
    """
    pair_clean = normalize_pair_name(pair)
    return f"{pair_clean}_{timeframe}_{model_type.lower()}_{target}_metadata.json"


def get_pair_from_config(config, default: str = 'BTCUSDT') -> str:
    """
    Get primary trading pair dari configuration
    
    Args:
        config: Configuration object
        default: Default pair jika tidak ditemukan dalam config
    
    Returns:
        Primary trading pair
    """
    # Try different config locations
    if hasattr(config, 'data') and hasattr(config.data, 'symbol'):
        return config.data.symbol
    
    if hasattr(config, 'pairs') and config.pairs:
        return config.pairs[0]  # First pair as primary
    
    return default


# Mapping untuk backward compatibility
PAIR_ALIASES: Dict[str, str] = {
    'btc': 'BTCUSDT',
    'eth': 'ETHUSDT',
    'bitcoin': 'BTCUSDT',
    'ethereum': 'ETHUSDT'
}


def resolve_pair_alias(pair_input: str) -> str:
    """
    Resolve pair alias ke full pair name
    
    Args:
        pair_input: Pair input (could be alias atau full name)
    
    Returns:
        Full pair name
    """
    pair_lower = pair_input.lower()
    return PAIR_ALIASES.get(pair_lower, pair_input.upper())