from dataclasses import dataclass, field, asdict
import yaml
from typing import Any, Dict


@dataclass
class Config:
    # Core data settings
    pairs: list = field(default_factory=list)
    timeframes: list = field(default_factory=list)
    months: int = 3
    # Rebuild/retention flags
    # Default behavior when flags are absent: force full rebuild ON, retain full history OFF
    force_full_rebuild: bool = True
    retain_full_history: bool = False
    bb_periods: list = field(default_factory=lambda: [48])
    bb_devs: list = field(default_factory=lambda: [2.0])
    ma_periods: list = field(default_factory=list)
    price_range_period: int = 5
    volatility_period: int = 5
    adx_period: int = 14
    rsi_period: int = 14
    macd_params: list = field(default_factory=lambda: [12,26,9])
    lagged_periods: list = field(default_factory=lambda: [2])
    ma_bb_cross: dict = field(default_factory=dict)
    bb_slope: dict = field(default_factory=dict)

    # Label / target + external enrich
    target: dict = field(default_factory=dict)
    external: dict = field(default_factory=dict)
    multi_horizon: dict = field(default_factory=dict)

    # Optional output / misc sections
    output: dict = field(default_factory=dict)
    catboost: dict = field(default_factory=dict)
    db_mode: str = 'replace'
    limit_rows: Any = None
    fetch: dict = field(default_factory=dict)

    # Backward compatibility placeholders (legacy keys may exist)
    database: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)

    # Store any unrecognised keys to avoid data loss when re-saving
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_raw(raw: Dict[str, Any]) -> 'Config':
        known = {}
        for field_name in Config.__dataclass_fields__.keys():
            if field_name == '_extra':
                continue
            if field_name in raw:
                known[field_name] = raw[field_name]
        cfg = Config(**known)  # type: ignore[arg-type]
        # Anything not mapped becomes part of _extra
        extras = {k: v for k, v in raw.items() if k not in known}
        cfg._extra = extras
        return cfg


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return Config.from_raw(raw)


def save_config(cfg: Config, path: str):
    data = asdict(cfg)
    # Merge extras back so we don't drop unknown sections
    extras = data.pop('_extra', {}) or {}
    merged = {**extras, **data}
    with open(path, "w") as f:
        yaml.dump(merged, f, sort_keys=False)
