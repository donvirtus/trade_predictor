import numpy as np
import pandas as pd


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.divide(b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _ewma(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = _safe_divide(avg_gain, avg_loss)
    return 100 - (100 / (1 + rs))


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr_components = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    plus_di = 100 * _safe_divide(plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean(), atr)
    minus_di = 100 * _safe_divide(minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean(), atr)
    dx = 100 * _safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
    return dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def add_price_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Bollinger Bands and related metrics (vectorised)
    for period in cfg.bb_periods:
        middle = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()
        middle_col = f'bb_{period}_middle'
        df[middle_col] = middle
        for dev in cfg.bb_devs:
            upper = middle + std * dev
            lower = middle - std * dev
            upper_col = f'bb_{period}_upper_{dev}'
            lower_col = f'bb_{period}_lower_{dev}'
            df[upper_col] = upper
            df[lower_col] = lower
            band_range = upper - lower
            df[f'bb_{period}_percent_b_{dev}'] = _safe_divide(close - lower, band_range)
            df[f'bb_{period}_bandwidth_{dev}'] = _safe_divide(band_range, middle)
        if cfg.bb_devs:
            widest = max(cfg.bb_devs)
            bw_col = f'bb_{period}_bandwidth_{widest}'
            rolling_quantile = df[bw_col].rolling(100, min_periods=20).quantile(0.2)
            df[f'bb_{period}_squeeze_flag'] = (df[bw_col] < rolling_quantile).astype('bool')

    # Moving averages (SMA & EMA)
    for p in cfg.ma_periods:
        sma = close.rolling(window=p, min_periods=p).mean()
        ema = _ewma(close, span=p)
        df[f'ma_{p}'] = sma
        df[f'sma_{p}'] = sma
        df[f'ema_{p}'] = ema

    # Range & volatility indicators
    df[f'price_range_{cfg.price_range_period}'] = (
        high.rolling(cfg.price_range_period, min_periods=1).max()
        - low.rolling(cfg.price_range_period, min_periods=1).min()
    )
    df[f'volatility_{cfg.volatility_period}'] = (
        close.pct_change(fill_method=None)
        .rolling(cfg.volatility_period, min_periods=cfg.volatility_period)
        .std()
    )

    # Momentum indicators
    df[f'rsi_{cfg.rsi_period}'] = _compute_rsi(close, cfg.rsi_period)

    fast, slow, signal = cfg.macd_params
    ema_fast = _ewma(close, span=fast)
    ema_slow = _ewma(close, span=slow)
    macd_line = ema_fast - ema_slow
    macd_signal = _ewma(macd_line, span=signal)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_line - macd_signal

    df[f'adx_{cfg.adx_period}'] = _compute_adx(high, low, close, cfg.adx_period)

    # Volume based indicators
    direction = np.sign(close.diff()).fillna(0)
    df['obv'] = (direction * volume).cumsum()
    cumulative_volume = volume.cumsum()
    df['vwap'] = _safe_divide((close * volume).cumsum(), cumulative_volume)

    return df
