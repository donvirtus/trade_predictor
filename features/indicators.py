import pandas as pd
import ta


def add_price_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    bb_periods = cfg.bb_periods
    bb_devs = cfg.bb_devs
    for period in bb_periods:
        base = ta.volatility.BollingerBands(df['close'], window=period, window_dev=bb_devs[0])
        df[f'bb_{period}_middle'] = base.bollinger_mavg()
        for dev in bb_devs:
            bb = ta.volatility.BollingerBands(df['close'], window=period, window_dev=dev)
            u = f'bb_{period}_upper_{dev}'
            l = f'bb_{period}_lower_{dev}'
            df[u] = bb.bollinger_hband()
            df[l] = bb.bollinger_lband()
            df[f'bb_{period}_percent_b_{dev}'] = (df['close'] - df[l]) / (df[u]-df[l])
            df[f'bb_{period}_bandwidth_{dev}'] = (df[u]-df[l]) / df[f'bb_{period}_middle']
        widest = max(bb_devs)
        bw_col = f'bb_{period}_bandwidth_{widest}'
        df[f'bb_{period}_squeeze_flag'] = df[bw_col] < df[bw_col].rolling(100).quantile(0.2)
    if cfg.ma_periods:
        for p in cfg.ma_periods:
            df[f'ma_{p}'] = ta.trend.SMAIndicator(df['close'], window=p).sma_indicator()
    df[f'price_range_{cfg.price_range_period}'] = df['high'].rolling(cfg.price_range_period).max() - df['low'].rolling(cfg.price_range_period).min()
    df[f'volatility_{cfg.volatility_period}'] = df['close'].pct_change().rolling(cfg.volatility_period).std()
    df[f'rsi_{cfg.rsi_period}'] = ta.momentum.RSIIndicator(df['close'], window=cfg.rsi_period).rsi()
    macd = ta.trend.MACD(df['close'], window_fast=cfg.macd_params[0], window_slow=cfg.macd_params[1], window_sign=cfg.macd_params[2])
    df['macd'] = macd.macd(); df['macd_signal'] = macd.macd_signal(); df['macd_histogram'] = macd.macd_diff()
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=cfg.adx_period)
    df[f'adx_{cfg.adx_period}'] = adx.adx()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['vwap'] = (df['close']*df['volume']).cumsum() / df['volume'].cumsum()
    return df
