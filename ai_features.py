import numpy as np
import pandas as pd
import ta

FEATURE_COLUMNS = [
    'rsi_14', 'rsi_slope',
    'macd_diff', 'macd_diff_slope',
    'adx', 'adx_pos_minus_neg',
    'bb_pctb',
    'atr_pct',
    'stoch_k',
    'cci',
    'mfi',
    'ewo',
    'keltner_pctk',
    'obv_slope',
    'vol_ratio',
    'ret_3', 'ret_8', 'ret_20', 'ret_50',
    'ema_fast_slow_gap',
]

PREDICT_HORIZON = 4  # 30분봉 기준 다음 4개 캔들(=2시간) 뒤 방향을 예측


def compute_features(df):
    """OHLCV df(open/high/low/close/volume, 시간순 정렬)에서 학습/추론에 공통으로 쓰는
    피처를 계산한다. 학습(train_ai_predictor.py)과 실시간 추론(ai_predictor.py)이
    반드시 이 함수 하나만 거치게 해서 train/serve 스큐를 원천 차단한다."""
    df = df.copy()
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    close, high, low, volume = df['close'], df['high'], df['low'], df['volume']

    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df['rsi_14'] = rsi
    df['rsi_slope'] = rsi.diff(3)

    macd = ta.trend.MACD(close=close)
    macd_diff = macd.macd_diff()
    df['macd_diff'] = macd_diff
    df['macd_diff_slope'] = macd_diff.diff(3)

    adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    df['adx'] = adx_ind.adx()
    df['adx_pos_minus_neg'] = adx_ind.adx_pos() - adx_ind.adx_neg()

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0)
    bb_range = (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, np.nan)
    df['bb_pctb'] = (close - bb.bollinger_lband()) / bb_range

    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    df['atr_pct'] = atr / close * 100.0

    df['stoch_k'] = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3).stoch()

    df['cci'] = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20).cci()

    df['mfi'] = ta.volume.MFIIndicator(high=high, low=low, close=close, volume=volume, window=14).money_flow_index()

    sma_fast = close.rolling(5).mean()
    sma_slow = close.rolling(35).mean()
    df['ewo'] = (sma_fast - sma_slow) / close * 100.0

    kc = ta.volatility.KeltnerChannel(high=high, low=low, close=close, window=20, window_atr=20, multiplier=2.0)
    kc_range = (kc.keltner_channel_hband() - kc.keltner_channel_lband()).replace(0, np.nan)
    df['keltner_pctk'] = (close - kc.keltner_channel_lband()) / kc_range

    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df['obv_slope'] = obv.diff(5) / (obv.abs().rolling(20).mean() + 1e-9)

    vol_ma = volume.rolling(20).mean()
    df['vol_ratio'] = volume / (vol_ma + 1e-9)

    df['ret_3'] = close.pct_change(3) * 100.0
    df['ret_8'] = close.pct_change(8) * 100.0
    df['ret_20'] = close.pct_change(20) * 100.0
    df['ret_50'] = close.pct_change(50) * 100.0

    ema_fast = ta.trend.ema_indicator(close, window=12)
    ema_slow = ta.trend.ema_indicator(close, window=50)
    df['ema_fast_slow_gap'] = (ema_fast - ema_slow) / close * 100.0

    return df


def make_labels(df, horizon=PREDICT_HORIZON):
    future_close = df['close'].shift(-horizon)
    label = (future_close > df['close']).astype(int)
    return label
