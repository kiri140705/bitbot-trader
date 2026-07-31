import json
import os
import ssl
import time
import urllib.parse
import urllib.request

import lightgbm as lgb
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from ai_features import compute_features, FEATURE_COLUMNS

MODEL_PATH = 'ai_predictor_model.txt'
META_PATH = 'ai_predictor_meta.json'

_SSL_CTX = ssl._create_unverified_context()


def model_files_exist():
    return os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)


def load_meta():
    with open(META_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _fetch_recent_candles(symbol, granularity, limit=200):
    end_time = int(time.time() * 1000)
    params = {'symbol': symbol, 'productType': 'USDT-FUTURES', 'granularity': granularity,
              'endTime': str(end_time), 'limit': str(limit)}
    q = urllib.parse.urlencode(params)
    url = f"https://api.bitget.com/api/v2/mix/market/history-candles?{q}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as response:
        data = json.loads(response.read().decode())
    candles = data.get('data', [])
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    df['timestamp'] = df['timestamp'].astype(int)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df


class AIPredictionThread(QThread):
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, interval_sec=60):
        super().__init__()
        self.interval_sec = interval_sec
        self.is_running = True
        self.booster = None
        self.meta = None

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            self.booster = lgb.Booster(model_file=MODEL_PATH)
            self.meta = load_meta()
        except Exception as e:
            self.error_occurred.emit(f"모델 로딩 실패: {e}")
            return

        symbol = self.meta['symbol']
        granularity = self.meta['granularity']

        while self.is_running:
            try:
                df = _fetch_recent_candles(symbol, granularity, limit=200)
                feat_df = compute_features(df)
                latest = feat_df.iloc[[-1]][FEATURE_COLUMNS]
                if latest.isna().any(axis=1).iloc[0]:
                    raise ValueError("최신 캔들 피처에 결측치가 있습니다 (데이터 부족)")

                up_prob = float(self.booster.predict(latest)[0])
                down_prob = 1.0 - up_prob

                latest_vals = {k: float(v) for k, v in latest.iloc[0].to_dict().items()}

                self.result_ready.emit({
                    'up_prob': up_prob,
                    'down_prob': down_prob,
                    'latest_close': float(df['close'].iloc[-1]),
                    'latest_features': latest_vals,
                    'ts': time.time(),
                })
            except Exception as e:
                self.error_occurred.emit(str(e))

            for _ in range(self.interval_sec * 2):
                if not self.is_running:
                    break
                time.sleep(0.5)
