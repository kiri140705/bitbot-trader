"""
'AI 예측 연구실' 탭에 쓰이는 모델을 학습시키는 스크립트.

절대 랜덤 셔플로 train/test를 나누지 않는다 - 시계열 데이터는 미래 정보가 과거로
새어 들어가면(look-ahead bias) 검증 정확도가 뻥튀기되므로, 반드시 시간순으로
앞부분은 학습, 뒷부분(한 번도 학습에 안 쓰인 미래)은 검증에만 쓴다.

python train_ai_predictor.py
"""
import json
import ssl
import time
import urllib.parse
import urllib.request

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

from ai_features import compute_features, make_labels, FEATURE_COLUMNS, PREDICT_HORIZON

SYMBOL = 'BTCUSDT'
GRANULARITY = '30m'
MONTHS_OF_HISTORY = 18
MODEL_PATH = 'ai_predictor_model.txt'
META_PATH = 'ai_predictor_meta.json'


def fetch_candles(months):
    end_time = int(time.time() * 1000)
    start_time = end_time - (months * 30 * 24 * 60 * 60 * 1000)
    all_candles = []
    current_end = end_time
    ctx = ssl._create_unverified_context()
    retry_left = 5
    while current_end > start_time:
        params = {'symbol': SYMBOL, 'productType': 'USDT-FUTURES', 'granularity': GRANULARITY,
                  'endTime': str(current_end), 'limit': '200'}
        q = urllib.parse.urlencode(params)
        url = f"https://api.bitget.com/api/v2/mix/market/history-candles?{q}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            retry_left -= 1
            if retry_left <= 0:
                print(f"다운로드 중단(재시도 소진): {e}")
                break
            time.sleep(2)
            continue
        candles = data.get('data', [])
        if not candles:
            break
        all_candles = candles + all_candles
        oldest_ts = int(candles[0][0])
        if oldest_ts >= current_end:
            break
        current_end = oldest_ts - 1
        time.sleep(0.03)
    return all_candles


def main():
    print(f"{MONTHS_OF_HISTORY}개월치 {SYMBOL} {GRANULARITY} 캔들 다운로드 중...")
    candles = fetch_candles(MONTHS_OF_HISTORY)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    df['timestamp'] = df['timestamp'].astype(int)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    print(f"{len(df)}개 캔들 확보 ({pd.to_datetime(df['timestamp'].iloc[0], unit='ms')} ~ {pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')})")

    print("피처 계산 중...")
    df = compute_features(df)
    df['label'] = make_labels(df, horizon=PREDICT_HORIZON)

    df_clean = df.dropna(subset=FEATURE_COLUMNS + ['label']).reset_index(drop=True)
    print(f"결측치 제거 후 {len(df_clean)}행 사용 가능")

    n = len(df_clean)
    train_end = int(n * 0.80)
    eval_start = int(n * 0.72)  # train 구간의 마지막 8%를 early-stopping용 eval set으로 분리 (역시 시간순)

    train_df = df_clean.iloc[:eval_start]
    eval_df = df_clean.iloc[eval_start:train_end]
    test_df = df_clean.iloc[train_end:]  # 학습 과정에 단 한 번도 안 쓰인 완전 미래 구간

    print(f"train={len(train_df)}  eval(early-stop)={len(eval_df)}  test(홀드아웃)={len(test_df)}")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df['label']
    X_eval, y_eval = eval_df[FEATURE_COLUMNS], eval_df['label']
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df['label']

    print(f"train 라벨 분포: 상승 {y_train.mean()*100:.1f}%")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_eval, y_eval)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_proba)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    baseline = max(y_test.mean(), 1 - y_test.mean())  # "항상 다수 클래스로 찍기"의 정확도

    print("\n=== 홀드아웃(완전 미검증 미래 데이터) 성능 ===")
    print(f"정확도(Accuracy): {acc*100:.2f}%   (참고: 무작정 다수쪽 찍기 정확도 {baseline*100:.2f}%)")
    print(f"AUC: {auc:.4f}")
    print(f"Precision(상승 예측시 실제 상승 비율): {prec*100:.2f}%")
    print(f"Recall: {rec*100:.2f}%")

    importances = dict(zip(FEATURE_COLUMNS, [float(x) for x in model.feature_importances_]))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:6]
    print("\n주요 피처 중요도 Top 6:", top_features)

    model.booster_.save_model(MODEL_PATH)

    meta = {
        'symbol': SYMBOL,
        'granularity': GRANULARITY,
        'predict_horizon_candles': PREDICT_HORIZON,
        'feature_columns': FEATURE_COLUMNS,
        'trained_at': pd.Timestamp.utcnow().isoformat(),
        'data_range': {
            'start': str(pd.to_datetime(df['timestamp'].iloc[0], unit='ms')),
            'end': str(pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')),
            'n_candles': int(len(df)),
        },
        'holdout_metrics': {
            'accuracy': float(acc),
            'baseline_majority_accuracy': float(baseline),
            'auc': float(auc),
            'precision': float(prec),
            'recall': float(rec),
            'n_test_samples': int(len(test_df)),
        },
        'top_features': top_features,
    }
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n모델 저장 완료: {MODEL_PATH}")
    print(f"메타데이터 저장 완료: {META_PATH}")


if __name__ == '__main__':
    main()
