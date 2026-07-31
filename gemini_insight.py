import time

from PyQt6.QtCore import QThread, pyqtSignal

import gemini_client


class StrategyExplainThread(QThread):
    """이미 학습된 LightGBM 예측(ai_predictor)의 결과를 Gemini가 자연어로 풀어서 설명한다.
    AIPredictionThread와 완전히 독립적으로 자체적으로 캔들/피처를 다시 계산한다 - 서로 다른
    스레드가 상태를 공유하지 않게 해서(이미 이 앱 전반에 쓰인 패턴) 한쪽이 느려지거나 실패해도
    다른 쪽에 영향이 없다."""
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, interval_sec=900):
        super().__init__()
        self.interval_sec = interval_sec
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        if not gemini_client.is_configured():
            self.error_occurred.emit("Gemini API 키가 설정되지 않았습니다 (gemini_setting.txt를 확인해주세요)")
            return

        from ai_predictor import model_files_exist, load_meta
        from ai_features import compute_features, FEATURE_COLUMNS
        import lightgbm as lgb
        from ai_predictor import _fetch_recent_candles

        if not model_files_exist():
            self.error_occurred.emit("AI 예측 모델이 아직 학습되지 않았습니다")
            return

        try:
            booster = lgb.Booster(model_file='ai_predictor_model.txt')
            meta = load_meta()
        except Exception as e:
            self.error_occurred.emit(f"모델 로딩 실패: {e}")
            return

        while self.is_running:
            try:
                df = _fetch_recent_candles(meta['symbol'], meta['granularity'], limit=200)
                feat_df = compute_features(df)
                latest = feat_df.iloc[[-1]][FEATURE_COLUMNS]
                if latest.isna().any(axis=1).iloc[0]:
                    raise ValueError("최신 캔들 피처에 결측치가 있습니다")

                up_prob = float(booster.predict(latest)[0])
                down_prob = 1.0 - up_prob
                latest_vals = {k: float(v) for k, v in latest.iloc[0].to_dict().items()}
                top_keys = [k for k, _ in meta.get('top_features', [])[:6]]

                explanation = gemini_client.explain_strategy(up_prob, down_prob, latest_vals, top_keys)
                self.result_ready.emit(explanation)
            except Exception as e:
                self.error_occurred.emit(str(e))

            for _ in range(self.interval_sec * 2):
                if not self.is_running:
                    break
                time.sleep(0.5)


class NewsSentimentThread(QThread):
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, interval_sec=5400):
        super().__init__()
        self.interval_sec = interval_sec
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        if not gemini_client.is_configured():
            self.error_occurred.emit("Gemini API 키가 설정되지 않았습니다 (gemini_setting.txt를 확인해주세요)")
            return

        while self.is_running:
            try:
                headlines = gemini_client.fetch_crypto_news_headlines(limit=10)
                result = gemini_client.analyze_news_sentiment(headlines)
                self.result_ready.emit(result)
            except Exception as e:
                self.error_occurred.emit(str(e))

            for _ in range(self.interval_sec * 2):
                if not self.is_running:
                    break
                time.sleep(0.5)


class DailyReportThread(QThread):
    """오늘(00:00 이후) 체결된 거래 내역을 거래소에서 조회해 Gemini로 브리핑 문장을 생성한다."""
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, exchange, symbol, interval_sec=10800):
        super().__init__()
        self.exchange = exchange
        self.symbol = symbol
        self.interval_sec = interval_sec
        self.is_running = True

    def stop(self):
        self.is_running = False

    def _build_today_summary_text(self):
        import datetime
        now = datetime.datetime.now()
        midnight = datetime.datetime(now.year, now.month, now.day)
        since_ms = int(midnight.timestamp() * 1000)

        params = {'productType': 'USDT-FUTURES'}
        hist = self.exchange.fetch_positions_history([self.symbol], since=since_ms, limit=100, params=params)
        if not hist:
            return "", 0, 0.0, 0

        wins, losses, total_pnl = 0, 0, 0.0
        lines = []
        for pos in hist:
            val = pos.get('pnl')
            if val is None:
                val = pos.get('info', {}).get('netProfit') or pos.get('info', {}).get('pnl')
            try:
                pnl = float(val) if val is not None else 0.0
            except Exception:
                pnl = 0.0
            if pnl == 0:
                continue
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            lines.append(f"거래 {'승' if pnl > 0 else '패'}: {pnl:+.2f} USDT")

        summary = "\n".join(lines) + f"\n\n총 거래 {wins + losses}건 (승 {wins} / 패 {losses}), 합산 손익 {total_pnl:+.2f} USDT"
        return summary, wins, total_pnl, losses

    def run(self):
        if not gemini_client.is_configured():
            self.error_occurred.emit("Gemini API 키가 설정되지 않았습니다 (gemini_setting.txt를 확인해주세요)")
            return

        while self.is_running:
            try:
                summary_text, wins, total_pnl, losses = self._build_today_summary_text()
                report = gemini_client.summarize_daily_report(summary_text)
                self.result_ready.emit(report)
            except Exception as e:
                self.error_occurred.emit(str(e))

            for _ in range(self.interval_sec * 2):
                if not self.is_running:
                    break
                time.sleep(0.5)
