import os
import re
import time

GEMINI_CONFIG_FILE = 'gemini_setting.txt'
MODEL_NAME = 'gemini-flash-latest'  # 구글이 계속 최신 flash 모델을 가리키도록 관리하는 별칭 - 특정 버전을 고정하면 모델이 구버전으로 사라질 때(예: gemini-2.5-flash 신규키 차단) 404가 나므로 별칭을 쓴다

_client = None


def _ensure_config_file():
    if not os.path.exists(GEMINI_CONFIG_FILE):
        content = """GEMINI_API_KEY=여기에_발급받은_키를_붙여넣으세요

# ==============================================================================
# AI 예측 연구실 - Gemini AI 브리핑(전략 해설 / 뉴스 심리 / 매매 브리핑) 사용 방법
# ==============================================================================
# "AI 예측 연구실" 탭의 Gemini AI 인사이트(전략 해설, 실시간 뉴스 심리, 오늘의 매매
# 브리핑)를 받아보려면 본인 명의의 무료 Gemini API 키가 1개 필요합니다.
#
# [발급 경로]
# 1. https://aistudio.google.com/apikey 접속 (본인 구글 계정으로 로그인)
# 2. "API 키 만들기" 버튼 클릭 -> 즉시 무료로 키가 발급됩니다.
# 3. 발급된 키를 복사해서 위 GEMINI_API_KEY= 뒤에 붙여넣고 이 파일을 저장하세요.
# 4. 프로그램을 재시작하면 AI 예측 연구실 탭에 자동으로 반영됩니다.
#
# * 키는 본인 계정 전용이며 타인과 공유하지 마세요.
# * 무료 티어로 충분하며, 별도 결제수단 등록은 필요하지 않습니다.
# ==============================================================================
"""
        with open(GEMINI_CONFIG_FILE, 'w', encoding='utf-8') as f:
            f.write(content)


def get_api_key():
    _ensure_config_file()
    try:
        with open(GEMINI_CONFIG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('GEMINI_API_KEY='):
                    key = line.split('=', 1)[1].strip()
                    if key and '여기에' not in key:
                        return key
    except Exception:
        pass
    return None


def is_configured():
    return get_api_key() is not None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        api_key = get_api_key()
        if not api_key:
            raise RuntimeError("Gemini API 키가 설정되지 않았습니다 (gemini_setting.txt 확인)")
        _client = genai.Client(api_key=api_key)
    return _client


def _generate(prompt, max_retries=2):
    client = _get_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            text = (response.text or "").strip()
            if text:
                return text
            last_err = RuntimeError("빈 응답")
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise last_err


# ==================== 1. 전략(예측) 자연어 설명 ====================

_FEATURE_NAME_KR = {
    'rsi_14': 'RSI', 'rsi_slope': 'RSI 변화 속도',
    'macd_diff': 'MACD 모멘텀', 'macd_diff_slope': 'MACD 모멘텀 변화 속도',
    'adx': 'ADX 추세 강도', 'adx_pos_minus_neg': 'ADX 방향성 우위',
    'bb_pctb': '볼린저밴드 위치', 'atr_pct': 'ATR 변동성',
    'stoch_k': '스토캐스틱', 'cci': 'CCI', 'mfi': 'MFI', 'ewo': '엘리어트 파동 오실레이터(EWO)',
    'keltner_pctk': '켈트너 채널 위치', 'obv_slope': 'OBV 거래량 추세', 'vol_ratio': '거래량 비율',
    'ret_3': '3봉 전 대비 수익률', 'ret_8': '8봉 전 대비 수익률', 'ret_20': '20봉 전 대비 수익률',
    'ret_50': '50봉 전 대비 수익률', 'ema_fast_slow_gap': '단/장기 이평선 격차',
}


def explain_strategy(up_prob, down_prob, features, top_feature_keys):
    lines = []
    for k in top_feature_keys:
        if k in features:
            name = _FEATURE_NAME_KR.get(k, k)
            lines.append(f"- {name}: {features[k]:.3f}")
    feature_text = "\n".join(lines)

    prompt = f"""당신은 암호화폐 트레이딩 데이터를 알기 쉽게 설명하는 애널리스트입니다.

아래는 BTC/USDT 30분봉 기준으로 통계 모델(LightGBM)이 계산한 다음 2시간 방향 확률과, 그 판단에 영향을 준 주요 기술 지표의 현재값입니다.

상승 확률: {up_prob*100:.1f}%
하락 확률: {down_prob*100:.1f}%

주요 지표 현재값:
{feature_text}

이 데이터를 바탕으로 지금 시장 상태를 트레이딩 초보자도 이해할 수 있게 2~3문장, 한국어로 자연스럽게 설명해주세요.
반드시 지켜야 할 것:
- 이건 확정적인 매매 신호가 아니라 참고용 통계 데이터라는 뉘앙스를 유지할 것 (과도한 확신 금지)
- "매수하세요/매도하세요" 같은 직접적인 투자 권유 표현은 쓰지 말 것
- 마크다운 기호(*, # 등) 없이 순수 텍스트로만 답변할 것
"""
    return _generate(prompt)


# ==================== 2. 뉴스 감성분석 ====================

def fetch_crypto_news_headlines(limit=10):
    import requests
    import xml.etree.ElementTree as ET
    url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    try:
        resp = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        root = ET.fromstring(resp.content)
        headlines = []
        for item in root.findall('.//item')[:limit]:
            title = item.find('title')
            if title is not None and title.text:
                headlines.append(title.text.strip())
        return headlines
    except Exception:
        return []


def analyze_news_sentiment(headlines):
    if not headlines:
        return {'score': 50, 'summary': '분석할 뉴스를 가져오지 못했습니다.', 'headlines': []}

    headline_text = "\n".join(f"- {h}" for h in headlines)
    prompt = f"""다음은 최근 암호화폐 관련 영어 뉴스 헤드라인 목록입니다:

{headline_text}

이 뉴스들의 전반적인 시장 심리(투자 심리)를 분석해서, 아래 형식 그대로만 한국어로 답변하세요 (다른 문장 추가 금지):
점수: [0~100 사이 정수, 0=매우 부정적, 50=중립, 100=매우 긍정적]
요약: [왜 이런 점수인지 한국어로 한 문장]
"""
    text = _generate(prompt)
    score = 50
    summary = text
    try:
        score_match = re.search(r'점수\s*:\s*(\d+)', text)
        if score_match:
            score = max(0, min(100, int(score_match.group(1))))
        summary_match = re.search(r'요약\s*:\s*(.+)', text)
        if summary_match:
            summary = summary_match.group(1).strip()
    except Exception:
        pass
    return {'score': score, 'summary': summary, 'headlines': headlines[:5]}


# ==================== 3. 오늘의 리포트 요약 ====================

def summarize_daily_report(trades_summary_text):
    if not trades_summary_text or not trades_summary_text.strip():
        return "오늘은 아직 체결된 거래가 없습니다."

    prompt = f"""다음은 오늘 자동매매 봇의 거래 기록입니다:

{trades_summary_text}

이 데이터를 바탕으로 오늘 하루 매매 성과를 트레이더에게 브리핑하듯 2~3문장, 한국어로 요약해주세요.
전문 용어는 최소화하고 담백하게 작성하고, 마크다운 기호 없이 순수 텍스트로만 답변하세요.
"""
    return _generate(prompt)
