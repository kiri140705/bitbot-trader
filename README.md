# BitBot Trader — Gemini AI Integration

BitBot Trader는 Bitget 선물 API와 연동되는 비트코인 자동매매 데스크톱 플랫폼입니다.
이 저장소는 **본 플랫폼에 실제로 통합된 Gemini API 연동 코드**를 공개용으로 정리한 것입니다.

## 이 저장소에 무엇이 있나

전체 애플리케이션(매매 진입/청산 로직, 리스크 관리, 커스텀 지표 39종, 백테스트 엔진 등)은
유료 구독 상품의 핵심 IP이기 때문에 이 저장소에는 포함하지 않았습니다. 대신 **Gemini API를
실제로 어떻게 활용하는지 확인할 수 있는 부분만** 공개합니다:

| 파일 | 역할 |
|---|---|
| [`gemini_client.py`](./gemini_client.py) | Gemini API 클라이언트. 전략 해설 / 뉴스 감성분석 / 매매 브리핑 생성 프롬프트와 응답 파싱 |
| [`gemini_insight.py`](./gemini_insight.py) | 위 3개 기능을 백그라운드에서 주기적으로 실행하는 PyQt6 `QThread` 3종 |
| [`ai_predictor.py`](./ai_predictor.py) | LightGBM 기반 가격 방향 예측 모델의 실시간 추론 스레드 (Gemini가 이 결과를 자연어로 풀어서 설명) |
| [`ai_features.py`](./ai_features.py) | 예측 모델에 입력되는 기술적 지표(RSI, MACD, ADX 등) 피처 엔지니어링 |
| [`train_ai_predictor.py`](./train_ai_predictor.py) | 예측 모델 학습 스크립트 (시계열 워크포워드 검증, 셔플 없음) |

## Gemini API 활용 기능 3가지

앱의 "AI 예측 연구실" 탭에서 다음 3가지를 Gemini(`gemini-flash-latest`)로 생성합니다.

1. **전략 해설** — LightGBM 예측 결과(상승/하락 확률)와 주요 기술 지표 값을 Gemini에게 전달해,
   초보자도 이해할 수 있는 자연어 시장 해설을 생성합니다. (`gemini_client.explain_strategy`)
2. **실시간 뉴스 심리** — 암호화폐 뉴스 헤드라인을 수집해 Gemini로 시장 심리 점수(0~100)와
   요약을 생성합니다. (`gemini_client.analyze_news_sentiment`)
3. **오늘의 매매 브리핑** — 당일 체결된 거래 내역을 집계해 Gemini가 트레이더에게 브리핑하듯
   요약합니다. (`gemini_client.summarize_daily_report`)

세 기능 모두 "확정적 매매 신호가 아닌 참고용 통계 지표"라는 뉘앙스를 유지하도록 프롬프트에
명시적으로 지시하고 있으며, 무료 티어 한도 안에서 안전하게 동작하도록 호출 주기를 15분/90분/3시간으로
설계했습니다.

## 참고

이 저장소는 실행 가능한 완전한 애플리케이션이 아니라, Gemini 연동 부분을 검토하기 위한 코드
스니펫 모음입니다. `requirements.txt`는 이 저장소에 포함된 파일들의 의존성만 기준으로 작성했습니다.
