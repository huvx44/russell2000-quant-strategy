# 🚀 Alpha Stack: Russell 2000 Multi-Anomaly 복합 전략

## 전략 철학

기존 팩터 전략들(Value+Momentum, Quality+Growth 등)은 이미 ETF로 상품화되어 알파가 감소하고 있습니다. **Alpha Stack**은 학술적으로 검증되었지만 아직 대중화되지 않은 "숨겨진" 알파 소스들을 **겹겹이 쌓아(Stack)** 기존 전략 대비 초과수익을 추구합니다.

---

## 핵심 아이디어: 4개 알파 레이어

### Layer 1: 🎯 Earnings Surprise Drift (PEAD)
> **학술적 근거**: Bernard & Thomas (1990) - 연환산 ~35% 비정상 수익률

소형주에서 어닝 서프라이즈 후 가격이 완전히 조정되지 않고 수주~수개월간 드리프트하는 현상을 포착합니다. **소형주에서 특히 강력한 이유**: 애널리스트 커버리지가 낮아 정보 비대칭이 크고, 차익거래 비용이 높아 비효율성이 오래 지속됩니다.

- **측정**: SUE (Standardized Unexpected Earnings) = (실제 EPS - 예상 EPS) / 표준편차
- **신호**: SUE 상위 20% 매수, 어닝 발표 후 2일부터 60 거래일 보유

### Layer 2: 🕵️ Insider Buying + Short Interest Signal
> **학술적 근거**: Seyhun (1998), CFA Institute 연구 - 소형주 내부자 매수 포트폴리오 연 7%+ 초과수익

내부자(CEO, CFO, 이사)가 자기 돈으로 매수하면서 동시에 공매도 잔고가 줄어드는 종목은 가장 강력한 정보 비대칭 신호입니다. 특히 **CFO의 매수**와 **3명 이상 동시 매수(클러스터 매수)**가 가장 예측력이 높습니다.

- **측정**: SEC Form 4 기반 순매수 금액 / 시가총액
- **강화 신호**: Short Interest Ratio 전월 대비 감소 + 내부자 매수 동시 발생
- **필터**: 스톡옵션 행사 후 자동매도 제외, 순수 open-market 매수만 포함

### Layer 3: 📊 Neglected Firm Effect (저커버리지 프리미엄)
> **학술적 근거**: Arbel & Strebel (1983) - 애널리스트 미커버리지 소형주 연 3-5% 초과수익

애널리스트 커버리지가 0~2명인 소형주는 정보 비효율성이 극대화됩니다. PEAD, 내부자 매수 신호 모두 저커버리지 종목에서 더 강하게 작동하며, 이 필터 하나만으로도 모든 레이어의 알파를 증폭시킵니다.

- **측정**: I/B/E/S 기준 애널리스트 수 0~3명
- **효과**: Layer 1의 PEAD 효과 약 40% 증폭, Layer 2의 내부자 신호 예측력 향상

### Layer 4: 🛡️ Quality + Value 안전장치
> **학술적 근거**: Fama-French 5-Factor, Novy-Marx (2013) - Gross Profitability Premium

위 3개 레이어가 공격적 알파 소스라면, Layer 4는 "좀비 기업"과 "가치함정"을 걸러내는 방어막입니다.

- **Quality 필터**: Gross Profit/Assets > 중위값 (Novy-Marx의 수익성 팩터)
- **Value 안전장치**: EV/EBITDA < 섹터 상위 50% (극단적 고평가 제외)
- **재무 건전성**: Altman Z-Score > 1.8 (부도 위험 기업 제거)

---

## 포트폴리오 구성 프로세스

```
Russell 2000 유니버스 (~2,000종목)
    │
    ▼ [Pre-Filter] 시가총액 $200M+, 일평균 거래대금 $500K+
    │  ≈ 1,200종목
    │
    ▼ [Layer 4] Quality + Value 안전장치
    │  GP/Assets > 중위값, EV/EBITDA < 섹터 50%, Z-Score > 1.8
    │  ≈ 400종목
    │
    ▼ [Layer 3] 저커버리지 프리미엄
    │  애널리스트 수 0~3명 우선, 4~6명 허용
    │  ≈ 250종목
    │
    ▼ [Layer 1 + 2] Alpha Signal Scoring
    │  PEAD Score (40%) + Insider Signal (30%) + 12M Momentum (30%)
    │
    ▼ 복합 점수 상위 30~40개 종목
    │  동일가중 포트폴리오
    │
    ▼ [Risk Control]
    │  섹터당 최대 25%, 개별 종목 최대 4%
    │  200일 이동평균 시장 필터 (선택)
    │
    ▼ 월별 리밸런싱 (어닝 시즌에 맞춰)
```

---

## 복합 스코어링 시스템

각 종목에 대해 아래 점수를 0~100으로 정규화한 후 가중 합산:

| 팩터 | 가중치 | 측정 방법 | 데이터 소스 |
|------|--------|-----------|-------------|
| **PEAD Score** | 40% | SUE 백분위 + EAR (3일 비정상수익률) | Compustat, Yahoo Finance |
| **Insider Signal** | 30% | 순매수 금액/시총 + 클러스터 여부 + SI 변화 | SEC Form 4, FINRA |
| **Price Momentum** | 30% | 12개월 수익률 (최근 1개월 제외) | Yahoo Finance |

**가점 사항** (각 +10점):
- 애널리스트 수 0명 (완전 미커버리지)
- CFO 또는 CEO 직접 매수
- 3명 이상 동시 매수 (클러스터)
- 직전 2분기 연속 어닝 서프라이즈

---

## 리밸런싱 일정

어닝 시즌에 맞춰 **월 1회** 리밸런싱:
- 1월 말 / 4월 말 / 7월 말 / 10월 말: **대규모 리밸런싱** (어닝 시즌 종료 후)
- 나머지 월: **부분 조정** (신규 어닝 서프라이즈, 내부자 매수 반영)

---

## 기존 전략 대비 기대 우위

| 차원 | 기존 Value+Momentum | **Alpha Stack** |
|------|---------------------|-----------------|
| 알파 소스 | 2개 (밸류, 모멘텀) | 4개 레이어 중첩 |
| 정보 유형 | 가격/재무 only | 가격 + 재무 + 내부자 + 커버리지 |
| 소형주 특화 | 일반적 팩터 적용 | 소형주 비효율성 극대화 설계 |
| PEAD 활용 | 없음 | 핵심 알파 드라이버 |
| 리밸런싱 | 분기 | 월별 (어닝 시즌 동기화) |
| 기대 초과수익 | +3~5% vs R2000 | **+7~12% vs R2000** (백테스트 기준) |

---

## 리스크 요인 및 약점

1. **유동성 리스크**: 저커버리지 소형주는 매도 시 슬리피지 클 수 있음 → $200M 시가총액 하한으로 완화
2. **PEAD 감소 추세**: 대형주에서는 사라졌지만 소형주(특히 마이크로캡 제외 시)에서는 여전히 유효
3. **거래비용**: 월별 리밸런싱으로 회전율 높음 → 편도 0.3~0.5% 가정 필요
4. **데이터 비용**: SEC Form 4, Short Interest 데이터 확보 필요
5. **생존자 편향**: 백테스트 시 상장폐지 종목 반드시 포함

---

## 무료 데이터 소스

| 데이터 | 소스 | 비용 |
|--------|------|------|
| 주가/재무제표 | Yahoo Finance (yfinance) | 무료 |
| 내부자 거래 | SEC EDGAR Form 4, OpenInsider.com | 무료 |
| Short Interest | FINRA (월 2회), Quandl | 무료~$30/월 |
| 애널리스트 수 | Yahoo Finance, Finviz | 무료 |
| 어닝 서프라이즈 | Yahoo Finance earnings calendar | 무료 |
| Russell 2000 목록 | iShares IWM Holdings CSV | 무료 |

---

## 구현 우선순위

### Phase 1 (즉시 구현 가능)
- Layer 4 (Quality + Value 필터) + 12M Momentum → 기존 전략 수준
- 데이터: yfinance만으로 가능

### Phase 2 (1~2주 소요)
- Layer 1 (PEAD) 추가: yfinance earnings calendar + surprise 데이터
- 복합 스코어링 구현

### Phase 3 (1개월 소요)
- Layer 2 (Insider + SI): SEC EDGAR API 또는 OpenInsider 스크래핑
- Layer 3 (커버리지 필터): Yahoo Finance analyst count
- 풀 전략 백테스트

---

## 참고 문헌

1. Bernard, V. & Thomas, J. (1990). "Post-Earnings-Announcement Drift" - *Journal of Accounting Research*
2. Seyhun, H.N. (1998). "Investment Intelligence from Insider Trading" - MIT Press
3. Novy-Marx, R. (2013). "The Other Side of Value: The Gross Profitability Premium" - *JFE*
4. Arbel, A. & Strebel, P. (1983). "Pay Attention to Neglected Firms" - *JPM*
5. Fama, E. & French, K. (2015). "A Five-Factor Asset Pricing Model" - *JFE*
6. Battalio, R. & Mendenhall, R. (2007). "Post-Earnings Announcement Drift" - *FAJ*
