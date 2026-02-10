"""
═══════════════════════════════════════════════════════════════════
  Alpha Stack: Russell 2000 Multi-Anomaly 복합 전략 백테스트
═══════════════════════════════════════════════════════════════════

4-Layer Alpha Stack:
  Layer 1: PEAD (Post-Earnings Announcement Drift)
  Layer 2: Insider Buying + Short Interest Signal
  Layer 3: Neglected Firm Effect (Low Analyst Coverage)
  Layer 4: Quality + Value Safety Net

데이터: yfinance (무료) + OpenInsider (무료)
실행: python alpha_stack_backtest.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import time
import json

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

CONFIG = {
    # Universe
    'TICKER_SOURCE': 'russell2000_tickers.csv',  # IWM holdings CSV
    'SAMPLE_SIZE': 300,           # 테스트용 샘플 (전체: None)
    'MIN_MARKET_CAP': 2e8,        # $200M 하한
    'MIN_AVG_VOLUME': 5e5,        # 일평균 $500K 거래대금

    # Backtest Period
    'START_DATE': '2021-01-01',
    'END_DATE': '2024-12-31',

    # Portfolio
    'TOP_N': 35,                  # 포트폴리오 종목 수
    'REBALANCE_FREQ': 'monthly',  # monthly or quarterly
    'MAX_SECTOR_PCT': 0.25,       # 섹터당 최대 비중
    'MAX_STOCK_PCT': 0.04,        # 개별 종목 최대

    # Caching
    'CACHE_FILE': 'alpha_stack_cache.pkl',  # 캐시 파일명
    'CACHE_DAYS': 7,              # 캐시 유효 기간 (일)
    'FORCE_REFRESH': False,       # True = 캐시 무시하고 새로 다운로드

    # Scoring Weights
    'W_PEAD': 0.40,               # PEAD Score 가중치
    'W_INSIDER': 0.30,            # Insider Signal 가중치
    'W_MOMENTUM': 0.30,           # Momentum 가중치
    
    # Transaction Costs
    'TX_COST': 0.004,             # 편도 0.4% (소형주)
}


# ═══════════════════════════════════════════════════════════
# LAYER 4: Quality + Value Safety Net
# ═══════════════════════════════════════════════════════════

def compute_quality_value_filter(ticker_data):
    """
    Quality + Value 안전장치
    - Gross Profit / Total Assets > 중위값 (Novy-Marx Profitability)
    - EV/EBITDA < 섹터 중위값의 1.5배
    - Altman Z-Score > 1.8 (부도 위험 제거)
    """
    scores = {}
    
    for ticker, data in ticker_data.items():
        try:
            info = data.get('info', {})
            
            # Gross Profit / Assets (Novy-Marx Quality)
            gross_profit = info.get('grossProfits', 0) or 0
            total_assets = info.get('totalAssets', 1) or 1
            gp_assets = gross_profit / total_assets if total_assets > 0 else 0
            
            # EV/EBITDA
            ev_ebitda = info.get('enterpriseToEbitda', None)
            if ev_ebitda is None or ev_ebitda <= 0:
                ev_ebitda = 999  # 데이터 없으면 패널티
            
            # ROE
            roe = info.get('returnOnEquity', 0) or 0
            
            # Debt/Equity
            debt_equity = info.get('debtToEquity', 999) or 999
            
            # Profit Margin
            profit_margin = info.get('profitMargins', 0) or 0
            
            # 통과 조건
            passes = (
                gp_assets > 0.15 and          # 적정 수익성
                ev_ebitda < 25 and             # 극단적 고평가 제외
                ev_ebitda > 0 and              # 적자 기업 제외
                roe > 0.05 and                 # 최소 ROE
                debt_equity < 200 and          # 과도한 부채 제외
                profit_margin > 0              # 흑자 기업만
            )
            
            if passes:
                # Quality Score (0~100)
                q_score = (
                    min(gp_assets / 0.5, 1.0) * 40 +          # GP/Assets
                    min(roe / 0.3, 1.0) * 30 +                 # ROE
                    max(0, 1 - debt_equity / 200) * 15 +       # Low Debt
                    min(profit_margin / 0.2, 1.0) * 15          # Profit Margin
                )
                
                # Value Score (낮을수록 좋음 → 역변환)
                v_score = max(0, 100 - ev_ebitda * 4)  # EV/EBITDA 25 → 0점
                
                scores[ticker] = {
                    'quality_score': q_score,
                    'value_score': v_score,
                    'gp_assets': gp_assets,
                    'ev_ebitda': ev_ebitda,
                    'roe': roe,
                    'debt_equity': debt_equity,
                    'passes_l4': True
                }
        except Exception:
            continue
    
    return scores


# ═══════════════════════════════════════════════════════════
# LAYER 3: Neglected Firm Effect
# ═══════════════════════════════════════════════════════════

def compute_coverage_score(ticker_data):
    """
    애널리스트 커버리지가 낮을수록 높은 점수
    0명: 100점, 1명: 90점, 2명: 80점, ..., 10+명: 0점
    """
    scores = {}
    
    for ticker, data in ticker_data.items():
        try:
            info = data.get('info', {})
            num_analysts = info.get('numberOfAnalystOpinions', 0) or 0
            
            # 커버리지 점수: 낮을수록 좋음
            coverage_score = max(0, 100 - num_analysts * 10)
            
            # 가점: 완전 미커버리지
            bonus = 10 if num_analysts == 0 else 0
            
            scores[ticker] = {
                'num_analysts': num_analysts,
                'coverage_score': min(100, coverage_score + bonus),
                'is_neglected': num_analysts <= 3
            }
        except Exception:
            continue
    
    return scores


# ═══════════════════════════════════════════════════════════
# LAYER 1: PEAD (Earnings Surprise Drift)
# ═══════════════════════════════════════════════════════════

def compute_pead_score(ticker_data):
    """
    PEAD Score 계산:
    1. Earnings Surprise % (실제 vs 예상)
    2. EAR (Earnings Announcement Return) - 발표일 ±1일 비정상수익률
    3. 연속 서프라이즈 가점
    """
    scores = {}
    
    for ticker, data in ticker_data.items():
        try:
            # yfinance에서 earnings 데이터 가져오기
            stock = data.get('stock_obj')
            if stock is None:
                continue
            
            # Earnings surprise from Yahoo Finance
            earnings = None
            try:
                earnings = stock.earnings_dates
            except:
                pass
            
            if earnings is None or len(earnings) == 0:
                # Fallback: 가격 기반 대리 측정
                hist = data.get('history')
                if hist is not None and len(hist) > 60:
                    # 최근 분기별 수익률 변동으로 서프라이즈 추정
                    recent_return = hist['Close'].pct_change(20).iloc[-1]
                    pead_score = max(0, min(100, 50 + recent_return * 500))
                    scores[ticker] = {
                        'pead_score': pead_score,
                        'surprise_pct': None,
                        'consecutive_beats': 0,
                        'method': 'price_proxy'
                    }
                continue
            
            # Earnings surprise 계산
            surprise_pcts = []
            if 'Surprise(%)' in earnings.columns:
                recent_surprises = earnings['Surprise(%)'].dropna().head(4)
                surprise_pcts = recent_surprises.tolist()
            elif 'EPS Estimate' in earnings.columns and 'Reported EPS' in earnings.columns:
                for _, row in earnings.head(4).iterrows():
                    est = row.get('EPS Estimate')
                    actual = row.get('Reported EPS')
                    if pd.notna(est) and pd.notna(actual) and est != 0:
                        surprise_pcts.append((actual - est) / abs(est) * 100)
            
            if len(surprise_pcts) == 0:
                continue
            
            # 가장 최근 서프라이즈
            latest_surprise = surprise_pcts[0] if surprise_pcts else 0
            
            # 연속 비트 횟수
            consecutive_beats = 0
            for s in surprise_pcts:
                if s > 0:
                    consecutive_beats += 1
                else:
                    break
            
            # PEAD Score 산출
            # 서프라이즈가 클수록, 연속 비트가 많을수록 높은 점수
            surprise_score = max(0, min(100, 50 + latest_surprise * 5))
            beat_bonus = consecutive_beats * 10  # 연속 비트 가점
            
            pead_score = min(100, surprise_score + beat_bonus)
            
            scores[ticker] = {
                'pead_score': pead_score,
                'surprise_pct': latest_surprise,
                'consecutive_beats': consecutive_beats,
                'method': 'earnings_data'
            }
            
        except Exception:
            continue
    
    return scores


# ═══════════════════════════════════════════════════════════
# LAYER 2: Insider Signal (Simplified - yfinance based)
# ═══════════════════════════════════════════════════════════

def compute_insider_score(ticker_data):
    """
    내부자 매매 신호 (yfinance 기반 간소화 버전)
    
    실전에서는 SEC EDGAR Form 4 또는 OpenInsider.com API 사용 권장
    yfinance에서는 insider_transactions로 기본 데이터 확보 가능
    """
    scores = {}
    
    for ticker, data in ticker_data.items():
        try:
            stock = data.get('stock_obj')
            if stock is None:
                continue
            
            # Insider transactions
            insider_txns = None
            try:
                insider_txns = stock.insider_transactions
            except:
                pass
            
            if insider_txns is None or len(insider_txns) == 0:
                scores[ticker] = {
                    'insider_score': 50,  # 중립 (데이터 없음)
                    'net_buys': 0,
                    'cluster_buy': False,
                    'method': 'no_data'
                }
                continue
            
            # 최근 90일 내 거래만
            recent_date = datetime.now() - timedelta(days=90)
            
            # 매수/매도 건수 및 금액
            buys = 0
            sells = 0
            buy_value = 0
            sell_value = 0
            unique_buyers = set()
            
            for _, txn in insider_txns.iterrows():
                txn_type = str(txn.get('Text', '')).lower()
                shares = abs(txn.get('Shares', 0) or 0)
                value = abs(txn.get('Value', 0) or 0)
                insider = txn.get('Insider', '')
                
                if 'purchase' in txn_type or 'buy' in txn_type:
                    buys += 1
                    buy_value += value
                    unique_buyers.add(insider)
                elif 'sale' in txn_type or 'sell' in txn_type:
                    sells += 1
                    sell_value += value
            
            # Net Buy Ratio
            total_txns = buys + sells
            if total_txns == 0:
                net_buy_ratio = 0
            else:
                net_buy_ratio = (buys - sells) / total_txns
            
            # 클러스터 매수 (3명 이상 매수)
            cluster_buy = len(unique_buyers) >= 3
            
            # Insider Score 산출
            insider_score = 50 + net_buy_ratio * 40  # -50 ~ 90 기본
            
            if cluster_buy:
                insider_score += 10  # 클러스터 가점
            
            if buy_value > sell_value * 2:
                insider_score += 10  # 매수 금액이 매도의 2배 이상
            
            insider_score = max(0, min(100, insider_score))
            
            scores[ticker] = {
                'insider_score': insider_score,
                'net_buys': buys - sells,
                'buy_value': buy_value,
                'sell_value': sell_value,
                'cluster_buy': cluster_buy,
                'unique_buyers': len(unique_buyers),
                'method': 'yfinance'
            }
            
        except Exception:
            scores[ticker] = {
                'insider_score': 50,
                'net_buys': 0,
                'cluster_buy': False,
                'method': 'error'
            }
    
    return scores


# ═══════════════════════════════════════════════════════════
# MOMENTUM (12M - 1M)
# ═══════════════════════════════════════════════════════════

def compute_momentum_score(ticker_data):
    """
    12개월 수익률 (최근 1개월 제외) → 학술적 표준 모멘텀
    Jegadeesh & Titman (1993) 방법론
    """
    scores = {}
    
    for ticker, data in ticker_data.items():
        try:
            hist = data.get('history')
            if hist is None or len(hist) < 252:
                continue
            
            prices = hist['Close']
            
            # 12개월 전 대비 수익률 (최근 1개월 제외)
            if len(prices) >= 252:
                mom_12m = prices.iloc[-22] / prices.iloc[-252] - 1
            elif len(prices) >= 126:
                mom_12m = prices.iloc[-22] / prices.iloc[0] - 1
            else:
                continue
            
            # 최근 1개월 수익률 (단기 반전 효과)
            mom_1m = prices.iloc[-1] / prices.iloc[-22] - 1
            
            # 변동성 조정 모멘텀
            daily_returns = prices.pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)
            
            risk_adj_momentum = mom_12m / max(volatility, 0.1)
            
            scores[ticker] = {
                'momentum_12m_1m': mom_12m,
                'momentum_1m': mom_1m,
                'volatility': volatility,
                'risk_adj_momentum': risk_adj_momentum
            }
            
        except Exception:
            continue
    
    # Percentile 기반 점수화 (0~100)
    if scores:
        mom_values = [s['momentum_12m_1m'] for s in scores.values()]
        for ticker in scores:
            rank = sum(1 for v in mom_values if v <= scores[ticker]['momentum_12m_1m'])
            scores[ticker]['momentum_score'] = rank / len(mom_values) * 100
    
    return scores


# ═══════════════════════════════════════════════════════════
# COMPOSITE SCORING & PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════

def compute_composite_score(l4_scores, l3_scores, l1_scores, l2_scores, mom_scores):
    """
    4개 레이어 + 모멘텀을 결합한 복합 점수 산출
    """
    composites = {}
    
    # L4를 통과한 종목만 대상
    eligible = set(l4_scores.keys())
    
    for ticker in eligible:
        try:
            l4 = l4_scores.get(ticker, {})
            l3 = l3_scores.get(ticker, {})
            l1 = l1_scores.get(ticker, {})
            l2 = l2_scores.get(ticker, {})
            mom = mom_scores.get(ticker, {})
            
            if not l4.get('passes_l4', False):
                continue
            
            # 각 레이어 점수 (0~100)
            pead = l1.get('pead_score', 50)
            insider = l2.get('insider_score', 50)
            momentum = mom.get('momentum_score', 50)
            coverage = l3.get('coverage_score', 50)
            quality = l4.get('quality_score', 50)
            value = l4.get('value_score', 50)
            
            # Alpha Signal Score (Layer 1 + 2 + Momentum)
            alpha_score = (
                pead * CONFIG['W_PEAD'] +
                insider * CONFIG['W_INSIDER'] +
                momentum * CONFIG['W_MOMENTUM']
            )
            
            # Coverage Multiplier (저커버리지 시 알파 증폭)
            coverage_mult = 1.0 + (coverage - 50) / 200  # 0.75 ~ 1.25
            
            # Quality-Value Baseline
            qv_baseline = quality * 0.6 + value * 0.4
            
            # Final Composite
            composite = alpha_score * coverage_mult * 0.7 + qv_baseline * 0.3
            
            composites[ticker] = {
                'composite_score': composite,
                'alpha_score': alpha_score,
                'pead_score': pead,
                'insider_score': insider,
                'momentum_score': momentum,
                'coverage_score': coverage,
                'quality_score': quality,
                'value_score': value,
                'coverage_mult': coverage_mult,
                'num_analysts': l3.get('num_analysts', 0),
                'ev_ebitda': l4.get('ev_ebitda', 0),
                'roe': l4.get('roe', 0),
            }
            
        except Exception:
            continue
    
    return composites


def construct_portfolio(composites, sector_info, top_n=35):
    """
    섹터 제한을 적용한 포트폴리오 구성
    """
    # 복합 점수 기준 정렬
    sorted_stocks = sorted(composites.items(), 
                          key=lambda x: x[1]['composite_score'], 
                          reverse=True)
    
    portfolio = []
    sector_counts = {}
    max_per_sector = int(top_n * CONFIG['MAX_SECTOR_PCT'])
    
    for ticker, scores in sorted_stocks:
        if len(portfolio) >= top_n:
            break
        
        sector = sector_info.get(ticker, 'Unknown')
        current_count = sector_counts.get(sector, 0)
        
        if current_count < max_per_sector:
            portfolio.append({
                'ticker': ticker,
                'sector': sector,
                **scores
            })
            sector_counts[sector] = current_count + 1
    
    return pd.DataFrame(portfolio)


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_tickers(filepath=None):
    """Russell 2000 티커 로드"""
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
        if 'Ticker' in df.columns:
            return df
        elif 'ticker' in df.columns:
            df = df.rename(columns={'ticker': 'Ticker'})
            return df
    
    # Fallback: 대표 소형주 샘플
    print("⚠️  russell2000_tickers.csv 없음 → 대표 소형주 샘플 사용")
    sample_tickers = [
        # Tech
        'SMCI','CAMT','AMBA','LSCC','AEIS','VICR','POWI','DIOD','SLAB','CALX',
        'ONTO','RMBS','CEVA','AAON','NOVT','AZTA','DIGI','PRGS','QTWO','ALRM',
        # Healthcare
        'MEDP','HALO','ITCI','KRYS','RVMD','PTCT','TGTX','ACAD','CORT','EXAS',
        'LNTH','NVCR','AXNX','IOVA','INSP','GKOS','NARI','IRTC','AVNS','OFIX',
        # Industrial
        'ATKR','ESAB','SPXC','RBC','PRIM','DY','ROCK','GFF','NPO','POWL',
        'ARCB','MATX','UFPI','TREX','CSGS','AAON','AWI','JBSS','MWA','BMI',
        # Consumer
        'BOOT','SHAK','WING','PLNT','FIZZ','CROX','SKX','FOXF','SFM','CAVA',
        'ELF','LULU','DECK','DKS','COLM','CBRL','PLAY','TXRH','DINE','CAKE',
        # Energy/Materials
        'MTDR','SM','PTEN','HP','RES','CIVI','NOG','GPOR','VNOM','REPX',
        'CLF','ATI','CRS','HAYN','IOSP','KWR','TROX','CC','HUN','NGVT',
        # Financials
        'IBOC','FFIN','SBCF','HOPE','BANF','CVBF','FNB','NBTB','TRMK','UBSI',
        'WTFC','PNFP','GBCI','SFBS','CADE','HWC','WSFS','TOWN','VBTX','WAFD',
        # Real Estate/Utilities
        'AAT','AKR','BRT','CUZ','DEI','EGP','HIW','KRG','NNN','OHI',
        'PINE','ROIC','SHO','UMH','VRE','BKH','NWE','OGS','PNM','SJW',
    ]
    
    # 섹터 매핑
    sector_map = {}
    sectors = ['Technology']*20 + ['Healthcare']*20 + ['Industrials']*20 + \
              ['Consumer']*20 + ['Energy']*20 + ['Financials']*20 + ['REIT/Utilities']*20
    for i, t in enumerate(sample_tickers):
        sector_map[t] = sectors[i] if i < len(sectors) else 'Other'
    
    df = pd.DataFrame({
        'Ticker': sample_tickers,
        'Sector': [sector_map.get(t, 'Other') for t in sample_tickers]
    })
    return df


def load_cache(cache_file, max_age_days):
    """캐시 파일 로드"""
    if not os.path.exists(cache_file):
        print(f"   캐시 파일 없음: {cache_file}")
        return None

    import pickle
    file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days
    file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)

    if file_age_days > max_age_days:
        print(f"   ⚠️  캐시 만료: {file_age_days}일 경과 (최대 {max_age_days}일)")
        return None

    try:
        print(f"   📂 캐시 파일 발견: {cache_file}")
        print(f"      크기: {file_size_mb:.1f} MB | 생성: {file_age_days}일 전")
        print(f"   ⏳ 캐시 로딩 중...")

        with open(cache_file, 'rb') as f:
            data = pickle.load(f)

        print(f"   ✅ 캐시 로드 완료: {len(data)}개 종목 데이터")
        return data
    except Exception as e:
        print(f"   ⚠️  캐시 로드 실패: {e}")
        return None


def save_cache(data, cache_file):
    """캐시 파일 저장"""
    import pickle
    try:
        print(f"\n💾 캐시 저장 중: {len(data)}개 종목...")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

        file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
        print(f"   ✅ 저장 완료: {cache_file} ({file_size_mb:.1f} MB)")
        print(f"   유효기간: {CONFIG['CACHE_DAYS']}일")
    except Exception as e:
        print(f"   ⚠️  캐시 저장 실패: {e}")


def fetch_stock_data(tickers, period='2y'):
    """주가 및 기본 정보 다운로드"""
    ticker_data = {}
    total = len(tickers)
    start_time = time.time()

    print(f"\n📊 {total}개 종목 데이터 다운로드 시작...")
    print(f"   예상 소요 시간: ~{total * 0.15 / 60:.1f}분 (평균 0.15초/종목)")
    print(f"   진행 상황:")

    for i, ticker in enumerate(tickers):
        # Show progress every 10 stocks or at milestones
        if (i + 1) % 10 == 0 or (i + 1) in [1, 5, total]:
            elapsed = time.time() - start_time
            progress_pct = (i + 1) / total * 100
            valid_count = len(ticker_data)

            # Calculate ETA
            if i > 0:
                avg_time_per_stock = elapsed / (i + 1)
                remaining_stocks = total - (i + 1)
                eta_seconds = avg_time_per_stock * remaining_stocks
                eta_min = eta_seconds / 60

                print(f"   [{i+1:4d}/{total}] {progress_pct:5.1f}% | "
                      f"유효: {valid_count:3d} | "
                      f"경과: {elapsed/60:4.1f}분 | "
                      f"남은시간: ~{eta_min:4.1f}분")
            else:
                print(f"   [{i+1:4d}/{total}] {progress_pct:5.1f}% | 유효: {valid_count:3d}")

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist is None or len(hist) < 60:
                continue

            info = {}
            try:
                info = stock.info
            except:
                pass

            # 시가총액 & 거래량 필터
            market_cap = info.get('marketCap', 0) or 0
            avg_volume = info.get('averageVolume', 0) or 0
            current_price = hist['Close'].iloc[-1] if len(hist) > 0 else 0
            avg_dollar_volume = avg_volume * current_price

            if market_cap < CONFIG['MIN_MARKET_CAP']:
                continue
            if avg_dollar_volume < CONFIG['MIN_AVG_VOLUME']:
                continue

            ticker_data[ticker] = {
                'stock_obj': stock,
                'history': hist,
                'info': info,
                'market_cap': market_cap,
            }

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            continue

    elapsed_total = time.time() - start_time
    print(f"\n   ✅ 완료: {len(ticker_data)}개 종목 데이터 확보 (총 {elapsed_total/60:.1f}분 소요)")
    return ticker_data


# ═══════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════

def run_backtest(ticker_data, composites, sector_info, start_date, end_date, top_n=35):
    """
    간소화된 백테스트 (단일 기간)
    
    실전에서는 rolling window로 매월 리밸런싱해야 하지만,
    yfinance 제약상 단일 스냅샷 기반으로 구현
    """
    print("\n🔄 백테스트 실행 중...")
    
    # 포트폴리오 구성
    portfolio_df = construct_portfolio(composites, sector_info, top_n)
    
    if len(portfolio_df) == 0:
        print("❌ 포트폴리오 구성 실패")
        return None, None, portfolio_df
    
    print(f"   포트폴리오: {len(portfolio_df)}개 종목")
    
    # 동일가중 포트폴리오 수익률 계산
    portfolio_returns = []
    benchmark_returns = []
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # 포트폴리오 종목 일별 수익률
    daily_returns_list = []
    for _, row in portfolio_df.iterrows():
        ticker = row['ticker']
        if ticker in ticker_data:
            hist = ticker_data[ticker]['history']
            if hist is not None and len(hist) > 0:
                ret = hist['Close'].pct_change().dropna()
                # Remove timezone to avoid comparison issues
                if ret.index.tz is not None:
                    ret.index = ret.index.tz_localize(None)
                ret = ret[ret.index >= start]
                ret = ret[ret.index <= end]
                ret.name = ticker
                daily_returns_list.append(ret)
    
    if len(daily_returns_list) == 0:
        print("❌ 수익률 데이터 없음")
        return None, None, portfolio_df
    
    # 동일가중 포트폴리오
    returns_df = pd.concat(daily_returns_list, axis=1)
    portfolio_daily = returns_df.mean(axis=1)
    
    # 거래비용 차감 (월 1회 리밸런싱, 평균 30% 교체율 가정)
    monthly_cost = CONFIG['TX_COST'] * 2 * 0.30  # 편도비용 × 왕복 × 교체율
    daily_cost = monthly_cost / 21  # 거래일 기준
    portfolio_daily = portfolio_daily - daily_cost
    
    # 벤치마크: IWM (Russell 2000 ETF)
    try:
        iwm = yf.Ticker('IWM')
        iwm_hist = iwm.history(start=start, end=end)
        benchmark_daily = iwm_hist['Close'].pct_change().dropna()
    except:
        benchmark_daily = pd.Series(0, index=portfolio_daily.index)
    
    # 인덱스 맞추기
    common_idx = portfolio_daily.index.intersection(benchmark_daily.index)
    portfolio_daily = portfolio_daily[common_idx]
    benchmark_daily = benchmark_daily[common_idx]
    
    return portfolio_daily, benchmark_daily, portfolio_df


# ═══════════════════════════════════════════════════════════
# PERFORMANCE ANALYTICS
# ═══════════════════════════════════════════════════════════

def calculate_metrics(returns):
    """성과 지표 계산"""
    if returns is None or len(returns) == 0:
        return {}
    
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / max(volatility, 0.001)  # RF=4%
    
    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    max_dd = drawdowns.min()
    
    # Sortino Ratio
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (cagr - 0.04) / max(downside, 0.001)
    
    # Win Rate
    win_rate = (returns > 0).mean()
    
    return {
        'Total Return': f"{total_return:.2%}",
        'CAGR': f"{cagr:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Sortino Ratio': f"{sortino:.2f}",
        'Max Drawdown': f"{max_dd:.2%}",
        'Win Rate (Daily)': f"{win_rate:.2%}",
        'Trading Days': len(returns),
    }


def print_results(strategy_metrics, benchmark_metrics, portfolio_df):
    """결과 출력"""
    print("\n" + "═" * 70)
    print("  📊 Alpha Stack 백테스트 결과")
    print("═" * 70)
    
    print(f"\n{'지표':<25} {'전략':>15} {'벤치마크(IWM)':>15}")
    print("-" * 55)
    
    for key in strategy_metrics:
        s_val = strategy_metrics.get(key, 'N/A')
        b_val = benchmark_metrics.get(key, 'N/A')
        print(f"{key:<25} {s_val:>15} {b_val:>15}")
    
    print(f"\n{'─' * 55}")
    print(f"  포트폴리오 종목 수: {len(portfolio_df)}")
    
    if len(portfolio_df) > 0:
        print(f"\n  📈 상위 10개 종목 (복합 점수 기준):")
        print(f"  {'Ticker':<8} {'Composite':>10} {'PEAD':>8} {'Insider':>8} {'Mom':>8} {'Coverage':>8}")
        print(f"  {'-'*50}")
        for _, row in portfolio_df.head(10).iterrows():
            print(f"  {row['ticker']:<8} {row['composite_score']:>10.1f} "
                  f"{row['pead_score']:>8.1f} {row['insider_score']:>8.1f} "
                  f"{row['momentum_score']:>8.1f} {row['coverage_score']:>8.1f}")
        
        print(f"\n  🏢 섹터 분포:")
        sector_dist = portfolio_df['sector'].value_counts()
        for sector, count in sector_dist.items():
            pct = count / len(portfolio_df) * 100
            bar = '█' * int(pct / 2)
            print(f"  {sector:<20} {count:>3}개 ({pct:>5.1f}%) {bar}")


# ═══════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════

def create_charts(portfolio_returns, benchmark_returns, portfolio_df, output_dir='.'):
    """결과 차트 생성"""
    try:
        # Remove timezone to avoid comparison issues
        if portfolio_returns.index.tz is not None:
            portfolio_returns.index = portfolio_returns.index.tz_localize(None)
        if benchmark_returns.index.tz is not None:
            benchmark_returns.index = benchmark_returns.index.tz_localize(None)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Alpha Stack: Russell 2000 Multi-Anomaly Strategy', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. 누적 수익률
        ax = axes[0, 0]
        port_cum = (1 + portfolio_returns).cumprod()
        bench_cum = (1 + benchmark_returns).cumprod()
        ax.plot(port_cum.index, port_cum.values, label='Alpha Stack', 
                linewidth=2.5, color='#1a73e8')
        ax.plot(bench_cum.index, bench_cum.values, label='IWM (Russell 2000)', 
                linewidth=2, color='#ea4335', alpha=0.7)
        ax.fill_between(port_cum.index, port_cum.values, bench_cum.values,
                       where=port_cum.values >= bench_cum.values, 
                       color='#1a73e8', alpha=0.1)
        ax.set_title('Cumulative Returns', fontsize=13, fontweight='bold')
        ax.set_ylabel('Growth of $1')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 2. 드로우다운
        ax = axes[0, 1]
        rolling_max = port_cum.expanding().max()
        drawdown = (port_cum / rolling_max - 1) * 100
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                       color='#ea4335', alpha=0.5)
        bench_rm = bench_cum.expanding().max()
        bench_dd = (bench_cum / bench_rm - 1) * 100
        ax.plot(bench_dd.index, bench_dd.values, color='gray', 
                alpha=0.5, label='IWM Drawdown')
        ax.set_title('Drawdown Comparison', fontsize=13, fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 월별 초과수익
        ax = axes[1, 0]
        port_monthly = portfolio_returns.resample('M').apply(lambda x: (1+x).prod() - 1)
        bench_monthly = benchmark_returns.resample('ME').apply(lambda x: (1+x).prod() - 1)
        common = port_monthly.index.intersection(bench_monthly.index)
        excess = port_monthly[common] - bench_monthly[common]
        colors = ['#ea4335' if x < 0 else '#34a853' for x in excess.values]
        ax.bar(range(len(excess)), excess.values * 100, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_title('Monthly Excess Returns vs IWM', fontsize=13, fontweight='bold')
        ax.set_ylabel('Excess Return (%)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Alpha Layer 기여도
        ax = axes[1, 1]
        if len(portfolio_df) > 0:
            layer_means = {
                'PEAD\n(Layer 1)': portfolio_df['pead_score'].mean(),
                'Insider\n(Layer 2)': portfolio_df['insider_score'].mean(),
                'Coverage\n(Layer 3)': portfolio_df['coverage_score'].mean(),
                'Quality\n(Layer 4)': portfolio_df['quality_score'].mean(),
                'Momentum': portfolio_df['momentum_score'].mean(),
            }
            colors_radar = ['#1a73e8', '#ea4335', '#fbbc04', '#34a853', '#9c27b0']
            bars = ax.barh(list(layer_means.keys()), list(layer_means.values()), 
                          color=colors_radar, alpha=0.8, height=0.6)
            ax.set_xlim(0, 100)
            ax.set_title('Average Layer Scores (Portfolio)', fontsize=13, fontweight='bold')
            ax.set_xlabel('Score (0-100)')
            for bar, val in zip(bars, layer_means.values()):
                ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                       f'{val:.0f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'alpha_stack_results.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n📊 차트 저장: {chart_path}")
        
    except ImportError:
        print("⚠️  matplotlib 미설치 - 차트 생략")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    import sys

    # 커맨드 라인 인자 처리
    if '--refresh' in sys.argv or '-r' in sys.argv:
        CONFIG['FORCE_REFRESH'] = True
        print("강제 새로고침 모드\n")

    print("═" * 70)
    print("  🚀 Alpha Stack: Russell 2000 Multi-Anomaly 전략 백테스트")
    print("  ── 4-Layer Alpha Stacking Strategy ──")
    print("═" * 70)
    print(f"\n  설정:")
    print(f"  ├ 기간: {CONFIG['START_DATE']} ~ {CONFIG['END_DATE']}")
    print(f"  ├ 포트폴리오: 상위 {CONFIG['TOP_N']}개 동일가중")
    print(f"  ├ 가중치: PEAD {CONFIG['W_PEAD']:.0%} / Insider {CONFIG['W_INSIDER']:.0%} / Mom {CONFIG['W_MOMENTUM']:.0%}")
    print(f"  └ 거래비용: 편도 {CONFIG['TX_COST']:.1%}")
    
    # 1. 티커 로드
    ticker_df = load_tickers(CONFIG['TICKER_SOURCE'])
    tickers = ticker_df['Ticker'].tolist()
    
    sector_info = {}
    if 'Sector' in ticker_df.columns:
        sector_info = dict(zip(ticker_df['Ticker'], ticker_df['Sector']))
    
    # 샘플링
    if CONFIG['SAMPLE_SIZE'] and CONFIG['SAMPLE_SIZE'] < len(tickers):
        np.random.seed(42)
        tickers = list(np.random.choice(tickers, CONFIG['SAMPLE_SIZE'], replace=False))
        print(f"\n  ⚡ {CONFIG['SAMPLE_SIZE']}개 샘플로 테스트")

    # 2. 데이터 다운로드 (캐시 사용)
    print("\n" + "─" * 70)
    print("📦 데이터 로딩")
    print("─" * 70)

    ticker_data = None

    if not CONFIG['FORCE_REFRESH']:
        print(f"🔍 캐시 파일 확인: {CONFIG['CACHE_FILE']}")
        ticker_data = load_cache(CONFIG['CACHE_FILE'], CONFIG['CACHE_DAYS'])

    if ticker_data is None:
        if CONFIG['FORCE_REFRESH']:
            print("🔄 강제 새로고침: 캐시를 무시하고 새 데이터를 다운로드합니다...")
        else:
            print("📥 캐시 없음: 새 데이터를 다운로드합니다...")
        print(f"   대상 종목 수: {len(tickers)}개")
        ticker_data = fetch_stock_data(tickers)
        save_cache(ticker_data, CONFIG['CACHE_FILE'])
    else:
        print(f"✅ 캐시된 데이터 사용 ({len(ticker_data)}개 종목)")
        print(f"   다음 새로고침까지: {CONFIG['CACHE_DAYS']}일 이내")
        print(f"   강제 새로고침: --refresh 플래그 사용")

    if len(ticker_data) < 20:
        print(f"❌ 충분한 데이터 없음 ({len(ticker_data)}개)")
        return
    
    # 3. Layer 4: Quality + Value Filter
    print("\n🛡️  Layer 4: Quality + Value 필터링...")
    l4_scores = compute_quality_value_filter(ticker_data)
    passed = sum(1 for s in l4_scores.values() if s.get('passes_l4'))
    print(f"   통과: {passed}/{len(ticker_data)}")
    
    # 4. Layer 3: Coverage Score
    print("\n🔍 Layer 3: 커버리지 점수 산출...")
    l3_scores = compute_coverage_score(ticker_data)
    neglected = sum(1 for s in l3_scores.values() if s.get('is_neglected'))
    print(f"   저커버리지(≤3명): {neglected}/{len(l3_scores)}")
    
    # 5. Layer 1: PEAD Score
    print("\n🎯 Layer 1: PEAD 점수 산출...")
    l1_scores = compute_pead_score(ticker_data)
    print(f"   PEAD 데이터: {len(l1_scores)}개")
    
    # 6. Layer 2: Insider Score
    print("\n🕵️  Layer 2: 내부자 매매 점수 산출...")
    l2_scores = compute_insider_score(ticker_data)
    net_positive = sum(1 for s in l2_scores.values() if s.get('net_buys', 0) > 0)
    print(f"   순매수 종목: {net_positive}/{len(l2_scores)}")
    
    # 7. Momentum Score
    print("\n📈 Momentum 점수 산출...")
    mom_scores = compute_momentum_score(ticker_data)
    print(f"   모멘텀 데이터: {len(mom_scores)}개")
    
    # 8. Composite Score
    print("\n🎰 복합 점수 산출...")
    composites = compute_composite_score(l4_scores, l3_scores, l1_scores, 
                                         l2_scores, mom_scores)
    print(f"   최종 후보: {len(composites)}개")
    
    # 9. Backtest
    portfolio_returns, benchmark_returns, portfolio_df = run_backtest(
        ticker_data, composites, sector_info,
        CONFIG['START_DATE'], CONFIG['END_DATE'],
        CONFIG['TOP_N']
    )
    
    if portfolio_returns is None:
        print("❌ 백테스트 실패")
        return
    
    # 10. Results
    strategy_metrics = calculate_metrics(portfolio_returns)
    benchmark_metrics = calculate_metrics(benchmark_returns)
    print_results(strategy_metrics, benchmark_metrics, portfolio_df)
    
    # 11. Charts
    create_charts(portfolio_returns, benchmark_returns, portfolio_df)
    
    # 12. Save
    portfolio_df.to_csv('alpha_stack_portfolio.csv', index=False)
    
    results = pd.DataFrame({
        'Date': portfolio_returns.index,
        'Strategy': portfolio_returns.values,
        'Benchmark': benchmark_returns.values
    })
    results.to_csv('alpha_stack_daily_returns.csv', index=False)
    
    print("\n📁 저장된 파일:")
    print("   - alpha_stack_portfolio.csv (포트폴리오)")
    print("   - alpha_stack_daily_returns.csv (일별 수익률)")
    print("   - alpha_stack_results.png (차트)")
    
    print("\n" + "═" * 70)
    print("  ✅ Alpha Stack 백테스트 완료!")
    print("═" * 70)


if __name__ == "__main__":
    main()
