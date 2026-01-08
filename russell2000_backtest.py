"""
Russell 2000 소형주 퀄리티+성장 퀀트 전략 백테스트
===================================================
실제 IWM ETF 보유 목록 기반 (1,959개 종목)

사용법:
1. yfinance 설치: pip install yfinance pandas numpy matplotlib
2. 이 스크립트 실행:
   python russell2000_backtest.py           # 캐시 사용 (있으면)
   python russell2000_backtest.py --refresh # 강제 새로고침
3. russell2000_tickers.csv 파일이 같은 폴더에 있어야 함

캐싱 (스마트 캐시 공유):
- 펀더멘털 데이터: stock_data_cache.pkl (generate_portfolio.py와 공유)
- 가격 이력 데이터: price_history_cache.pkl (백테스트 전용)
- 기본 유효기간: 7일
- 두 번째 실행부터는 몇 초 만에 분석 시작
- portfolio generator와 fundamental 데이터 공유로 중복 다운로드 방지
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import time
import os
import pickle
import sys
import hashlib
warnings.filterwarnings('ignore')

# ============================================================
# 설정
# ============================================================
CONFIG = {
    'START_DATE': '2021-01-01',
    'END_DATE': datetime.now().strftime('%Y-%m-%d'),
    'TOP_N': 20,                    # 포트폴리오 종목 수
    'REBALANCE_FREQ': 'Q',          # Q: 분기, M: 월별
    'MAX_SECTOR_WEIGHT': 0.30,      # 섹터 집중 제한
    'MIN_MARKET_CAP': 300e6,        # 최소 시가총액 $300M
    'MAX_MARKET_CAP': 10e9,         # 최대 시가총액 $10B
    'MIN_AVG_VOLUME': 100000,       # 최소 평균 거래량
    'INITIAL_BALANCE': 100000,      # 초기 투자 금액 ($)
    'BATCH_SIZE': 50,               # API 호출 배치 크기
    'SLEEP_TIME': 1,                # API 호출 간 대기 시간
    'FUND_CACHE_FILE': 'stock_data_cache.pkl',  # 펀더멘털 캐시
    'PRICE_CACHE_FILE': 'price_history_cache.pkl',  # 가격 이력 캐시
    'CACHE_DAYS': 7,                # 캐시 유효 기간 (일)
    'FORCE_REFRESH': False,         # True = 강제 새로고침
}

# ============================================================
# 1. 데이터 로드
# ============================================================
def load_russell2000_tickers(filepath='russell2000_tickers.csv'):
    """Russell 2000 티커 목록 로드"""
    df = pd.read_csv(filepath)
    print(f"Russell 2000 종목 로드: {len(df)}개")
    print(f"섹터 분포:\n{df['Sector'].value_counts()}\n")
    return df


def load_fundamental_cache(cache_file, max_age_days):
    """펀더멘털 데이터 캐시 로드 (portfolio generator와 공유)"""
    if not os.path.exists(cache_file):
        print(f"펀더멘털 캐시 없음: {cache_file}")
        return None

    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        cache_time = cache_data['timestamp']
        age = datetime.now() - cache_time

        if age.days > max_age_days:
            print(f"펀더멘털 캐시 만료: {age.days}일 경과 (최대 {max_age_days}일)")
            return None

        print(f"펀더멘털 캐시 로드: {cache_file}")
        print(f"  데이터: {len(cache_data['data'])}개 종목")
        print(f"  생성: {age.days}일 전")

        return cache_data['data']

    except Exception as e:
        print(f"펀더멘털 캐시 로드 실패: {e}")
        return None


def save_fundamental_cache(data, cache_file):
    """펀더멘털 데이터 캐시 저장 (portfolio generator와 공유)"""
    cache_data = {
        'timestamp': datetime.now(),
        'data': data
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"펀더멘털 캐시 저장: {cache_file} ({len(data)}개 종목)")


def load_price_cache(cache_file, start_date, end_date, max_age_days):
    """가격 이력 캐시 로드"""
    if not os.path.exists(cache_file):
        print(f"가격 이력 캐시 없음: {cache_file}")
        return None

    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        cache_time = cache_data['timestamp']
        age = datetime.now() - cache_time

        # 날짜 범위 확인
        if cache_data['start_date'] != start_date or cache_data['end_date'] != end_date:
            print(f"가격 이력 캐시 날짜 불일치 (캐시: {cache_data['start_date']}~{cache_data['end_date']})")
            return None

        if age.days > max_age_days:
            print(f"가격 이력 캐시 만료: {age.days}일 경과 (최대 {max_age_days}일)")
            return None

        print(f"가격 이력 캐시 로드: {cache_file}")
        print(f"  데이터: {len(cache_data['data'])}개 종목")
        print(f"  날짜: {start_date} ~ {end_date}")
        print(f"  생성: {age.days}일 전")

        return cache_data['data']

    except Exception as e:
        print(f"가격 이력 캐시 로드 실패: {e}")
        return None


def save_price_cache(data, cache_file, start_date, end_date):
    """가격 이력 캐시 저장"""
    cache_data = {
        'timestamp': datetime.now(),
        'start_date': start_date,
        'end_date': end_date,
        'data': data
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"가격 이력 캐시 저장: {cache_file} ({len(data)}개 종목)")


def load_combined_cache(fund_cache_file, price_cache_file, start_date, end_date, max_age_days):
    """펀더멘털 + 가격 캐시를 결합하여 로드"""
    fund_data = load_fundamental_cache(fund_cache_file, max_age_days)
    price_data = load_price_cache(price_cache_file, start_date, end_date, max_age_days)

    if fund_data is None or price_data is None:
        return None

    # 두 캐시를 결합
    combined_data = {}
    for ticker in price_data.keys():
        if ticker in fund_data:
            combined_data[ticker] = {
                'price': price_data[ticker],
                'info': fund_data[ticker]
            }

    print(f"\n캐시 결합 완료: {len(combined_data)}개 종목")
    return combined_data


def save_combined_cache(data, fund_cache_file, price_cache_file, start_date, end_date):
    """펀더멘털 + 가격 데이터를 분리하여 캐시 저장"""
    # 펀더멘털 데이터 분리
    fund_data = {}
    price_data = {}

    for ticker, ticker_data in data.items():
        if 'info' in ticker_data:
            fund_data[ticker] = ticker_data['info']
        if 'price' in ticker_data:
            price_data[ticker] = ticker_data['price']

    # 각각 저장
    save_fundamental_cache(fund_data, fund_cache_file)
    save_price_cache(price_data, price_cache_file, start_date, end_date)


def get_stock_data_batch(tickers, start_date, end_date, batch_size=50):
    """
    배치로 주가 및 재무 데이터 수집
    yfinance API 제한을 고려한 배치 처리
    """
    print(f"\n데이터 수집 시작 ({len(tickers)}개 종목)...")
    
    all_data = {}
    failed_tickers = []
    
    # 배치로 나누기
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        print(f"  배치 {batch_idx+1}/{len(batches)} 처리 중...")
        
        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)
                
                # 주가 데이터
                hist = stock.history(start=start_date, end=end_date)
                if hist.empty or len(hist) < 50:
                    failed_tickers.append(ticker)
                    continue
                
                # 재무 데이터
                info = stock.info
                
                all_data[ticker] = {
                    'price': hist,
                    'info': info
                }
                
            except Exception as e:
                failed_tickers.append(ticker)
                continue
        
        # API 제한 방지
        if batch_idx < len(batches) - 1:
            time.sleep(CONFIG['SLEEP_TIME'])
    
    print(f"데이터 수집 완료: {len(all_data)}개 성공, {len(failed_tickers)}개 실패")
    return all_data, failed_tickers


# ============================================================
# 2. 팩터 계산
# ============================================================
def calculate_factors(stock_data, sector_info):
    """각 종목별 퀄리티, 성장, 밸류, 모멘텀 팩터 계산"""
    
    print("\n팩터 계산 중...")
    factors_list = []
    
    for ticker, data in stock_data.items():
        try:
            info = data['info']
            price_data = data['price']
            
            # 기본 필터
            market_cap = info.get('marketCap', 0)
            if market_cap < CONFIG['MIN_MARKET_CAP'] or market_cap > CONFIG['MAX_MARKET_CAP']:
                continue
            
            avg_volume = info.get('averageVolume', 0)
            if avg_volume < CONFIG['MIN_AVG_VOLUME']:
                continue
            
            # 섹터 정보
            sector = sector_info.get(ticker, info.get('sector', 'Unknown'))
            
            # ---- 퀄리티 팩터 ----
            roe = info.get('returnOnEquity', np.nan)
            roa = info.get('returnOnAssets', np.nan)
            gross_margin = info.get('grossMargins', np.nan)
            operating_margin = info.get('operatingMargins', np.nan)
            debt_equity = info.get('debtToEquity', np.nan)
            current_ratio = info.get('currentRatio', np.nan)
            
            # ---- 성장 팩터 ----
            revenue_growth = info.get('revenueGrowth', np.nan)
            earnings_growth = info.get('earningsGrowth', np.nan)
            earnings_quarterly_growth = info.get('earningsQuarterlyGrowth', np.nan)
            
            # ---- 밸류 팩터 ----
            pe_ratio = info.get('trailingPE', np.nan)
            forward_pe = info.get('forwardPE', np.nan)
            pb_ratio = info.get('priceToBook', np.nan)
            peg_ratio = info.get('pegRatio', np.nan)
            ps_ratio = info.get('priceToSalesTrailing12Months', np.nan)
            ev_ebitda = info.get('enterpriseToEbitda', np.nan)
            
            # ---- 모멘텀 팩터 ----
            if len(price_data) >= 252:
                price_12m_ago = price_data['Close'].iloc[-252]
                price_1m_ago = price_data['Close'].iloc[-21]
                price_now = price_data['Close'].iloc[-1]
                
                mom_12m = (price_now / price_12m_ago - 1) if price_12m_ago > 0 else np.nan
                mom_1m = (price_now / price_1m_ago - 1) if price_1m_ago > 0 else np.nan
                momentum_12_1 = mom_12m - mom_1m  # 12-1 모멘텀 (최근 1개월 제외)
            elif len(price_data) >= 63:
                price_3m_ago = price_data['Close'].iloc[-63]
                price_now = price_data['Close'].iloc[-1]
                momentum_12_1 = (price_now / price_3m_ago - 1) if price_3m_ago > 0 else np.nan
            else:
                momentum_12_1 = np.nan
            
            # 변동성
            daily_returns = price_data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 20 else np.nan
            
            factors_list.append({
                'ticker': ticker,
                'sector': sector,
                'market_cap': market_cap,
                'avg_volume': avg_volume,
                # 퀄리티
                'roe': roe,
                'roa': roa,
                'gross_margin': gross_margin,
                'operating_margin': operating_margin,
                'debt_equity': debt_equity,
                'current_ratio': current_ratio,
                # 성장
                'revenue_growth': revenue_growth,
                'earnings_growth': earnings_growth,
                'earnings_quarterly_growth': earnings_quarterly_growth,
                # 밸류
                'pe_ratio': pe_ratio,
                'forward_pe': forward_pe,
                'pb_ratio': pb_ratio,
                'peg_ratio': peg_ratio,
                'ps_ratio': ps_ratio,
                'ev_ebitda': ev_ebitda,
                # 모멘텀
                'momentum': momentum_12_1,
                'volatility': volatility
            })
            
        except Exception as e:
            continue
    
    df = pd.DataFrame(factors_list)
    print(f"팩터 계산 완료: {len(df)}개 종목")
    return df


def calculate_composite_score(df):
    """종합 스코어 계산"""
    
    df = df.copy()
    
    def zscore(series):
        """Z-score 정규화 (이상치 처리 포함)"""
        # 상하위 5% 윈저라이징
        lower = series.quantile(0.05)
        upper = series.quantile(0.95)
        clipped = series.clip(lower, upper)
        std = clipped.std()
        # Handle zero or NaN standard deviation
        if pd.isna(std) or std == 0:
            return pd.Series(0, index=series.index)
        return (clipped - clipped.mean()) / std

    def safe_zscore(series, reverse=False):
        """안전한 Z-score 계산"""
        # Fill NaN with median, or 0 if all NaN
        median_val = series.median()
        if pd.isna(median_val):
            filled = series.fillna(0)
        else:
            filled = series.fillna(median_val)
        z = zscore(filled)
        return -z if reverse else z
    
    # ========== 퀄리티 스코어 (35%) ==========
    df['roe_z'] = safe_zscore(df['roe'])
    df['roa_z'] = safe_zscore(df['roa'])
    df['gross_margin_z'] = safe_zscore(df['gross_margin'])
    df['operating_margin_z'] = safe_zscore(df['operating_margin'])
    df['debt_equity_z'] = safe_zscore(df['debt_equity'], reverse=True)  # 낮을수록 좋음
    
    df['quality_score'] = (
        df['roe_z'] * 0.25 +
        df['roa_z'] * 0.20 +
        df['gross_margin_z'] * 0.20 +
        df['operating_margin_z'] * 0.20 +
        df['debt_equity_z'] * 0.15
    )
    
    # ========== 성장 스코어 (35%) ==========
    df['rev_growth_z'] = safe_zscore(df['revenue_growth'])
    df['earn_growth_z'] = safe_zscore(df['earnings_growth'])
    df['earn_q_growth_z'] = safe_zscore(df['earnings_quarterly_growth'])
    
    df['growth_score'] = (
        df['rev_growth_z'] * 0.40 +
        df['earn_growth_z'] * 0.35 +
        df['earn_q_growth_z'] * 0.25
    )
    
    # ========== 밸류 스코어 (15%) ==========
    # 밸류 지표는 낮을수록 좋음 (저평가)
    # Convert to numeric first to handle any string values
    df['pe_z'] = safe_zscore(pd.to_numeric(df['pe_ratio'], errors='coerce').clip(upper=100), reverse=True)
    df['peg_z'] = safe_zscore(pd.to_numeric(df['peg_ratio'], errors='coerce').clip(upper=5), reverse=True)
    df['ps_z'] = safe_zscore(pd.to_numeric(df['ps_ratio'], errors='coerce').clip(upper=20), reverse=True)
    
    df['value_score'] = (
        df['pe_z'] * 0.35 +
        df['peg_z'] * 0.35 +
        df['ps_z'] * 0.30
    )
    
    # ========== 모멘텀 스코어 (15%) ==========
    df['momentum_z'] = safe_zscore(df['momentum'])
    df['vol_z'] = safe_zscore(df['volatility'], reverse=True)  # 낮은 변동성 선호
    
    df['momentum_score'] = (
        df['momentum_z'] * 0.70 +
        df['vol_z'] * 0.30
    )
    
    # ========== 종합 스코어 ==========
    df['composite_score'] = (
        df['quality_score'] * 0.35 +
        df['growth_score'] * 0.35 +
        df['value_score'] * 0.15 +
        df['momentum_score'] * 0.15
    )
    
    return df


def select_portfolio(factors_df, top_n=30, max_sector_weight=0.30):
    """
    포트폴리오 종목 선정 (섹터 제한 적용)
    """
    df = factors_df.sort_values('composite_score', ascending=False).copy()
    
    selected = []
    sector_counts = {}
    max_per_sector = int(top_n * max_sector_weight)
    
    for _, row in df.iterrows():
        if len(selected) >= top_n:
            break
        
        sector = row['sector']
        current_count = sector_counts.get(sector, 0)
        
        if current_count < max_per_sector:
            selected.append(row['ticker'])
            sector_counts[sector] = current_count + 1
    
    return selected


# ============================================================
# 3. 백테스트 엔진
# ============================================================
def run_backtest(stock_data, factors_df, start_date, end_date, 
                 top_n=30, rebalance_freq='Q'):
    """
    백테스트 실행
    """
    print(f"\n백테스트 실행 중...")
    print(f"  - 기간: {start_date} ~ {end_date}")
    print(f"  - 포트폴리오 종목 수: {top_n}")
    print(f"  - 리밸런싱: {'분기별' if rebalance_freq == 'Q' else '월별'}")
    
    # 수익률 데이터 준비
    returns_dict = {}
    for ticker, data in stock_data.items():
        if ticker in factors_df['ticker'].values:
            price = data['price']['Close']
            returns_dict[ticker] = price.pct_change()
    
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna(how='all')

    # Remove timezone from index to avoid comparison issues
    if returns_df.index.tz is not None:
        returns_df.index = returns_df.index.tz_localize(None)

    # 리밸런싱 날짜
    if rebalance_freq == 'Q':
        rebal_dates = pd.date_range(start=start_date, end=end_date, freq='QS')
    else:
        rebal_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    portfolio_returns = []
    benchmark_returns = []
    holdings_history = []
    turnover_list = []

    prev_holdings = set()
    prev_holdings_dict = {}  # Track previous holdings with entry info

    # Initial portfolio value (normalized to 1)
    portfolio_value = 1.0

    for i, date in enumerate(rebal_dates):
        # 기간 설정
        if i < len(rebal_dates) - 1:
            next_date = rebal_dates[i + 1]
        else:
            next_date = pd.Timestamp(end_date)

        period_returns = returns_df[(returns_df.index >= date) & (returns_df.index < next_date)]

        if period_returns.empty:
            continue

        # 포트폴리오 선정
        current_holdings = select_portfolio(
            factors_df,
            top_n=top_n,
            max_sector_weight=CONFIG['MAX_SECTOR_WEIGHT']
        )

        # 유효한 종목만
        valid_holdings = [t for t in current_holdings if t in period_returns.columns]

        if len(valid_holdings) < 5:
            continue

        # Get entry prices (first available price at or after rebalance date)
        entry_prices = {}
        for ticker in valid_holdings:
            ticker_data = stock_data.get(ticker)
            if ticker_data and 'price' in ticker_data:
                price_data = ticker_data['price']['Close']
                # Remove timezone if present
                if price_data.index.tz is not None:
                    price_data.index = price_data.index.tz_localize(None)
                available_prices = price_data[price_data.index >= date]
                if len(available_prices) > 0:
                    entry_prices[ticker] = available_prices.iloc[0]

        # Calculate position info for each holding
        position_value = portfolio_value / len(valid_holdings)  # Equal weight
        holdings_with_prices = []

        for ticker in valid_holdings:
            entry_price = entry_prices.get(ticker, np.nan)
            if pd.notna(entry_price) and entry_price > 0:
                n_shares = position_value / entry_price
            else:
                n_shares = np.nan

            holdings_with_prices.append({
                'ticker': ticker,
                'entry_price': entry_price,
                'position_value': position_value,
                'n_shares': n_shares,
                'entry_date': date
            })

        # 회전율 계산
        current_set = set(valid_holdings)
        if prev_holdings:
            turnover = len(current_set.symmetric_difference(prev_holdings)) / (2 * len(current_set))
            turnover_list.append(turnover)
        prev_holdings = current_set

        # Calculate exit info for previous holdings
        if prev_holdings_dict:
            for holding_info in prev_holdings_dict:
                ticker = holding_info['ticker']
                # Get exit price (last price before next rebalance or at end)
                ticker_data = stock_data.get(ticker)
                if ticker_data and 'price' in ticker_data:
                    price_data = ticker_data['price']['Close']
                    if price_data.index.tz is not None:
                        price_data.index = price_data.index.tz_localize(None)
                    exit_prices = price_data[(price_data.index >= holding_info['entry_date']) &
                                            (price_data.index < date)]
                    if len(exit_prices) > 0:
                        holding_info['exit_price'] = exit_prices.iloc[-1]
                        holding_info['exit_date'] = date
                        if pd.notna(holding_info.get('n_shares')) and holding_info['n_shares'] > 0:
                            holding_info['exit_value'] = holding_info['exit_price'] * holding_info['n_shares']
                            holding_info['profit'] = holding_info['exit_value'] - holding_info['position_value']
                            holding_info['profit_pct'] = (holding_info['exit_price'] / holding_info['entry_price'] - 1) * 100

        # Save previous holdings to history before updating
        if prev_holdings_dict:
            for holding_info in prev_holdings_dict:
                holdings_history.append(holding_info.copy())

        # Update previous holdings for next iteration
        prev_holdings_dict = holdings_with_prices

        # 포트폴리오 수익률 (동일가중)
        port_ret = period_returns[valid_holdings].mean(axis=1)
        portfolio_returns.append(port_ret)

        # Update portfolio value for next rebalance
        period_total_return = (1 + port_ret).prod() - 1
        portfolio_value = portfolio_value * (1 + period_total_return)

        # 벤치마크 (전체 유니버스 동일가중)
        all_tickers = [t for t in factors_df['ticker'].values if t in period_returns.columns]
        bench_ret = period_returns[all_tickers].mean(axis=1)
        benchmark_returns.append(bench_ret)

    # Handle final holdings (still open positions)
    if prev_holdings_dict:
        final_date = pd.Timestamp(end_date)
        for holding_info in prev_holdings_dict:
            ticker = holding_info['ticker']
            # Get final exit price (last available price)
            ticker_data = stock_data.get(ticker)
            if ticker_data and 'price' in ticker_data:
                price_data = ticker_data['price']['Close']
                if price_data.index.tz is not None:
                    price_data.index = price_data.index.tz_localize(None)
                exit_prices = price_data[price_data.index >= holding_info['entry_date']]
                if len(exit_prices) > 0:
                    holding_info['exit_price'] = exit_prices.iloc[-1]
                    holding_info['exit_date'] = final_date
                    if pd.notna(holding_info.get('n_shares')) and holding_info['n_shares'] > 0:
                        holding_info['exit_value'] = holding_info['exit_price'] * holding_info['n_shares']
                        holding_info['profit'] = holding_info['exit_value'] - holding_info['position_value']
                        holding_info['profit_pct'] = (holding_info['exit_price'] / holding_info['entry_price'] - 1) * 100
            holdings_history.append(holding_info.copy())

    portfolio_series = pd.concat(portfolio_returns)
    benchmark_series = pd.concat(benchmark_returns)
    
    avg_turnover = np.mean(turnover_list) if turnover_list else 0
    print(f"  - 평균 회전율: {avg_turnover:.1%}")
    
    return portfolio_series, benchmark_series, holdings_history


# ============================================================
# 4. 성과 분석
# ============================================================
def calculate_metrics(returns, risk_free_rate=0.04):
    """종합 성과 지표 계산"""
    
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    annual_volatility = returns.std() * np.sqrt(252)
    
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()
    
    win_rate = (returns > 0).sum() / len(returns)
    
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar,
        'Win Rate': win_rate
    }


def print_results(strategy_metrics, benchmark_metrics):
    """결과 출력"""
    
    print("\n" + "=" * 70)
    print("                      백테스트 성과 비교")
    print("=" * 70)
    
    metrics = [
        ('Total Return', '{:.2%}'),
        ('Annual Return', '{:.2%}'),
        ('Annual Volatility', '{:.2%}'),
        ('Sharpe Ratio', '{:.2f}'),
        ('Sortino Ratio', '{:.2f}'),
        ('Max Drawdown', '{:.2%}'),
        ('Calmar Ratio', '{:.2f}'),
        ('Win Rate', '{:.2%}'),
    ]
    
    print(f"\n{'지표':<25} {'전략':>15} {'벤치마크':>15} {'차이':>15}")
    print("-" * 70)
    
    for metric, fmt in metrics:
        strat = strategy_metrics[metric]
        bench = benchmark_metrics[metric]
        diff = strat - bench
        
        print(f"{metric:<25} {fmt.format(strat):>15} {fmt.format(bench):>15} "
              f"{('+' if diff > 0 else '')}{fmt.format(diff):>14}")
    
    print("=" * 70)


def create_charts(portfolio_returns, benchmark_returns, factors_df, output_dir='.'):
    """결과 차트 생성"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 누적 수익률
    port_cum = (1 + portfolio_returns).cumprod()
    bench_cum = (1 + benchmark_returns).cumprod()
    
    ax = axes[0, 0]
    ax.plot(port_cum.index, port_cum.values, label='Strategy', linewidth=2, color='#2E86AB')
    ax.plot(bench_cum.index, bench_cum.values, label='Benchmark', linewidth=2, color='#A23B72', alpha=0.7)
    ax.fill_between(port_cum.index, port_cum.values, bench_cum.values,
                    where=port_cum.values >= bench_cum.values, color='#2E86AB', alpha=0.1)
    ax.set_title('Cumulative Returns: Small-Cap Quality + Growth', fontsize=14, fontweight='bold')
    ax.set_ylabel('Growth of $1')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 2. 드로우다운
    ax = axes[0, 1]
    rolling_max = port_cum.expanding().max()
    drawdown = (port_cum / rolling_max - 1) * 100
    ax.fill_between(drawdown.index, drawdown.values, 0, color='#E74C3C', alpha=0.6)
    ax.set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    
    # 3. 월별 수익률
    ax = axes[1, 0]
    monthly = portfolio_returns.resample('M').apply(lambda x: (1+x).prod() - 1) * 100
    colors = ['#E74C3C' if x < 0 else '#27AE60' for x in monthly.values]
    ax.bar(monthly.index, monthly.values, width=20, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('Monthly Returns', fontsize=14, fontweight='bold')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 섹터 분포 (상위 종목)
    ax = axes[1, 1]
    top_sectors = factors_df.nlargest(30, 'composite_score')['sector'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_sectors)))
    ax.pie(top_sectors.values, labels=top_sectors.index, autopct='%1.0f%%', 
           colors=colors, startangle=90)
    ax.set_title('Sector Distribution (Top 30)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/backtest_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n차트 저장: {output_dir}/backtest_results.png")


def generate_quarterly_summary(details_df, initial_balance=100000):
    """
    분기별 성과 요약 생성

    Args:
        details_df: 포지션 상세 데이터프레임
        initial_balance: 초기 투자 금액 (기본값: $100,000)
    """
    # Convert dates to datetime and extract quarter
    details_df = details_df.copy()
    details_df['Exit_Date'] = pd.to_datetime(details_df['Exit_Date'], errors='coerce')
    details_df['Quarter'] = details_df['Exit_Date'].dt.to_period('Q')

    # Filter out positions without exit dates
    closed_positions = details_df[details_df['Quarter'].notna()].copy()

    if len(closed_positions) == 0:
        return pd.DataFrame()

    # Scale profits to initial balance (백테스트는 normalized units 사용, 1.0 = 100%)
    # Profit is in normalized units, so multiply by initial_balance to get dollar amounts
    closed_positions['Profit_Scaled'] = closed_positions['Profit'] * initial_balance

    # Group by quarter and calculate metrics
    quarterly_groups = closed_positions.groupby('Quarter')

    summary_rows = []
    for quarter, group in quarterly_groups:
        # Basic metrics
        n_trades = len(group)
        total_profit = group['Profit_Scaled'].sum()  # Use scaled profit
        total_profit_pct = group['Profit_Pct'].mean()  # Average return per trade

        # Win/Loss metrics
        winners = group[group['Profit_Pct'] > 0]
        losers = group[group['Profit_Pct'] <= 0]
        win_rate = len(winners) / n_trades if n_trades > 0 else 0

        # Average profits
        avg_win = winners['Profit_Pct'].mean() if len(winners) > 0 else 0
        avg_loss = losers['Profit_Pct'].mean() if len(losers) > 0 else 0

        # Best/Worst trades
        best_trade = group['Profit_Pct'].max()
        worst_trade = group['Profit_Pct'].min()
        best_ticker = group.loc[group['Profit_Pct'].idxmax(), 'Ticker'] if n_trades > 0 else ''
        worst_ticker = group.loc[group['Profit_Pct'].idxmin(), 'Ticker'] if n_trades > 0 else ''

        # Position metrics
        avg_position_value = group['Position_Value'].mean()
        total_capital_deployed = group['Position_Value'].sum()

        # Sector breakdown (top 3 sectors)
        if 'Sector' in group.columns:
            sector_counts = group['Sector'].value_counts()
            top_sectors = ', '.join([f"{s}({c})" for s, c in sector_counts.head(3).items()])
        else:
            top_sectors = 'N/A'

        summary_rows.append({
            'Quarter': str(quarter),
            'N_Trades': n_trades,
            'Total_Profit': total_profit,
            'Avg_Return_Pct': total_profit_pct,
            'Win_Rate': win_rate * 100,
            'N_Winners': len(winners),
            'N_Losers': len(losers),
            'Avg_Win_Pct': avg_win,
            'Avg_Loss_Pct': avg_loss,
            'Best_Trade_Pct': best_trade,
            'Best_Ticker': best_ticker,
            'Worst_Trade_Pct': worst_trade,
            'Worst_Ticker': worst_ticker,
            'Avg_Position_Value': avg_position_value,
            'Total_Capital_Deployed': total_capital_deployed,
            'Top_Sectors': top_sectors
        })

    summary_df = pd.DataFrame(summary_rows)

    # Add cumulative metrics
    summary_df['Cumulative_Profit'] = summary_df['Total_Profit'].cumsum()
    summary_df['Cumulative_Trades'] = summary_df['N_Trades'].cumsum()

    # Calculate portfolio value (initial balance + cumulative profit)
    summary_df['Portfolio_Value'] = initial_balance + summary_df['Cumulative_Profit']

    # Calculate cumulative return percentage
    summary_df['Cumulative_Return_Pct'] = (summary_df['Cumulative_Profit'] / initial_balance) * 100

    return summary_df


def visualize_quarterly_summary(quarterly_df, output_dir='.'):
    """
    분기별 성과 시각화
    """
    if quarterly_df is None or len(quarterly_df) == 0:
        print("분기별 요약 데이터가 없어 시각화를 건너뜁니다.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 분기별 평균 수익률
    ax = axes[0, 0]
    colors = ['#E74C3C' if x < 0 else '#27AE60' for x in quarterly_df['Avg_Return_Pct'].values]
    bars = ax.bar(range(len(quarterly_df)), quarterly_df['Avg_Return_Pct'].values,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_title('Quarterly Average Returns', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Return per Trade (%)')
    ax.set_xlabel('Quarter')
    ax.set_xticks(range(len(quarterly_df)))
    ax.set_xticklabels(quarterly_df['Quarter'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8, fontweight='bold')

    # 2. 승률 추이
    ax = axes[0, 1]
    ax.plot(range(len(quarterly_df)), quarterly_df['Win_Rate'].values,
            marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% breakeven')
    ax.fill_between(range(len(quarterly_df)), quarterly_df['Win_Rate'].values, 50,
                     where=quarterly_df['Win_Rate'].values >= 50,
                     color='#27AE60', alpha=0.2, label='Above 50%')
    ax.set_title('Win Rate Trend', fontsize=14, fontweight='bold')
    ax.set_ylabel('Win Rate (%)')
    ax.set_xlabel('Quarter')
    ax.set_xticks(range(len(quarterly_df)))
    ax.set_xticklabels(quarterly_df['Quarter'], rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Add value labels
    for i, val in enumerate(quarterly_df['Win_Rate'].values):
        ax.text(i, val + 2, f'{val:.1f}%', ha='center', fontsize=8)

    # 3. 거래 건수 및 포트폴리오 가치
    ax = axes[1, 0]
    ax2 = ax.twinx()

    # 거래 건수 (막대)
    bars = ax.bar(range(len(quarterly_df)), quarterly_df['N_Trades'].values,
                   color='#95A5A6', alpha=0.6, label='Number of Trades')
    ax.set_ylabel('Number of Trades', color='#95A5A6', fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#95A5A6')
    ax.set_xlabel('Quarter')
    ax.set_xticks(range(len(quarterly_df)))
    ax.set_xticklabels(quarterly_df['Quarter'], rotation=45, ha='right')

    # 포트폴리오 가치 (선) - $100,000 초기 투자 기준
    portfolio_values = quarterly_df['Portfolio_Value'].values
    line = ax2.plot(range(len(quarterly_df)), portfolio_values,
                     marker='o', linewidth=2.5, markersize=8, color='#2E86AB',
                     label='Portfolio Value ($100K start)')
    ax2.set_ylabel('Portfolio Value ($)', color='#2E86AB', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2E86AB')

    # Format y-axis to avoid scientific notation
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    # 초기 투자액 기준선 표시
    initial_balance = portfolio_values[0] - quarterly_df['Cumulative_Profit'].values[0]
    ax2.axhline(y=initial_balance, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_title('Trading Activity & Portfolio Growth ($100K Initial)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 4. 평균 승패 비교
    ax = axes[1, 1]
    x = range(len(quarterly_df))
    width = 0.35

    wins = ax.bar([i - width/2 for i in x], quarterly_df['Avg_Win_Pct'].values,
                   width, label='Avg Win', color='#27AE60', alpha=0.7)
    losses = ax.bar([i + width/2 for i in x], quarterly_df['Avg_Loss_Pct'].values,
                     width, label='Avg Loss', color='#E74C3C', alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_title('Average Win vs Loss per Quarter', fontsize=14, fontweight='bold')
    ax.set_ylabel('Return (%)')
    ax.set_xlabel('Quarter')
    ax.set_xticks(x)
    ax.set_xticklabels(quarterly_df['Quarter'], rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (win, loss) in enumerate(zip(quarterly_df['Avg_Win_Pct'].values,
                                         quarterly_df['Avg_Loss_Pct'].values)):
        ax.text(i - width/2, win + 0.5, f'{win:.1f}%', ha='center', fontsize=7)
        ax.text(i + width/2, loss - 0.5, f'{loss:.1f}%', ha='center', fontsize=7, va='top')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/quarterly_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"분기별 요약 차트 저장: {output_dir}/quarterly_summary.png")


def save_rebalancing_details(holdings_history, factors_df, output_dir='.'):
    """
    리밸런싱 상세 내역 저장

    세 개의 파일 생성:
    1. rebalancing_history.csv: 각 리밸런싱 날짜의 보유 종목 목록 (요약)
    2. rebalancing_details.csv: 각 보유 종목의 상세 정보 (포지션, 가격, 수익 등)
    3. quarterly_summary.csv: 분기별 성과 요약
    """

    # 1. 리밸런싱 이력 (요약 - 날짜별 그룹화)
    history_rows = []
    holdings_by_date = {}

    # Group holdings by entry date
    for entry in holdings_history:
        entry_date = entry.get('entry_date')
        if entry_date:
            date_str = entry_date.strftime('%Y-%m-%d')
            if date_str not in holdings_by_date:
                holdings_by_date[date_str] = []
            holdings_by_date[date_str].append(entry['ticker'])

    for date_str, tickers in sorted(holdings_by_date.items()):
        history_rows.append({
            'Date': date_str,
            'N_Holdings': len(tickers),
            'Tickers': ', '.join(tickers)
        })

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(f'{output_dir}/rebalancing_history.csv', index=False)

    # 2. 리밸런싱 상세 내역 (각 포지션의 전체 정보)
    details_rows = []

    for entry in holdings_history:
        ticker = entry['ticker']

        # 해당 종목의 팩터 정보 찾기
        stock_info = factors_df[factors_df['ticker'] == ticker]
        if not stock_info.empty:
            stock_info = stock_info.iloc[0]
        else:
            stock_info = {}

        details_rows.append({
            'Entry_Date': entry.get('entry_date', pd.NaT).strftime('%Y-%m-%d') if pd.notna(entry.get('entry_date')) else '',
            'Exit_Date': entry.get('exit_date', pd.NaT).strftime('%Y-%m-%d') if pd.notna(entry.get('exit_date')) else '',
            'Ticker': ticker,
            'Sector': stock_info.get('sector', 'Unknown'),
            'Entry_Price': entry.get('entry_price', np.nan),
            'Exit_Price': entry.get('exit_price', np.nan),
            'Position_Value': entry.get('position_value', np.nan),
            'N_Shares': entry.get('n_shares', np.nan),
            'Exit_Value': entry.get('exit_value', np.nan),
            'Profit': entry.get('profit', np.nan),
            'Profit_Pct': entry.get('profit_pct', np.nan),
            'Composite_Score': stock_info.get('composite_score', np.nan),
            'Quality_Score': stock_info.get('quality_score', np.nan),
            'Growth_Score': stock_info.get('growth_score', np.nan),
            'ROE': stock_info.get('roe', np.nan),
            'Revenue_Growth': stock_info.get('revenue_growth', np.nan),
        })

    details_df = pd.DataFrame(details_rows)
    details_df.to_csv(f'{output_dir}/rebalancing_details.csv', index=False)

    # 3. 분기별 요약 리포트
    if len(details_df) > 0:
        # $100,000 초기 투자 기준으로 계산
        quarterly_summary = generate_quarterly_summary(details_df, initial_balance=100000)
        quarterly_summary.to_csv(f'{output_dir}/quarterly_summary.csv', index=False)
        # 분기별 요약 시각화
        visualize_quarterly_summary(quarterly_summary, output_dir=output_dir)

    print(f"\n리밸런싱 내역 저장 완료:")
    print(f"  - 총 {len(holdings_history)}개 포지션 추적")
    if len(details_df) > 0:
        avg_profit = details_df['Profit_Pct'].mean()
        win_rate = (details_df['Profit_Pct'] > 0).sum() / len(details_df)
        print(f"  - 평균 수익률: {avg_profit:.2f}%")
        print(f"  - 승률: {win_rate:.1%}")


def generate_current_portfolio(factors_df, top_n=20, max_sector_weight=0.30, output_dir='.', initial_balance=100000):
    """
    현재 시점의 추천 포트폴리오 생성 (실행 가능한 매수 주문 포함)

    Args:
        factors_df: 팩터 데이터프레임
        top_n: 포트폴리오 종목 수
        max_sector_weight: 최대 섹터 비중
        output_dir: 출력 디렉토리
        initial_balance: 초기 투자 금액 (기본값: $100,000)
    """
    print("\n" + "=" * 70)
    print(f"현재 포트폴리오 추천 (as of {datetime.now().strftime('%Y-%m-%d')})")
    print("=" * 70)
    print(f"초기 투자 금액: ${initial_balance:,.2f}")

    # 포트폴리오 선정 (섹터 제한 적용)
    selected_tickers = select_portfolio(factors_df, top_n=top_n, max_sector_weight=max_sector_weight)

    # 선정된 종목의 상세 정보
    portfolio_df = factors_df[factors_df['ticker'].isin(selected_tickers)].copy()
    portfolio_df = portfolio_df.sort_values('composite_score', ascending=False)

    # Equal weight
    portfolio_df['Weight'] = 1.0 / len(portfolio_df)
    position_size = initial_balance / len(portfolio_df)

    # 현재가 조회
    print(f"\n현재가 조회 중 ({len(selected_tickers)}개 종목)...")
    current_prices = {}
    for ticker in selected_tickers:
        try:
            stock = yf.Ticker(ticker)
            # 최근 1일 데이터로 현재가 가져오기
            hist = stock.history(period='1d')
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                current_prices[ticker] = current_price
            else:
                # fallback: info에서 가져오기
                info = stock.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if current_price:
                    current_prices[ticker] = current_price
                else:
                    print(f"  {ticker}: 가격 정보 없음")
                    current_prices[ticker] = None
        except Exception as e:
            print(f"  {ticker}: 오류 ({e})")
            current_prices[ticker] = None

    successful = sum(1 for p in current_prices.values() if p is not None)
    print(f"현재가 조회 완료: {successful}/{len(selected_tickers)}개 성공")

    # 현재가 및 매수 수량 추가
    portfolio_df['Current_Price'] = portfolio_df['ticker'].map(current_prices)
    portfolio_df['Position_Size'] = position_size
    portfolio_df['Shares_to_Buy'] = (portfolio_df['Position_Size'] / portfolio_df['Current_Price']).apply(
        lambda x: int(x) if pd.notna(x) else 0
    )
    portfolio_df['Actual_Amount'] = portfolio_df['Shares_to_Buy'] * portfolio_df['Current_Price']

    # 출력용 DataFrame 구성
    output_df = portfolio_df[[
        'ticker', 'sector', 'market_cap', 'composite_score',
        'quality_score', 'growth_score', 'value_score', 'momentum_score',
        'roe', 'revenue_growth', 'pe_ratio', 'Weight',
        'Current_Price', 'Shares_to_Buy', 'Actual_Amount'
    ]].copy()

    output_df.columns = [
        'Ticker', 'Sector', 'Market_Cap', 'Composite_Score',
        'Quality_Score', 'Growth_Score', 'Value_Score', 'Momentum_Score',
        'ROE', 'Revenue_Growth', 'PE_Ratio', 'Weight',
        'Current_Price', 'Shares_to_Buy', 'Actual_Amount'
    ]

    # 포맷팅
    output_df['Market_Cap_B'] = (output_df['Market_Cap'] / 1e9).round(2)
    output_df['Weight_%'] = (output_df['Weight'] * 100).round(2)
    output_df = output_df.drop(['Market_Cap', 'Weight'], axis=1)

    # CSV 저장
    output_df.to_csv(f'{output_dir}/current_portfolio.csv', index=False)

    # 투자 금액 계산
    total_invested = output_df['Actual_Amount'].sum()
    cash_remaining = initial_balance - total_invested

    # 콘솔 출력
    print(f"\n포트폴리오 구성: {len(output_df)}개 종목 (동일가중)")
    print(f"\n{'Ticker':<8} {'Price':>10} {'Shares':>8} {'Amount':>12} {'Score':>7}")
    print("-" * 70)

    for _, row in output_df.iterrows():
        price_str = f"${row['Current_Price']:.2f}" if pd.notna(row['Current_Price']) else "N/A"
        shares = int(row['Shares_to_Buy']) if pd.notna(row['Shares_to_Buy']) else 0
        amount = row['Actual_Amount'] if pd.notna(row['Actual_Amount']) else 0
        print(f"{row['Ticker']:<8} {price_str:>10} {shares:>8} ${amount:>11,.2f} {row['Composite_Score']:>7.3f}")

    # 섹터 분포
    print("\n섹터 분포:")
    sector_dist = output_df.groupby('Sector').size()
    for sector, count in sector_dist.items():
        pct = count / len(output_df) * 100
        print(f"  {sector}: {count}개 ({pct:.1f}%)")

    # 평균 팩터 스코어
    print("\n평균 팩터 스코어:")
    print(f"  Quality:  {output_df['Quality_Score'].mean():>6.2f}")
    print(f"  Growth:   {output_df['Growth_Score'].mean():>6.2f}")
    print(f"  Value:    {output_df['Value_Score'].mean():>6.2f}")
    print(f"  Momentum: {output_df['Momentum_Score'].mean():>6.2f}")

    # 투자 요약
    print("\n" + "=" * 70)
    print("투자 요약")
    print("=" * 70)
    print(f"초기 금액:      ${initial_balance:>12,.2f}")
    print(f"총 투자 금액:   ${total_invested:>12,.2f} ({total_invested/initial_balance*100:.1f}%)")
    print(f"잔액 (미투자):  ${cash_remaining:>12,.2f} ({cash_remaining/initial_balance*100:.1f}%)")
    print(f"평균 포지션:    ${total_invested/len(output_df):>12,.2f}")

    print(f"\n포트폴리오 저장: {output_dir}/current_portfolio.csv")

    return output_df


# ============================================================
# 5. 메인 실행
# ============================================================
def main():
    # 커맨드 라인 인자 처리
    if '--refresh' in sys.argv or '-r' in sys.argv:
        CONFIG['FORCE_REFRESH'] = True
        print("강제 새로고침 모드\n")

    print("=" * 70)
    print("     Russell 2000 소형주 퀄리티+성장 퀀트 전략 백테스트")
    print("=" * 70)

    # 0. 결과 폴더 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n결과 저장 폴더: {results_dir}/")

    # 1. 티커 목록 로드
    ticker_df = load_russell2000_tickers('russell2000_tickers.csv')
    tickers = ticker_df['Ticker'].tolist()

    # 섹터 정보 딕셔너리
    sector_info = dict(zip(ticker_df['Ticker'], ticker_df['Sector']))

    # 2. 샘플링 (전체 1959개 중 일부만 테스트 - 시간 절약)
    # 실제 운용시에는 전체 사용 권장
    # sample_size = min(100, len(tickers))  # 100개 샘플
    sample_size = len(tickers)
    np.random.seed(42)
    sampled_tickers = np.random.choice(tickers, sample_size, replace=False).tolist()

    print(f"\n샘플 크기: {sample_size}개 (전체 {len(tickers)}개 중)")
    print("(전체 종목 분석시 sample_size = len(tickers) 로 변경)")

    # 3. 캐시 확인 또는 데이터 수집
    stock_data = None

    if not CONFIG['FORCE_REFRESH']:
        stock_data = load_combined_cache(
            CONFIG['FUND_CACHE_FILE'],
            CONFIG['PRICE_CACHE_FILE'],
            CONFIG['START_DATE'],
            CONFIG['END_DATE'],
            CONFIG['CACHE_DAYS']
        )

    if stock_data is None:
        print("\n새로운 데이터 다운로드 중...")
        stock_data, failed = get_stock_data_batch(
            sampled_tickers,
            CONFIG['START_DATE'],
            CONFIG['END_DATE'],
            batch_size=CONFIG['BATCH_SIZE']
        )
        # 캐시 저장 (펀더멘털 + 가격 분리 저장)
        save_combined_cache(
            stock_data,
            CONFIG['FUND_CACHE_FILE'],
            CONFIG['PRICE_CACHE_FILE'],
            CONFIG['START_DATE'],
            CONFIG['END_DATE']
        )
    else:
        print("캐시된 데이터 사용\n")
    
    if len(stock_data) < 50:
        print("충분한 데이터를 수집하지 못했습니다.")
        return
    
    # 4. 팩터 계산 (백테스트용 - 전체 기간 데이터 사용)
    factors_df = calculate_factors(stock_data, sector_info)
    factors_df = calculate_composite_score(factors_df)

    # 4a. 현재 포트폴리오용 팩터 계산 (최근 1년 데이터만 사용 - generate_portfolio와 동일)
    # 최근 1년 데이터로 잘라서 재계산
    current_stock_data = {}
    lookback_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    for ticker, data in stock_data.items():
        if 'price' in data and data['price'] is not None:
            price_data = data['price']
            # 최근 1년으로 필터링
            recent_prices = price_data[price_data.index >= lookback_date]
            current_stock_data[ticker] = {
                'info': data['info'],
                'price': recent_prices
            }
        else:
            current_stock_data[ticker] = data

    # 현재 포트폴리오용 팩터 재계산 (generate_portfolio.py와 동일한 데이터 사용)
    current_factors_df = calculate_factors(current_stock_data, sector_info)
    current_factors_df = calculate_composite_score(current_factors_df)
    
    # 5. 상위 종목 출력
    print("\n" + "=" * 70)
    print("상위 20개 종목 (종합 스코어 기준)")
    print("=" * 70)
    
    top20 = factors_df.nlargest(20, 'composite_score')[
        ['ticker', 'sector', 'market_cap', 'roe', 'revenue_growth', 'pe_ratio', 'composite_score']
    ].copy()
    
    print(f"\n{'Ticker':<8} {'Sector':<20} {'MCap(B)':>8} {'ROE':>8} {'RevGr':>8} {'P/E':>8} {'Score':>8}")
    print("-" * 70)
    
    for _, row in top20.iterrows():
        mcap = row['market_cap'] / 1e9
        roe = row['roe'] * 100 if pd.notna(row['roe']) else 0
        rev_gr = row['revenue_growth'] * 100 if pd.notna(row['revenue_growth']) else 0
        pe = row['pe_ratio'] if pd.notna(row['pe_ratio']) else 0
        
        print(f"{row['ticker']:<8} {row['sector'][:20]:<20} {mcap:>7.2f} {roe:>7.1f}% {rev_gr:>7.1f}% {pe:>7.1f} {row['composite_score']:>7.3f}")

    # 5a. 현재 포트폴리오 생성 (최근 1년 데이터 기반 + 실행 가능한 매수 주문)
    current_portfolio = generate_current_portfolio(
        current_factors_df,  # 최근 1년 데이터로 계산된 팩터 사용
        top_n=CONFIG['TOP_N'],
        max_sector_weight=CONFIG['MAX_SECTOR_WEIGHT'],
        output_dir=results_dir,
        initial_balance=CONFIG['INITIAL_BALANCE']
    )

    # 6. 백테스트
    portfolio_returns, benchmark_returns, holdings = run_backtest(
        stock_data, factors_df,
        CONFIG['START_DATE'], CONFIG['END_DATE'],
        top_n=CONFIG['TOP_N'],
        rebalance_freq=CONFIG['REBALANCE_FREQ']
    )

    # 7. 성과 분석
    strategy_metrics = calculate_metrics(portfolio_returns)
    benchmark_metrics = calculate_metrics(benchmark_returns)
    print_results(strategy_metrics, benchmark_metrics)

    # 8. 차트 생성
    create_charts(portfolio_returns, benchmark_returns, factors_df, output_dir=results_dir)

    # 9. 결과 저장
    results_df = pd.DataFrame({
        'Date': portfolio_returns.index,
        'Strategy': portfolio_returns.values,
        'Benchmark': benchmark_returns.values
    })
    results_df.to_csv(f'{results_dir}/backtest_daily_returns.csv', index=False)
    # 현재 포트폴리오용 팩터 저장 (generate_portfolio.py와 동일)
    current_factors_df.to_csv(f'{results_dir}/factor_scores.csv', index=False)

    # 10. 리밸런싱 상세 내역 저장
    save_rebalancing_details(holdings, factors_df, output_dir=results_dir)

    print("\n저장된 파일:")
    print(f"  폴더: {results_dir}/")
    print("  - current_portfolio.csv (현재 추천 포트폴리오)")
    print("  - backtest_results.png")
    print("  - backtest_daily_returns.csv")
    print("  - factor_scores.csv")
    print("  - rebalancing_history.csv")
    print("  - rebalancing_details.csv (개별 포지션 상세)")
    print("  - quarterly_summary.csv (분기별 성과 요약)")
    
    print("\n" + "=" * 70)
    print("백테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
