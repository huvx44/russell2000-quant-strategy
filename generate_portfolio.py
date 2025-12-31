"""
Russell 2000 포트폴리오 생성기
==============================
백테스트 없이 현재 시점의 추천 포트폴리오만 빠르게 생성
백테스트와 동일한 전략 사용 (Quality 35%, Growth 35%, Value 15%, Momentum 15%)

사용법:
    python generate_portfolio.py           # 캐시 사용 (있으면)
    python generate_portfolio.py --refresh # 강제 새로고침

설정:
    - INITIAL_BALANCE: 초기 투자 금액 (기본값: $100,000)
    - TOP_N: 포트폴리오 종목 수 (기본값: 20)
    - CONFIG 딕셔너리에서 설정 변경 가능

캐싱:
    - 다운로드한 데이터는 stock_data_cache.pkl에 저장
    - 1년치 가격 이력 포함 (모멘텀 계산용)
    - 기본 유효기간: 7일
    - 두 번째 실행부터는 몇 초 만에 완료

출력:
    - current_portfolio_{YYYYMMDD}.csv
      * 각 종목의 현재가, 매수 수량, 투자 금액 포함
      * 동일 가중 (equal-weight) 포트폴리오
      * 모든 팩터 스코어 (Quality, Growth, Value, Momentum) 포함
    - portfolio_scores_{YYYYMMDD}.csv (전체 종목 스코어)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import time
import os
import pickle
import sys
warnings.filterwarnings('ignore')

# ============================================================
# 설정
# ============================================================
CONFIG = {
    'TOP_N': 20,                    # 포트폴리오 종목 수
    'MAX_SECTOR_WEIGHT': 0.30,      # 섹터 집중 제한
    'MIN_MARKET_CAP': 300e6,        # 최소 시가총액 $300M
    'MAX_MARKET_CAP': 10e9,         # 최대 시가총액 $10B
    'MIN_AVG_VOLUME': 100000,       # 최소 평균 거래량
    'BATCH_SIZE': 50,               # API 호출 배치 크기
    'SLEEP_TIME': 1,                # API 호출 간 대기 시간
    'SAMPLE_SIZE': None,            # None = 전체, 숫자 = 샘플링
    'CACHE_FILE': 'stock_data_cache.pkl',  # 캐시 파일명
    'CACHE_DAYS': 7,                # 캐시 유효 기간 (일)
    'FORCE_REFRESH': False,         # True = 강제 새로고침
    'INITIAL_BALANCE': 100000,      # 초기 투자 금액 ($)
}


def save_cache(data, cache_file):
    """데이터를 캐시 파일로 저장"""
    cache_data = {
        'timestamp': datetime.now(),
        'data': data
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"캐시 저장: {cache_file} ({len(data)}개 종목)")


def load_cache(cache_file, max_age_days):
    """캐시 파일 로드 (유효기간 확인)"""
    if not os.path.exists(cache_file):
        print(f"캐시 파일 없음: {cache_file}")
        return None

    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        cache_time = cache_data['timestamp']
        age = datetime.now() - cache_time

        if age.days > max_age_days:
            print(f"캐시 만료: {age.days}일 경과 (최대 {max_age_days}일)")
            return None

        print(f"캐시 로드: {cache_file} ({len(cache_data['data'])}개 종목, {age.days}일 전 데이터)")
        return cache_data['data']

    except Exception as e:
        print(f"캐시 로드 실패: {e}")
        return None


def load_russell2000_tickers(filepath='russell2000_tickers.csv'):
    """Russell 2000 티커 목록 로드"""
    df = pd.read_csv(filepath)
    print(f"Russell 2000 종목 로드: {len(df)}개")
    return df


def get_stock_info_batch(tickers, batch_size=50):
    """
    현재 시점의 주가 및 재무 데이터 수집 (모멘텀 계산을 위한 가격 이력 포함)
    """
    print(f"\n데이터 수집 시작 ({len(tickers)}개 종목)...")

    all_data = {}
    failed_tickers = []

    # 배치로 나누기
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

    # 모멘텀 계산을 위한 가격 이력 (1년치)
    lookback_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    current_date = datetime.now().strftime('%Y-%m-%d')

    for batch_idx, batch in enumerate(batches):
        print(f"  배치 {batch_idx+1}/{len(batches)} 처리 중... ({len(all_data)} 성공, {len(failed_tickers)} 실패)")

        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)

                # 재무 데이터
                info = stock.info

                # 기본 검증
                if not info or 'marketCap' not in info:
                    failed_tickers.append(ticker)
                    continue

                # 가격 이력 (모멘텀 계산용 - 1년치)
                try:
                    hist = stock.history(start=lookback_date, end=current_date)
                    if hist.empty or len(hist) < 50:
                        # 가격 데이터 없어도 재무 데이터는 저장 (모멘텀 없이 진행)
                        all_data[ticker] = {'info': info, 'price': None}
                    else:
                        all_data[ticker] = {'info': info, 'price': hist}
                except:
                    # 가격 데이터 실패해도 재무 데이터는 저장
                    all_data[ticker] = {'info': info, 'price': None}

            except Exception as e:
                failed_tickers.append(ticker)
                continue

        # API 제한 방지
        if batch_idx < len(batches) - 1:
            time.sleep(CONFIG['SLEEP_TIME'])

    print(f"데이터 수집 완료: {len(all_data)}개 성공, {len(failed_tickers)}개 실패")
    return all_data, failed_tickers


def calculate_factors(stock_data, sector_info):
    """각 종목별 퀄리티, 성장, 밸류, 모멘텀 팩터 계산"""

    print("\n팩터 계산 중...")
    factors_list = []

    for ticker, data in stock_data.items():
        try:
            # 데이터 구조 처리 (dict with 'info' and 'price' keys)
            if isinstance(data, dict) and 'info' in data:
                info = data['info']
                price_data = data.get('price')
            else:
                # 이전 캐시 호환성 (info만 있는 경우)
                info = data
                price_data = None

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
            momentum_12_1 = np.nan
            volatility = np.nan

            if price_data is not None and not price_data.empty:
                try:
                    # 12-1 month momentum (same as backtest)
                    if len(price_data) >= 252:
                        price_12m_ago = price_data['Close'].iloc[-252]
                        price_1m_ago = price_data['Close'].iloc[-21]
                        price_now = price_data['Close'].iloc[-1]

                        mom_12m = (price_now / price_12m_ago - 1) if price_12m_ago > 0 else np.nan
                        mom_1m = (price_now / price_1m_ago - 1) if price_1m_ago > 0 else np.nan
                        momentum_12_1 = mom_12m - mom_1m
                    elif len(price_data) >= 63:
                        # Fallback: 3-month momentum
                        price_3m_ago = price_data['Close'].iloc[-63]
                        price_now = price_data['Close'].iloc[-1]
                        momentum_12_1 = (price_now / price_3m_ago - 1) if price_3m_ago > 0 else np.nan

                    # Volatility
                    daily_returns = price_data['Close'].pct_change().dropna()
                    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 20 else np.nan
                except:
                    pass

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
    """종합 스코어 계산 (백테스트와 동일한 전략)"""

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


def select_portfolio(factors_df, top_n=20, max_sector_weight=0.30):
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


def get_current_prices(tickers):
    """
    선정된 종목의 현재가 조회
    """
    print(f"\n현재가 조회 중 ({len(tickers)}개 종목)...")
    prices = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # 최근 1일 데이터로 현재가 가져오기
            hist = stock.history(period='1d')
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prices[ticker] = current_price
            else:
                # fallback: info에서 가져오기
                info = stock.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if current_price:
                    prices[ticker] = current_price
                else:
                    print(f"  {ticker}: 가격 정보 없음")
                    prices[ticker] = None
        except Exception as e:
            print(f"  {ticker}: 오류 ({e})")
            prices[ticker] = None

    successful = sum(1 for p in prices.values() if p is not None)
    print(f"현재가 조회 완료: {successful}/{len(tickers)}개 성공")

    return prices


def generate_portfolio_report(factors_df, top_n=20, max_sector_weight=0.30, initial_balance=100000):
    """
    포트폴리오 리포트 생성 (투자금액 및 매수 수량 포함)
    """
    print("\n" + "=" * 70)
    print(f"포트폴리오 생성 (as of {datetime.now().strftime('%Y-%m-%d')})")
    print("=" * 70)
    print(f"초기 투자 금액: ${initial_balance:,.2f}")

    # 포트폴리오 선정
    selected_tickers = select_portfolio(factors_df, top_n=top_n, max_sector_weight=max_sector_weight)

    # 현재가 조회
    current_prices = get_current_prices(selected_tickers)

    # 선정된 종목의 상세 정보
    portfolio_df = factors_df[factors_df['ticker'].isin(selected_tickers)].copy()
    portfolio_df = portfolio_df.sort_values('composite_score', ascending=False)

    # Equal weight 계산
    portfolio_df['Weight'] = 1.0 / len(portfolio_df)
    position_size = initial_balance / len(portfolio_df)

    # 현재가 및 매수 수량 추가
    portfolio_df['Current_Price'] = portfolio_df['ticker'].map(current_prices)
    portfolio_df['Position_Size'] = position_size
    portfolio_df['Shares_to_Buy'] = (portfolio_df['Position_Size'] / portfolio_df['Current_Price']).apply(lambda x: int(x) if pd.notna(x) else 0)
    portfolio_df['Actual_Amount'] = portfolio_df['Shares_to_Buy'] * portfolio_df['Current_Price']

    # 출력용 DataFrame
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

    # 날짜 기반 파일명
    date_str = datetime.now().strftime('%Y%m%d')
    portfolio_file = f'current_portfolio_{date_str}.csv'
    output_df.to_csv(portfolio_file, index=False)

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

    print(f"\n저장된 파일:")
    print(f"  - {portfolio_file}")

    return output_df, portfolio_file


def main():
    # 커맨드 라인 인자 처리
    if '--refresh' in sys.argv or '-r' in sys.argv:
        CONFIG['FORCE_REFRESH'] = True
        print("강제 새로고침 모드")

    print("=" * 70)
    print("     Russell 2000 포트폴리오 생성기")
    print("=" * 70)

    # 1. 티커 목록 로드
    ticker_df = load_russell2000_tickers('russell2000_tickers.csv')
    tickers = ticker_df['Ticker'].tolist()

    # 섹터 정보 딕셔너리
    sector_info = dict(zip(ticker_df['Ticker'], ticker_df['Sector']))

    # 2. 샘플링 (옵션)
    if CONFIG['SAMPLE_SIZE'] is not None:
        sample_size = min(CONFIG['SAMPLE_SIZE'], len(tickers))
        np.random.seed(42)
        sampled_tickers = np.random.choice(tickers, sample_size, replace=False).tolist()
        print(f"\n샘플 크기: {sample_size}개 (전체 {len(tickers)}개 중)")
    else:
        sampled_tickers = tickers
        print(f"\n전체 분석: {len(tickers)}개 종목")

    # 3. 캐시 확인 또는 데이터 수집
    stock_data = None

    if not CONFIG['FORCE_REFRESH']:
        stock_data = load_cache(CONFIG['CACHE_FILE'], CONFIG['CACHE_DAYS'])

    if stock_data is None:
        print("\n새로운 데이터 다운로드 중...")
        stock_data, failed = get_stock_info_batch(
            sampled_tickers,
            batch_size=CONFIG['BATCH_SIZE']
        )
        # 캐시 저장
        save_cache(stock_data, CONFIG['CACHE_FILE'])
    else:
        print("캐시된 데이터 사용")

    if len(stock_data) < 50:
        print("충분한 데이터를 수집하지 못했습니다.")
        return

    # 4. 팩터 계산
    factors_df = calculate_factors(stock_data, sector_info)
    factors_df = calculate_composite_score(factors_df)

    # 전체 스코어 저장
    date_str = datetime.now().strftime('%Y%m%d')
    factors_df.to_csv(f'portfolio_scores_{date_str}.csv', index=False)
    print(f"\n전체 종목 스코어 저장: portfolio_scores_{date_str}.csv")

    # 5. 포트폴리오 생성
    portfolio_df, portfolio_file = generate_portfolio_report(
        factors_df,
        top_n=CONFIG['TOP_N'],
        max_sector_weight=CONFIG['MAX_SECTOR_WEIGHT'],
        initial_balance=CONFIG['INITIAL_BALANCE']
    )

    # 6. 상위 종목 출력
    print("\n" + "=" * 70)
    print("상위 30개 종목 (종합 스코어 기준)")
    print("=" * 70)

    top30 = factors_df.nlargest(30, 'composite_score')[
        ['ticker', 'sector', 'market_cap', 'composite_score', 'quality_score', 'growth_score', 'momentum_score']
    ].copy()

    print(f"\n{'Ticker':<8} {'Sector':<20} {'MCap(B)':>8} {'Score':>7} {'Quality':>8} {'Growth':>8} {'Momentum':>9}")
    print("-" * 80)

    for _, row in top30.iterrows():
        mcap = row['market_cap'] / 1e9
        print(f"{row['ticker']:<8} {row['sector'][:20]:<20} {mcap:>7.2f} "
              f"{row['composite_score']:>7.3f} {row['quality_score']:>7.2f} {row['growth_score']:>7.2f} {row['momentum_score']:>8.2f}")

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
