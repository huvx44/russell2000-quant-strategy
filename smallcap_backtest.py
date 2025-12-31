"""
소형주 퀄리티+성장 퀀트 전략 백테스트
=====================================
- 유니버스: Russell 2000 구성종목 (소형주)
- 팩터: 퀄리티 + 성장 + 밸류 안전장치
- 리밸런싱: 분기별
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Russell 2000 대표 종목 샘플 (실제로는 전체 목록 필요)
# ============================================================
# 참고: 전체 Russell 2000 목록은 유료 데이터 필요
# 여기서는 소형주 대표 샘플로 시연

SAMPLE_SMALL_CAPS = [
    # 테크/소프트웨어
    'SMCI', 'AEHR', 'CAMT', 'AMBA', 'LSCC', 'AEIS', 'VICR', 'POWI', 'DIOD', 'SLAB',
    # 헬스케어
    'MEDP', 'HALO', 'ITCI', 'KRYS', 'RVMD', 'PTCT', 'RARE', 'FOLD', 'TGTX', 'ACAD',
    # 산업재
    'ATKR', 'ESAB', 'SPXC', 'RBC', 'PRIM', 'DY', 'ROCK', 'GFF', 'NPO', 'POWL',
    # 소비재
    'BOOT', 'SHAK', 'WING', 'PLNT', 'FIZZ', 'LULU', 'DECK', 'CROX', 'SKX', 'FOXF',
    # 에너지/소재
    'MTDR', 'SM', 'PTEN', 'HP', 'RES', 'CLB', 'CIVI', 'OVV', 'NOG', 'GPOR'
]

# ============================================================
# 2. 데이터 수집 함수
# ============================================================
def get_stock_data(tickers, start_date, end_date):
    """주가 및 재무 데이터 수집"""
    print(f"데이터 수집 중... ({len(tickers)}개 종목)")
    
    all_data = {}
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            
            # 주가 데이터
            hist = stock.history(start=start_date, end=end_date)
            if hist.empty:
                failed_tickers.append(ticker)
                continue
            
            # 재무 데이터
            info = stock.info
            financials = stock.financials
            balance = stock.balance_sheet
            cashflow = stock.cashflow
            
            all_data[ticker] = {
                'price': hist,
                'info': info,
                'financials': financials,
                'balance': balance,
                'cashflow': cashflow
            }
            
            if (i + 1) % 10 == 0:
                print(f"  진행: {i+1}/{len(tickers)}")
                
        except Exception as e:
            failed_tickers.append(ticker)
            continue
    
    print(f"완료: {len(all_data)}개 성공, {len(failed_tickers)}개 실패")
    return all_data, failed_tickers


# ============================================================
# 3. 팩터 계산 함수
# ============================================================
def calculate_factors(stock_data):
    """각 종목별 퀄리티, 성장, 밸류 팩터 계산"""
    
    factors_list = []
    
    for ticker, data in stock_data.items():
        try:
            info = data['info']
            
            # 기본 정보
            market_cap = info.get('marketCap', 0)
            if market_cap == 0 or market_cap > 10e9:  # $10B 초과 제외
                continue
            
            # ---- 퀄리티 팩터 ----
            roe = info.get('returnOnEquity', np.nan)
            roa = info.get('returnOnAssets', np.nan)
            gross_margin = info.get('grossMargins', np.nan)
            debt_equity = info.get('debtToEquity', np.nan)
            current_ratio = info.get('currentRatio', np.nan)
            
            # ---- 성장 팩터 ----
            revenue_growth = info.get('revenueGrowth', np.nan)
            earnings_growth = info.get('earningsGrowth', np.nan)
            
            # ---- 밸류 팩터 ----
            pe_ratio = info.get('trailingPE', np.nan)
            pb_ratio = info.get('priceToBook', np.nan)
            peg_ratio = info.get('pegRatio', np.nan)
            ev_ebitda = info.get('enterpriseToEbitda', np.nan)
            
            # ---- 모멘텀 ----
            price_data = data['price']
            if len(price_data) >= 252:
                mom_12m = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[-252] - 1)
                mom_1m = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[-21] - 1)
                momentum = mom_12m - mom_1m  # 12-1 모멘텀
            else:
                momentum = np.nan
            
            factors_list.append({
                'ticker': ticker,
                'market_cap': market_cap,
                'roe': roe,
                'roa': roa,
                'gross_margin': gross_margin,
                'debt_equity': debt_equity,
                'current_ratio': current_ratio,
                'revenue_growth': revenue_growth,
                'earnings_growth': earnings_growth,
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'peg_ratio': peg_ratio,
                'ev_ebitda': ev_ebitda,
                'momentum': momentum
            })
            
        except Exception as e:
            continue
    
    return pd.DataFrame(factors_list)


def calculate_composite_score(df):
    """종합 스코어 계산"""
    
    df = df.copy()

    # Z-score 정규화 함수
    def zscore(series):
        std = series.std()
        # Handle zero or NaN standard deviation
        if pd.isna(std) or std == 0:
            return pd.Series(0, index=series.index)
        return (series - series.mean()) / std

    def safe_fill(series, default=0):
        """Safely fill NaN with median, or default if median is NaN"""
        median_val = series.median()
        return series.fillna(median_val if pd.notna(median_val) else default)

    # 퀄리티 스코어 (높을수록 좋음)
    df['roe_z'] = zscore(safe_fill(df['roe']))
    df['roa_z'] = zscore(safe_fill(df['roa']))
    df['gross_margin_z'] = zscore(safe_fill(df['gross_margin']))
    df['debt_equity_z'] = -zscore(safe_fill(df['debt_equity']))  # 낮을수록 좋음
    
    df['quality_score'] = (
        df['roe_z'] * 0.30 +
        df['roa_z'] * 0.25 +
        df['gross_margin_z'] * 0.25 +
        df['debt_equity_z'] * 0.20
    )
    
    # 성장 스코어
    df['rev_growth_z'] = zscore(df['revenue_growth'].fillna(0))
    df['earn_growth_z'] = zscore(df['earnings_growth'].fillna(0))
    
    df['growth_score'] = (
        df['rev_growth_z'] * 0.50 +
        df['earn_growth_z'] * 0.50
    )
    
    # 밸류 스코어 (낮을수록 좋음 -> 역수 처리)
    # Convert to numeric first to handle any string values
    df['pe_z'] = -zscore(pd.to_numeric(df['pe_ratio'], errors='coerce').clip(upper=100).fillna(50))
    df['peg_z'] = -zscore(pd.to_numeric(df['peg_ratio'], errors='coerce').clip(upper=5).fillna(2))
    
    df['value_score'] = (
        df['pe_z'] * 0.50 +
        df['peg_z'] * 0.50
    )
    
    # 모멘텀 스코어
    df['momentum_z'] = zscore(df['momentum'].fillna(0))
    
    # 종합 스코어
    df['composite_score'] = (
        df['quality_score'] * 0.35 +
        df['growth_score'] * 0.35 +
        df['value_score'] * 0.15 +
        df['momentum_z'] * 0.15
    )
    
    return df


# ============================================================
# 4. 백테스트 엔진
# ============================================================
def run_backtest(stock_data, factors_df, start_date, end_date, 
                 top_n=20, rebalance_freq='Q'):
    """
    백테스트 실행
    - top_n: 포트폴리오 종목 수
    - rebalance_freq: 리밸런싱 주기 ('M': 월별, 'Q': 분기별)
    """
    
    print("\n백테스트 실행 중...")
    
    # 모든 종목의 일별 수익률 계산
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

    # 리밸런싱 날짜 생성
    if rebalance_freq == 'Q':
        rebal_dates = pd.date_range(start=start_date, end=end_date, freq='QS')
    else:
        rebal_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # 포트폴리오 수익률 계산
    portfolio_returns = []
    benchmark_returns = []
    holdings_history = []

    prev_holdings_dict = {}
    portfolio_value = 1.0

    for i, date in enumerate(rebal_dates):
        # 다음 리밸런싱 날짜까지의 기간
        if i < len(rebal_dates) - 1:
            next_date = rebal_dates[i + 1]
        else:
            next_date = pd.Timestamp(end_date)

        # 해당 기간의 수익률 데이터
        period_returns = returns_df[(returns_df.index >= date) & (returns_df.index < next_date)]

        if period_returns.empty:
            continue

        # 포트폴리오 리밸런싱 (상위 N개 선택)
        valid_tickers = [t for t in factors_df.nlargest(top_n, 'composite_score')['ticker']
                        if t in period_returns.columns]

        if len(valid_tickers) < 5:
            continue

        current_holdings = valid_tickers[:top_n]

        # Get entry prices
        entry_prices = {}
        for ticker in current_holdings:
            ticker_data = stock_data.get(ticker)
            if ticker_data and 'price' in ticker_data:
                price_data = ticker_data['price']['Close']
                if price_data.index.tz is not None:
                    price_data.index = price_data.index.tz_localize(None)
                available_prices = price_data[price_data.index >= date]
                if len(available_prices) > 0:
                    entry_prices[ticker] = available_prices.iloc[0]

        # Calculate position info
        position_value = portfolio_value / len(current_holdings)
        holdings_with_prices = []

        for ticker in current_holdings:
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

        # Calculate exit info for previous holdings
        if prev_holdings_dict:
            for holding_info in prev_holdings_dict:
                ticker = holding_info['ticker']
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

        # Save previous holdings to history
        if prev_holdings_dict:
            for holding_info in prev_holdings_dict:
                holdings_history.append(holding_info.copy())

        prev_holdings_dict = holdings_with_prices

        # 동일가중 포트폴리오 수익률
        port_ret = period_returns[current_holdings].mean(axis=1)
        portfolio_returns.append(port_ret)

        # Update portfolio value
        period_total_return = (1 + port_ret).prod() - 1
        portfolio_value = portfolio_value * (1 + period_total_return)

        # 벤치마크 (전체 유니버스 동일가중)
        bench_ret = period_returns.mean(axis=1)
        benchmark_returns.append(bench_ret)

    # Handle final holdings
    if prev_holdings_dict:
        final_date = pd.Timestamp(end_date)
        for holding_info in prev_holdings_dict:
            ticker = holding_info['ticker']
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

    # 결과 합치기
    portfolio_series = pd.concat(portfolio_returns)
    benchmark_series = pd.concat(benchmark_returns)

    return portfolio_series, benchmark_series, holdings_history


# ============================================================
# 5. 성과 분석 함수
# ============================================================
def calculate_metrics(returns, risk_free_rate=0.04):
    """성과 지표 계산"""
    
    # 연환산 수익률
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # 변동성
    annual_volatility = returns.std() * np.sqrt(252)
    
    # 샤프 비율
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
    
    # 최대 낙폭
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()
    
    # 승률
    win_rate = (returns > 0).sum() / len(returns)
    
    # 소르티노 비율
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    # 칼마 비율
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Annual Return': f"{annual_return:.2%}",
        'Annual Volatility': f"{annual_volatility:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Sortino Ratio': f"{sortino:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Calmar Ratio': f"{calmar:.2f}",
        'Win Rate': f"{win_rate:.2%}"
    }


def plot_results(portfolio_returns, benchmark_returns, save_path='backtest_results.png'):
    """결과 시각화"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 누적 수익률
    port_cumulative = (1 + portfolio_returns).cumprod()
    bench_cumulative = (1 + benchmark_returns).cumprod()
    
    axes[0].plot(port_cumulative.index, port_cumulative.values, 
                 label='Strategy', linewidth=2, color='#2E86AB')
    axes[0].plot(bench_cumulative.index, bench_cumulative.values, 
                 label='Benchmark (Equal Weight)', linewidth=2, color='#A23B72', alpha=0.7)
    axes[0].set_title('Cumulative Returns: Small-Cap Quality + Growth Strategy', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 2. 드로우다운
    rolling_max = port_cumulative.expanding().max()
    drawdown = port_cumulative / rolling_max - 1
    
    axes[1].fill_between(drawdown.index, drawdown.values, 0, 
                         color='#E74C3C', alpha=0.5, label='Drawdown')
    axes[1].set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 3. 롤링 12개월 수익률
    rolling_return = portfolio_returns.rolling(252).apply(lambda x: (1+x).prod() - 1)
    bench_rolling = benchmark_returns.rolling(252).apply(lambda x: (1+x).prod() - 1)
    
    axes[2].plot(rolling_return.index, rolling_return.values, 
                 label='Strategy', linewidth=2, color='#2E86AB')
    axes[2].plot(bench_rolling.index, bench_rolling.values, 
                 label='Benchmark', linewidth=2, color='#A23B72', alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Rolling 12-Month Returns', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('12M Return')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n차트 저장: {save_path}")


def generate_current_portfolio(factors_df, top_n=15, output_dir='.'):
    """
    현재 시점의 추천 포트폴리오 생성
    """
    print("\n" + "=" * 60)
    print(f"현재 포트폴리오 추천 (as of {datetime.now().strftime('%Y-%m-%d')})")
    print("=" * 60)

    # 상위 N개 선정
    portfolio_df = factors_df.nlargest(top_n, 'composite_score').copy()

    # Equal weight
    portfolio_df['Weight'] = 1.0 / len(portfolio_df)

    # 출력용 DataFrame 구성
    output_df = portfolio_df[[
        'ticker', 'market_cap', 'composite_score',
        'quality_score', 'growth_score', 'value_score',
        'roe', 'revenue_growth', 'pe_ratio', 'Weight'
    ]].copy()

    output_df.columns = [
        'Ticker', 'Market_Cap', 'Composite_Score',
        'Quality_Score', 'Growth_Score', 'Value_Score',
        'ROE', 'Revenue_Growth', 'PE_Ratio', 'Weight'
    ]

    # 포맷팅
    output_df['Market_Cap_B'] = (output_df['Market_Cap'] / 1e9).round(2)
    output_df['Weight_%'] = (output_df['Weight'] * 100).round(2)
    output_df = output_df.drop(['Market_Cap', 'Weight'], axis=1)

    # CSV 저장
    output_df.to_csv(f'{output_dir}/current_portfolio.csv', index=False)

    # 콘솔 출력
    print(f"\n포트폴리오 구성: {len(output_df)}개 종목 (동일가중)")
    print(f"\n{'Ticker':<8} {'MCap(B)':>8} {'Weight':>7} {'Score':>7} {'Quality':>8} {'Growth':>8}")
    print("-" * 60)

    for _, row in output_df.iterrows():
        print(f"{row['Ticker']:<8} {row['Market_Cap_B']:>7.2f} {row['Weight_%']:>6.1f}% "
              f"{row['Composite_Score']:>7.3f} {row['Quality_Score']:>7.2f} {row['Growth_Score']:>7.2f}")

    # 평균 팩터 스코어
    print("\n평균 팩터 스코어:")
    print(f"  Quality: {output_df['Quality_Score'].mean():>6.2f}")
    print(f"  Growth:  {output_df['Growth_Score'].mean():>6.2f}")
    print(f"  Value:   {output_df['Value_Score'].mean():>6.2f}")

    print(f"\n포트폴리오 저장: current_portfolio.csv")

    return output_df


def generate_quarterly_summary(details_df):
    """
    분기별 성과 요약 생성
    """
    details_df = details_df.copy()
    details_df['Exit_Date'] = pd.to_datetime(details_df['Exit_Date'], errors='coerce')
    details_df['Quarter'] = details_df['Exit_Date'].dt.to_period('Q')

    closed_positions = details_df[details_df['Quarter'].notna()].copy()

    if len(closed_positions) == 0:
        return pd.DataFrame()

    quarterly_groups = closed_positions.groupby('Quarter')

    summary_rows = []
    for quarter, group in quarterly_groups:
        n_trades = len(group)
        total_profit = group['Profit'].sum()
        total_profit_pct = group['Profit_Pct'].mean()

        winners = group[group['Profit_Pct'] > 0]
        losers = group[group['Profit_Pct'] <= 0]
        win_rate = len(winners) / n_trades if n_trades > 0 else 0

        avg_win = winners['Profit_Pct'].mean() if len(winners) > 0 else 0
        avg_loss = losers['Profit_Pct'].mean() if len(losers) > 0 else 0

        best_trade = group['Profit_Pct'].max()
        worst_trade = group['Profit_Pct'].min()
        best_ticker = group.loc[group['Profit_Pct'].idxmax(), 'Ticker'] if n_trades > 0 else ''
        worst_ticker = group.loc[group['Profit_Pct'].idxmin(), 'Ticker'] if n_trades > 0 else ''

        avg_position_value = group['Position_Value'].mean()
        total_capital_deployed = group['Position_Value'].sum()

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
            'Total_Capital_Deployed': total_capital_deployed
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df['Cumulative_Profit'] = summary_df['Total_Profit'].cumsum()
    summary_df['Cumulative_Trades'] = summary_df['N_Trades'].cumsum()

    return summary_df


def save_rebalancing_details(holdings_history, factors_df, output_dir='.'):
    """
    리밸런싱 상세 내역 저장

    세 개의 파일 생성:
    1. rebalancing_history.csv: 각 리밸런싱 날짜의 보유 종목 목록 (요약)
    2. rebalancing_details.csv: 각 보유 종목의 상세 정보 (포지션, 가격, 수익 등)
    3. quarterly_summary.csv: 분기별 성과 요약
    """

    # 1. 리밸런싱 이력 (요약)
    history_rows = []
    holdings_by_date = {}

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

    # 2. 리밸런싱 상세 내역
    details_rows = []

    for entry in holdings_history:
        ticker = entry['ticker']
        stock_info = factors_df[factors_df['ticker'] == ticker]
        if not stock_info.empty:
            stock_info = stock_info.iloc[0]
        else:
            stock_info = {}

        details_rows.append({
            'Entry_Date': entry.get('entry_date', pd.NaT).strftime('%Y-%m-%d') if pd.notna(entry.get('entry_date')) else '',
            'Exit_Date': entry.get('exit_date', pd.NaT).strftime('%Y-%m-%d') if pd.notna(entry.get('exit_date')) else '',
            'Ticker': ticker,
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
        quarterly_summary = generate_quarterly_summary(details_df)
        quarterly_summary.to_csv(f'{output_dir}/quarterly_summary.csv', index=False)

    print(f"\n리밸런싱 내역 저장:")
    print(f"  - 총 {len(holdings_history)}개 포지션 추적")
    if len(details_df) > 0:
        avg_profit = details_df['Profit_Pct'].mean()
        win_rate = (details_df['Profit_Pct'] > 0).sum() / len(details_df)
        print(f"  - 평균 수익률: {avg_profit:.2f}%")
        print(f"  - 승률: {win_rate:.1%}")


# ============================================================
# 6. 메인 실행
# ============================================================
def main():
    print("=" * 60)
    print("소형주 퀄리티+성장 퀀트 전략 백테스트")
    print("=" * 60)

    # 결과 폴더 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n결과 저장 폴더: {results_dir}/")

    # 백테스트 기간 설정
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    
    print(f"\n백테스트 기간: {start_date} ~ {end_date}")
    print(f"유니버스: 소형주 샘플 {len(SAMPLE_SMALL_CAPS)}개")
    
    # 1. 데이터 수집
    stock_data, failed = get_stock_data(SAMPLE_SMALL_CAPS, start_date, end_date)
    
    if len(stock_data) < 10:
        print("충분한 데이터를 수집하지 못했습니다.")
        return
    
    # 2. 팩터 계산
    print("\n팩터 계산 중...")
    factors_df = calculate_factors(stock_data)
    print(f"팩터 계산 완료: {len(factors_df)}개 종목")
    
    # 3. 종합 스코어 계산
    factors_df = calculate_composite_score(factors_df)
    
    # 상위 종목 출력
    print("\n" + "=" * 60)
    print("상위 10개 종목 (종합 스코어 기준)")
    print("=" * 60)
    top10 = factors_df.nlargest(10, 'composite_score')[
        ['ticker', 'market_cap', 'roe', 'revenue_growth', 'pe_ratio', 'composite_score']
    ].copy()
    top10['market_cap'] = (top10['market_cap'] / 1e9).round(2).astype(str) + 'B'
    top10['roe'] = (top10['roe'] * 100).round(1).astype(str) + '%'
    top10['revenue_growth'] = (top10['revenue_growth'] * 100).round(1).astype(str) + '%'
    top10['pe_ratio'] = top10['pe_ratio'].round(1)
    top10['composite_score'] = top10['composite_score'].round(3)
    print(top10.to_string(index=False))

    # 3a. 현재 포트폴리오 생성
    current_portfolio = generate_current_portfolio(factors_df, top_n=15, output_dir=results_dir)

    # 4. 백테스트 실행
    portfolio_returns, benchmark_returns, holdings = run_backtest(
        stock_data, factors_df, start_date, end_date,
        top_n=15, rebalance_freq='Q'
    )

    # 5. 성과 분석
    print("\n" + "=" * 60)
    print("백테스트 결과")
    print("=" * 60)

    print("\n[전략 성과]")
    strategy_metrics = calculate_metrics(portfolio_returns)
    for key, value in strategy_metrics.items():
        print(f"  {key}: {value}")

    print("\n[벤치마크 성과 (동일가중)]")
    benchmark_metrics = calculate_metrics(benchmark_returns)
    for key, value in benchmark_metrics.items():
        print(f"  {key}: {value}")

    # 6. 시각화
    try:
        plot_results(portfolio_returns, benchmark_returns, f'{results_dir}/backtest_results.png')
    except Exception as e:
        print(f"\n차트 생성 실패: {e}")

    # 7. 상세 결과 저장
    results_df = pd.DataFrame({
        'Strategy': portfolio_returns,
        'Benchmark': benchmark_returns
    })
    results_df.to_csv(f'{results_dir}/backtest_daily_returns.csv')

    # 팩터 데이터 저장
    factors_df.to_csv(f'{results_dir}/factor_scores.csv', index=False)

    # 8. 리밸런싱 상세 내역 저장
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

    print("\n" + "=" * 60)
    print("백테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
