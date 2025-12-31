# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a quantitative trading research repository focused on small-cap stock strategies. The codebase implements backtesting frameworks for quality + growth factor strategies on Russell 2000 constituents.

## Running the Code

### Dependencies
```bash
pip install yfinance pandas numpy matplotlib
```

### Execute Backtests
```bash
# Run the small-cap sample backtest (50 stocks)
python smallcap_backtest.py

# Run the full Russell 2000 backtest (uses russell2000_tickers.csv)
python russell2000_backtest.py           # Uses cache if available
python russell2000_backtest.py --refresh # Force refresh data
```

**Smart caching system:**
- Fundamental data cached in `stock_data_cache.pkl` (shared with portfolio generator)
- Price history cached in `price_history_cache.pkl` (backtest-specific)
- Cache is valid for 7 days (configurable via `CACHE_DAYS`)
- First run: 30-60 minutes (downloads all data)
- Subsequent runs: Few seconds to start analysis (uses cache)
- Avoids duplicate downloads - both scripts share fundamental data
- Price cache is date-range specific (changing START_DATE/END_DATE triggers re-download)

### Generate Portfolio Only (Fast)

If you only need current portfolio recommendations without running the full backtest:

```bash
# Generate portfolio using cached data (few seconds if cache exists)
python generate_portfolio.py

# Force refresh data from yfinance (5-10 minutes)
python generate_portfolio.py --refresh

# Or use the update utility
python update_cache.py
```

This standalone tool:
- **Uses smart caching**: Downloaded data is cached for 7 days
  - First run: 5-10 minutes (downloads all data)
  - Subsequent runs: Few seconds (uses cache)
  - Cache file: `stock_data_cache.pkl`
- Skips historical price data collection (faster than backtest)
- Calculates current fundamental factors only
- Applies same scoring and selection logic
- Outputs: `current_portfolio_YYYYMMDD.csv` and `portfolio_scores_YYYYMMDD.csv`

Configure in the script's CONFIG dictionary:
- `TOP_N`: Portfolio size (default: 20)
- `INITIAL_BALANCE`: Initial investment amount (default: $100,000)
  - Script calculates shares to buy for each stock
  - Equal-weight allocation: `position_size = INITIAL_BALANCE / TOP_N`
  - Outputs actual shares and dollar amounts for each position
- `CACHE_DAYS`: Cache validity period (default: 7 days)
- `SAMPLE_SIZE`: Set to a number for testing, None for full universe
- Sector/market cap filters same as backtest

### Which Tool to Use?

**Use `generate_portfolio.py` when:**
- You just need updated portfolio holdings
- You want to see current top-ranked stocks
- You're running this frequently (e.g., weekly/monthly rebalancing checks)
- You want faster results

**Use `russell2000_backtest.py` when:**
- You want to evaluate strategy performance over time
- You need to analyze historical trade performance
- You're testing different factor weights or parameters
- You want comprehensive reports (quarterly summary, trade analysis, etc.)

### Output Location

Backtest outputs are saved in timestamped folders: `results_YYYYMMDD_HHMMSS/`

Portfolio generator outputs go to the current directory with date suffixes.

Example:
```
results_20251231_143052/
├── current_portfolio.csv
├── backtest_results.png
├── quarterly_summary.png
├── backtest_daily_returns.csv
├── factor_scores.csv
├── rebalancing_history.csv
├── rebalancing_details.csv
└── quarterly_summary.csv
```

This keeps your workspace clean and allows you to compare multiple backtest runs.

## Architecture

### Two Backtest Implementations

1. **smallcap_backtest.py**: Simplified prototype
   - Hardcoded sample of 50 representative small-cap stocks
   - Basic factor implementation
   - Suitable for quick experimentation

2. **russell2000_backtest.py**: Production-ready version
   - Uses complete Russell 2000 constituent list from CSV (1,959 stocks)
   - Advanced features:
     - Batch data collection with API rate limiting
     - Sector concentration limits
     - Portfolio turnover tracking
     - Winsorization for outlier handling

### Shared Architecture Pattern

Both scripts follow the same pipeline structure:

1. **Data Collection** (`get_stock_data*` functions)
   - Fetches historical prices via yfinance
   - Collects fundamental data from stock.info
   - Handles API failures gracefully

2. **Factor Calculation** (`calculate_factors`)
   - Computes raw factor values from fundamentals:
     - Quality: ROE, ROA, margins, debt/equity, current ratio
     - Growth: Revenue/earnings growth rates
     - Value: P/E, P/B, PEG, EV/EBITDA
     - Momentum: 12-1 month momentum (excludes most recent month)

3. **Score Normalization** (`calculate_composite_score`)
   - Z-score standardization of each factor
   - Factor weighting to create composite score:
     - Quality: 35%
     - Growth: 35%
     - Value: 15%
     - Momentum: 15%
   - Higher scores = better stocks

4. **Portfolio Construction** (`select_portfolio` in russell2000_backtest.py)
   - Ranks stocks by composite score
   - Applies sector concentration limits (max 30% per sector)
   - Equal-weight allocation across selected stocks

5. **Backtesting Engine** (`run_backtest`)
   - Quarterly rebalancing (configurable to monthly)
   - Equal-weight portfolio of top N stocks
   - Benchmark: Equal-weight of entire universe
   - Tracks holdings history and turnover

6. **Performance Analysis** (`calculate_metrics`)
   - Annual return, volatility
   - Risk-adjusted metrics: Sharpe, Sortino, Calmar ratios
   - Drawdown analysis
   - Win rate

### Key Configuration Parameters

In `generate_portfolio.py`, the CONFIG dictionary controls:
- `INITIAL_BALANCE`: Initial investment amount (default: $100,000)
  - Change this to match your actual capital
  - Script calculates exact shares to buy for each position
- `TOP_N`: Number of stocks in portfolio (default: 20)
- `MAX_SECTOR_WEIGHT`: Maximum allocation to any sector (default: 30%)
- `MIN_MARKET_CAP` / `MAX_MARKET_CAP`: Market cap filters (300M - 10B)
- `MIN_AVG_VOLUME`: Liquidity filter (100k shares/day)
- `SAMPLE_SIZE`: Set to a number for testing, None for full universe

In `russell2000_backtest.py`, the CONFIG dictionary controls:
- `TOP_N`: Number of stocks in portfolio (default: 20)
- `REBALANCE_FREQ`: 'Q' for quarterly, 'M' for monthly
- `MAX_SECTOR_WEIGHT`: Maximum allocation to any sector (default: 30%)
- `MIN_MARKET_CAP` / `MAX_MARKET_CAP`: Market cap filters (300M - 10B)
- `MIN_AVG_VOLUME`: Liquidity filter (100k shares/day)
- `BATCH_SIZE` / `SLEEP_TIME`: API rate limiting controls

### Data Files

- **russell2000_tickers.csv**: Russell 2000 constituent list with columns:
  - Ticker: Stock symbol
  - Name: Company name
  - Sector: Industry sector classification
  - Weight (%): Index weight
  - Market Value: Market capitalization
  - Price: Current price

- **russell2000_tickers.txt**: Plain text ticker list (alternative format)

- **stock_data_cache.pkl**: Cached fundamental data (SHARED by both scripts)
  - Automatically created on first run of either script
  - Contains stock info from yfinance (market cap, financials, ratios)
  - Shared between `generate_portfolio.py` and `russell2000_backtest.py`
  - Expires after 7 days (configurable via `CACHE_DAYS`)
  - Delete to force fresh download, or use `--refresh` flag
  - Significantly speeds up both scripts by avoiding duplicate downloads

- **price_history_cache.pkl**: Cached price history for backtest
  - Automatically created on first backtest run
  - Contains historical price data for the configured date range
  - Used only by `russell2000_backtest.py`
  - Expires after 7 days (configurable via `CACHE_DAYS`)
  - Changing `START_DATE` or `END_DATE` invalidates cache
  - Use `--refresh` flag or delete file to force re-download
  - Combined with fundamental cache, reduces 30-60 minute startup to seconds

### Output Files

Both scripts generate:
- **`current_portfolio.csv`**: Current recommended portfolio as of today's date
  - Ticker list with equal weights
  - All factor scores and key fundamentals for each holding
  - **`generate_portfolio.py` includes:**
    - `Current_Price`: Current market price per share
    - `Shares_to_Buy`: Number of shares to purchase (integer)
    - `Actual_Amount`: Dollar amount for this position
  - Sector distribution (russell2000_backtest.py only)
  - Average factor scores across the portfolio
  - **This is your actionable portfolio to trade today**
- `backtest_results.png`: Charts (cumulative returns, drawdown, monthly returns, sector distribution)
- `quarterly_summary.png`: Quarterly performance visualization including:
  - Quarterly average returns (bar chart with red/green colors)
  - Win rate trend over time (line chart showing consistency)
  - Trading activity and cumulative profit (dual-axis showing volume and P&L)
  - Average win vs loss comparison (side-by-side bars per quarter)
  - **Use this for visual quarterly performance analysis**
- `backtest_daily_returns.csv`: Daily strategy vs benchmark returns
- `factor_scores.csv`: All stocks with computed factors and composite scores
- `rebalancing_history.csv`: Summary of each rebalancing date with list of tickers held
- `rebalancing_details.csv`: Detailed position tracking for each holding including:
  - Entry/exit dates and prices
  - Position size (number of shares, position value)
  - Exit value and profit (both $ and %)
  - Sector and factor scores
  - Key fundamentals (ROE, revenue growth)
  - **Use this to analyze individual trade performance**
- `quarterly_summary.csv`: Quarterly performance aggregation including:
  - Number of trades, winners/losers, win rate
  - Total and average profit per quarter
  - Best/worst trades each quarter
  - Capital deployed and position sizing
  - Top sectors traded (russell2000_backtest.py only)
  - Cumulative metrics
  - **Use this for high-level performance trends**

## Using the Current Portfolio

The `current_portfolio.csv` file contains your recommended holdings as of today. This is generated using the most recent fundamental and price data.

### From Portfolio Generator (generate_portfolio.py)

The portfolio generator includes ready-to-execute buy orders with share counts:

```python
import pandas as pd

# Load portfolio from generate_portfolio.py
portfolio = pd.read_csv('current_portfolio_20251231.csv')

# View actionable buy orders
print("\nBuy Orders:")
print(portfolio[['Ticker', 'Current_Price', 'Shares_to_Buy', 'Actual_Amount']])

# Get shopping list
for _, row in portfolio.iterrows():
    print(f"BUY {int(row['Shares_to_Buy'])} shares of {row['Ticker']} @ ${row['Current_Price']:.2f}")

# Investment summary
total_invested = portfolio['Actual_Amount'].sum()
print(f"\nTotal to invest: ${total_invested:,.2f}")
print(f"Number of positions: {len(portfolio)}")
print(f"Average position size: ${total_invested/len(portfolio):,.2f}")
```

### From Backtest (russell2000_backtest.py)

The backtest output requires calculating shares from weights:

```python
import pandas as pd

# Load current portfolio from latest results folder
results_folder = 'results_20251231_143052'
portfolio = pd.read_csv(f'{results_folder}/current_portfolio.csv')

# Each stock gets equal weight
total_capital = 100000  # $100,000 example
position_size = total_capital * portfolio['Weight_%'].iloc[0] / 100

# View holdings
print(f"Number of positions: {len(portfolio)}")
print(f"Position size per stock: ${position_size:,.2f}")
print(f"\nTickers to buy:\n{portfolio['Ticker'].tolist()}")
```

The portfolio is:
- **Equal-weighted**: Each position gets 1/N of capital (e.g., 5% for 20 stocks)
- **Sector-constrained** (russell2000_backtest.py): Max 30% per sector to ensure diversification
- **Rebalanced quarterly**: Re-run the script every quarter to get updated holdings

## Analyzing Rebalancing Output

The rebalancing CSV files provide insights into portfolio composition over time:

### rebalancing_history.csv
Quick overview of what was held at each rebalance:
```python
import pandas as pd

# Use your actual results folder
results_folder = 'results_20251231_143052'
history = pd.read_csv(f'{results_folder}/rebalancing_history.csv')
print(history)  # Shows dates and comma-separated ticker lists
```

### rebalancing_details.csv
Deep dive into each position's performance:
```python
details = pd.read_csv(f'{results_folder}/rebalancing_details.csv')

# Analyze trade performance
print(f"Average profit per trade: {details['Profit_Pct'].mean():.2f}%")
print(f"Win rate: {(details['Profit_Pct'] > 0).mean():.1%}")
print(f"Best trade: {details.nlargest(1, 'Profit_Pct')[['Ticker', 'Profit_Pct']].values}")
print(f"Worst trade: {details.nsmallest(1, 'Profit_Pct')[['Ticker', 'Profit_Pct']].values}")

# Top performing stocks
top_performers = details.nlargest(10, 'Profit_Pct')[['Ticker', 'Entry_Price', 'Exit_Price', 'Profit_Pct']]

# Analyze by sector (russell2000_backtest.py only)
sector_performance = details.groupby('Sector')['Profit_Pct'].agg(['mean', 'count'])

# Identify stocks held multiple periods (core holdings)
holding_frequency = details['Ticker'].value_counts()

# Calculate holding period returns distribution
import matplotlib.pyplot as plt
details['Profit_Pct'].hist(bins=50)
plt.xlabel('Profit %')
plt.title('Distribution of Trade Returns')
```

### quarterly_summary.csv and quarterly_summary.png

High-level quarterly performance overview. The backtest automatically generates both a CSV file and visualization charts:

**Automatic visualization (`quarterly_summary.png`):**
- Chart 1: Quarterly average returns (green/red bar chart)
- Chart 2: Win rate trend (line chart with 50% reference line)
- Chart 3: Trading activity & cumulative profit (dual-axis)
- Chart 4: Average win vs loss per quarter (side-by-side comparison)

**Programmatic analysis:**
```python
quarterly = pd.read_csv(f'{results_folder}/quarterly_summary.csv')

# View quarterly performance
print(quarterly[['Quarter', 'N_Trades', 'Avg_Return_Pct', 'Win_Rate', 'Total_Profit']])

# Identify best/worst quarters
print("\nBest quarter:")
print(quarterly.nlargest(1, 'Avg_Return_Pct')[['Quarter', 'Avg_Return_Pct', 'Win_Rate']])

print("\nWorst quarter:")
print(quarterly.nsmallest(1, 'Avg_Return_Pct')[['Quarter', 'Avg_Return_Pct', 'Win_Rate']])

# Track performance over time
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Quarterly returns
ax1.bar(range(len(quarterly)), quarterly['Avg_Return_Pct'],
        color=['g' if x > 0 else 'r' for x in quarterly['Avg_Return_Pct']])
ax1.set_xticks(range(len(quarterly)))
ax1.set_xticklabels(quarterly['Quarter'], rotation=45)
ax1.set_ylabel('Avg Return %')
ax1.set_title('Quarterly Returns')
ax1.grid(True, alpha=0.3)

# Cumulative profit
ax2.plot(quarterly['Quarter'], quarterly['Cumulative_Profit'], marker='o')
ax2.set_ylabel('Cumulative Profit')
ax2.set_title('Cumulative Profit Over Time')
ax2.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
```

## Code Modification Guidelines

### Adjusting Factor Weights

To change factor importance, modify the weights in `calculate_composite_score`:
```python
df['composite_score'] = (
    df['quality_score'] * 0.35 +  # Adjust these weights
    df['growth_score'] * 0.35 +
    df['value_score'] * 0.15 +
    df['momentum_score'] * 0.15
)
```

### Adding New Factors

1. Extract raw data in `calculate_factors` from `info` dictionary
2. Add Z-score normalization in `calculate_composite_score`
3. Include in appropriate factor score calculation (quality/growth/value/momentum)

### Changing Portfolio Construction

In `russell2000_backtest.py`, modify `select_portfolio` to implement:
- Different weighting schemes (risk parity, market cap weighted)
- Alternative filtering criteria
- Multi-factor ranking variations

### Testing Different Universes

Replace the ticker source in `main()`:
- Use a different CSV file
- Filter by market cap, sector, or liquidity
- Modify `sample_size` for faster iteration

## Version Control

The repository includes a `.gitignore` file that excludes:
- Cache files (`stock_data_cache.pkl`, `backtest_cache/`)
- Results folders (`results_*/`)
- Generated CSV files
- Python/IDE/OS temporary files

This keeps your repository clean and avoids committing large cache files or temporary results.

## Known Limitations

- yfinance API rate limits: Scripts use sleep delays and batch processing
- Historical fundamental data may be limited for some stocks
- survivorship bias: Only includes current Russell 2000 constituents
- Transaction costs not modeled (though turnover is tracked)
- Chart generation requires GUI environment (comment out `plot_results`/`create_charts` for headless)

## Common Issues

### String values in numeric columns
yfinance occasionally returns string values (e.g., 'N/A') for numeric fields. The code handles this with `pd.to_numeric(..., errors='coerce')` before clipping operations. If you add new factor calculations involving `.clip()`, ensure numeric conversion first:

```python
# Good
df['new_ratio_z'] = safe_zscore(pd.to_numeric(df['new_ratio'], errors='coerce').clip(upper=100), reverse=True)

# Bad - will fail if column contains strings
df['new_ratio_z'] = safe_zscore(df['new_ratio'].clip(upper=100), reverse=True)
```

### Timezone-aware datetime indices
yfinance returns price data with timezone-aware datetime indices (typically America/New_York). The backtest functions strip timezone information using `tz_localize(None)` to avoid comparison errors with timezone-naive date ranges. If you modify date filtering logic, ensure timezone consistency.

### NaN scores from insufficient data
When testing with small sample sizes (e.g., 100 stocks), many may fail data collection or lack fundamental data. If too few stocks have valid values for a factor, the standard deviation becomes 0, causing NaN z-scores. The code handles this by:
- Returning 0 for all z-scores when std = 0 or NaN
- Filling NaN factor values with median (or 0 if all values are NaN)

If you see all NaN scores, increase the sample size or check data quality:
```python
# In main() function
sample_size = min(500, len(tickers))  # Increase from 100 to 500
```

### Cache Management

Both `generate_portfolio.py` and `russell2000_backtest.py` use smart caching with shared fundamental data:

**How it works:**
- **Shared fundamental cache** (`stock_data_cache.pkl`):
  - Used by BOTH scripts
  - Contains stock info (market cap, financials, ratios)
  - First run: 5-10 minutes to download
  - Shared to avoid duplicate downloads
  - Cache expires after 7 days

- **Backtest price cache** (`price_history_cache.pkl`):
  - Used only by backtest script
  - Contains historical price data for configured date range
  - First run: ~25 minutes to download price history
  - Cache is date-range specific (invalidated if START_DATE or END_DATE changes)
  - Cache expires after 7 days

**Performance:**
- Portfolio generator: 5-10 min first run → few seconds subsequent runs
- Backtest: 30-60 min first run → few seconds subsequent runs
- Running portfolio generator first downloads fundamentals, speeding up backtest
- Running backtest first downloads both, speeding up portfolio generator

**Refreshing cache:**
```bash
# Refresh both scripts (fundamental data)
python generate_portfolio.py --refresh
python update_cache.py
rm stock_data_cache.pkl

# Refresh backtest only (price history)
python russell2000_backtest.py --refresh
rm price_history_cache.pkl

# Refresh everything
rm stock_data_cache.pkl price_history_cache.pkl
```

**Adjusting cache duration:**
```python
# In CONFIG dictionary (both scripts)
CONFIG = {
    ...
    'CACHE_DAYS': 7,  # Change to 1 for daily updates, 30 for monthly
}
```

**When to refresh:**
- Before important portfolio rebalancing decisions
- When fundamental data may have changed (after earnings season)
- After changing backtest date range (price cache auto-invalidates)
- If you suspect stale data (check cache age in console output)

**Cache behavior:**
- Price cache is automatically invalidated when START_DATE or END_DATE changes
- Fundamental cache is shared between both scripts
- Both caches work independently but complement each other
- Deleting one cache doesn't affect the other
