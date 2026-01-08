# Russell 2000 Small-Cap Quant Strategy

A quantitative trading strategy for Russell 2000 small-cap stocks based on quality, growth, value, and momentum factors. Features comprehensive backtesting, portfolio optimization with sector constraints, and actionable trade recommendations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Strategy Overview

This strategy identifies high-quality, high-growth small-cap stocks using a multi-factor approach:

- **Quality (35%)**: ROE, ROA, margins, debt ratios
- **Growth (35%)**: Revenue and earnings growth
- **Value (15%)**: P/E, PEG, P/S ratios
- **Momentum (15%)**: 12-1 month price momentum, volatility

**Key Features:**
- Equal-weight portfolio with sector concentration limits (max 30% per sector)
- Quarterly rebalancing
- Smart caching system for fast iteration (7-day cache validity)
- Comprehensive performance analytics with quarterly breakdowns

## 📊 Performance Highlights

The backtest engine provides detailed analytics including:
- Cumulative returns vs benchmark (equal-weight Russell 2000)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Maximum drawdown analysis
- Trade-level P&L tracking
- Quarterly performance summaries
- Win rate and average profit per trade

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/russell2000-quant-strategy.git
cd russell2000-quant-strategy

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install yfinance pandas numpy matplotlib Pillow openpyxl
```

### Basic Usage

#### Option A: Graphical User Interface (Recommended for Beginners)

```bash
# Launch the GUI application
python gui_app.py
```

**Features:**
- 📈 **Backtest Tab**: Run historical backtests with configurable parameters
- 📊 **Results Tab**: View and analyze outputs in tabular format
- 🖼️ **Chart Viewer**: View performance charts and analytics
- 💾 **Export to Excel**: Export all results to Excel workbook
- ⚙️ **Configuration**: View and plan strategy parameters

**GUI Interface:**
- Interactive controls for all parameters
- Real-time progress indicators
- Integrated chart viewer
- Excel export functionality

#### Option B: Command Line Interface

Run the backtest (generates current portfolio + historical analysis):

```bash
# First run: Downloads data (~30-60 minutes)
python russell2000_backtest.py

# Subsequent runs: Uses cache (few seconds)
python russell2000_backtest.py

# Force refresh all data
python russell2000_backtest.py --refresh
```

**Output:** (saved in `results_YYYYMMDD_HHMMSS/`)
- `current_portfolio.csv` - **Actionable buy list** with share counts and prices
- `backtest_results.png` - Performance charts
- `quarterly_summary.png` - Quarterly analytics
- `rebalancing_details.csv` - Trade-by-trade breakdown
- `quarterly_summary.csv` - Quarterly metrics
- `factor_scores.csv` - All stocks with factor scores

**The backtest generates everything you need:**
- Current portfolio recommendations (with exact shares to buy)
- Historical performance analysis
- Trade-level P&L tracking
- Quarterly performance metrics

## 📈 Using the Output

### Execute Portfolio Trades

The backtest generates ready-to-execute buy orders in `current_portfolio.csv`:

```python
import pandas as pd

# Load portfolio from latest results folder
portfolio = pd.read_csv('results_20251231_143052/current_portfolio.csv')

# View buy orders
print(portfolio[['Ticker', 'Current_Price', 'Shares_to_Buy', 'Actual_Amount']])

# Example output:
# Ticker  Current_Price  Shares_to_Buy  Actual_Amount
# TGTX         29.75           168        4998.00
# SLDE         19.55           255        4985.25
# ...

# Total investment
total = portfolio['Actual_Amount'].sum()
print(f"\nTotal to invest: ${total:,.2f}")
```

### Analyze Trade History

```python
# Load detailed trade history
details = pd.read_csv('results_20251231_143052/rebalancing_details.csv')

# Top performers
print(details.nlargest(10, 'Profit_Pct')[['Ticker', 'Entry_Price', 'Exit_Price', 'Profit_Pct']])

# Performance by sector
print(details.groupby('Sector')['Profit_Pct'].agg(['mean', 'count', 'std']))

# Win rate
win_rate = (details['Profit_Pct'] > 0).mean()
print(f"Win Rate: {win_rate:.1%}")
```

## ⚙️ Configuration

Customize strategy parameters in the `CONFIG` dictionary (`russell2000_backtest.py`):

```python
CONFIG = {
    'START_DATE': '2021-01-01',
    'END_DATE': datetime.now().strftime('%Y-%m-%d'),
    'TOP_N': 20,                    # Number of stocks
    'REBALANCE_FREQ': 'Q',          # 'Q' = quarterly, 'M' = monthly
    'MAX_SECTOR_WEIGHT': 0.30,      # Max 30% per sector
    'MIN_MARKET_CAP': 300e6,        # Min $300M market cap
    'MAX_MARKET_CAP': 10e9,         # Max $10B market cap
    'MIN_AVG_VOLUME': 100000,       # Min daily volume
    'INITIAL_BALANCE': 100000,      # Starting capital for portfolio
    'CACHE_DAYS': 7,                # Cache validity (days)
}
```

## 🔧 Advanced Usage

### Adjust Factor Weights

Modify the composite score calculation in `calculate_composite_score()`:

```python
df['composite_score'] = (
    df['quality_score'] * 0.35 +    # Adjust these weights
    df['growth_score'] * 0.35 +
    df['value_score'] * 0.15 +
    df['momentum_score'] * 0.15
)
```

### Add New Factors

1. Extract raw data in `calculate_factors()` from `info` dictionary
2. Add Z-score normalization in `calculate_composite_score()`
3. Include in appropriate factor score calculation

### Change Universe

Replace `russell2000_tickers.csv` with your own ticker list:

```csv
Ticker,Name,Sector,Weight (%),Market Value,Price
AAPL,Apple Inc.,Technology,2.5,3000000000000,175.50
...
```

## 📁 Project Structure

```
russell2000-quant-strategy/
├── gui_app.py                   # 🎨 Graphical user interface
├── russell2000_backtest.py      # Main backtest engine (generates everything)
├── smallcap_backtest.py         # Simplified prototype
├── russell2000_tickers.csv      # Stock universe (1,959 stocks)
├── russell2000_tickers.txt      # Alternative format
├── requirements.txt             # Python dependencies
├── CLAUDE.md                    # Comprehensive documentation
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

### Output Files (Ignored by Git)

```
stock_data_cache.pkl             # Fundamental data cache
price_history_cache.pkl          # Price history cache
results_YYYYMMDD_HHMMSS/         # Backtest results folders
    ├── current_portfolio.csv    # Actionable buy list
    ├── backtest_results.png     # Performance charts
    ├── quarterly_summary.png    # Quarterly analytics
    ├── rebalancing_details.csv  # Trade details
    ├── quarterly_summary.csv    # Quarterly metrics
    └── factor_scores.csv        # All stock rankings
```

## 🎓 Strategy Details

### Factor Calculation

#### Quality Score (35%)
- **ROE** (25%): Return on equity
- **ROA** (20%): Return on assets
- **Gross Margin** (20%): Profitability
- **Operating Margin** (20%): Operating efficiency
- **Debt/Equity** (15%): Financial leverage (lower is better)

#### Growth Score (35%)
- **Revenue Growth** (40%): Top-line growth
- **Earnings Growth** (35%): Bottom-line growth
- **Quarterly Earnings Growth** (25%): Recent momentum

#### Value Score (15%)
- **P/E Ratio** (35%): Price to earnings (lower is better)
- **PEG Ratio** (35%): P/E to growth (lower is better)
- **P/S Ratio** (30%): Price to sales (lower is better)

#### Momentum Score (15%)
- **12-1 Month Momentum** (70%): Price momentum excluding last month
- **Low Volatility** (30%): Stability preference

### Portfolio Construction

1. **Rank** all stocks by composite score
2. **Filter** by market cap ($300M - $10B) and liquidity (100K+ daily volume)
3. **Select** top N stocks (default: 20)
4. **Apply** sector constraints (max 30% per sector)
5. **Equal-weight** allocation across selected stocks

### Backtest Methodology

- **Rebalancing**: Quarterly (first day of quarter)
- **Benchmark**: Equal-weight portfolio of entire universe
- **Transaction costs**: Not modeled
- **Survivorship bias**: Present (uses current constituents only)

## 📊 Performance Metrics

The backtest calculates:

| Metric | Description |
|--------|-------------|
| Total Return | Cumulative return over backtest period |
| Annual Return | Annualized return (CAGR) |
| Annual Volatility | Standard deviation of returns |
| Sharpe Ratio | Risk-adjusted return (excess return / volatility) |
| Sortino Ratio | Downside risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| Calmar Ratio | Annual return / max drawdown |
| Win Rate | Percentage of profitable days |

## 🔄 Cache Management

The backtest uses smart caching to speed up execution:

### Cache Types

1. **Fundamental Data** (`stock_data_cache.pkl`)
   - Contains: Market cap, financials, ratios
   - Validity: 7 days (configurable)
   - First download: ~5-10 minutes
   - Reused across runs

2. **Price History** (`price_history_cache.pkl`)
   - Contains: Historical price data for configured date range
   - Validity: 7 days, invalidated if date range changes
   - First download: ~25 minutes
   - Reused across runs

**Performance:**
- First run: 30-60 minutes (downloads all data)
- Subsequent runs: Few seconds (uses cache)

### Refresh Cache

```bash
# Delete all cache
rm *.pkl

# Or use --refresh flag
python russell2000_backtest.py --refresh
```

**When to refresh:**
- Before important rebalancing decisions
- After changing date range (price cache auto-invalidates)
- If you suspect stale data (check cache age in console)

## ⚠️ Known Limitations

- **Survivorship bias**: Only current Russell 2000 constituents included
- **Transaction costs**: Not modeled in backtest
- **Market impact**: Assumes perfect execution at market prices
- **Data quality**: Depends on yfinance API availability and accuracy
- **Look-ahead bias**: None (factors calculated using only historical data)

## 🛠️ Troubleshooting

### Common Issues

**"Insufficient data" error:**
- Increase `SAMPLE_SIZE` or check internet connection
- Some stocks may have incomplete fundamental data

**"String values in numeric columns" error:**
- Already handled with `pd.to_numeric(..., errors='coerce')`
- If adding new factors, ensure numeric conversion before operations

**Timezone warnings:**
- Automatically handled by `tz_localize(None)`
- Price data is converted to timezone-naive format

**All NaN scores:**
- Increase sample size for better factor statistics
- Check data quality in `factor_scores.csv`

## 📚 Documentation

- **CLAUDE.md**: Comprehensive guide for using with Claude Code
- **Code comments**: Detailed inline documentation
- **Docstrings**: Function-level documentation

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add transaction cost modeling
- [ ] Implement risk parity weighting
- [ ] Add machine learning factor selection
- [ ] Support for different rebalancing frequencies
- [ ] Web dashboard for results visualization
- [ ] Real-time portfolio monitoring
- [ ] Options overlay strategies

## 📄 License

MIT License - feel free to use for personal or commercial purposes.

## ⚡ Credits

Built with:
- [yfinance](https://github.com/ranaroussi/yfinance) - Financial data
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [numpy](https://numpy.org/) - Numerical computing
- [matplotlib](https://matplotlib.org/) - Visualization

---

**Disclaimer**: This is educational software for backtesting purposes only. Not financial advice. Past performance does not guarantee future results. Always do your own research and consult with financial professionals before making investment decisions.

🤖 *Generated with [Claude Code](https://claude.com/claude-code)*
