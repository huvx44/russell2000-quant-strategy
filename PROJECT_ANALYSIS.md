# Russell 2000 Quant Strategy - Project Analysis

## Executive Summary

This is a quantitative trading research system for small-cap stocks (Russell 2000 universe). It implements a multi-factor strategy combining quality, growth, value, and momentum signals with backtesting, portfolio generation, and portfolio management capabilities. The system includes both CLI and GUI interfaces.

**Tech stack:** Python, yfinance, pandas, numpy, matplotlib, tkinter
**Universe:** 1,959 Russell 2000 constituents
**Strategy:** Multi-factor ranking with quarterly/monthly rebalancing

---

## Project Structure

```
russell2000-quant-strategy/
├── russell2000_backtest.py     (1,533 lines) - Production backtest engine
├── smallcap_backtest.py        (773 lines)   - Simplified prototype (50 stocks)
├── gui_app.py                  (2,056 lines) - Tkinter GUI application
├── portfolio_manager.py        (785 lines)   - Portfolio tracking & versioning
├── russell2000_tickers.csv     (1,960 rows)  - Russell 2000 constituent data
├── russell2000_tickers.txt     (533 tickers)  - Plain text ticker list
├── requirements.txt            - Python dependencies
├── CLAUDE.md                   - Development documentation
├── README.md                   - User documentation
└── .gitignore                  - Cache/results exclusions
```

**Total:** ~5,147 lines of Python across 4 scripts.

---

## Architecture Overview

### Data Pipeline

```
Russell 2000 CSV (1,959 tickers)
        │
        ▼
yfinance API (batched, rate-limited)
        │
        ├──► Fundamental cache (stock_data_cache.pkl, shared)
        └──► Price history cache (price_history_cache.pkl, backtest-only)
                │
                ▼
Factor Calculation (20+ raw factors)
        │
        ▼
Z-score Normalization (winsorized 5-95th percentile)
        │
        ▼
Composite Score = Quality×0.35 + Growth×0.35 + Value×0.15 + Momentum×0.15
        │
        ▼
Portfolio Selection (top N, sector-constrained)
        │
        ▼
Backtest Simulation (quarterly/monthly rebalancing)
        │
        ▼
Performance Metrics + Visualizations + CSV Exports
```

### Component Relationships

| Component | Role | Calls |
|-----------|------|-------|
| `gui_app.py` | User interface | Spawns `russell2000_backtest.py` as subprocess; imports `PortfolioManager` |
| `russell2000_backtest.py` | Backtest engine | Standalone CLI script |
| `smallcap_backtest.py` | Prototype | Standalone CLI script |
| `portfolio_manager.py` | Holdings tracker | Imported by `gui_app.py` |

---

## Factor Model Details

### Quality Score (35% of composite)

| Factor | Sub-weight | Direction | Source |
|--------|-----------|-----------|--------|
| ROE | 25% | Higher is better | `info['returnOnEquity']` |
| ROA | 20% | Higher is better | `info['returnOnAssets']` |
| Gross Margin | 20% | Higher is better | `info['grossMargins']` |
| Operating Margin | 20% | Higher is better | `info['operatingMargins']` |
| Debt/Equity | 15% | Lower is better (reversed) | `info['debtToEquity']` |

### Growth Score (35% of composite)

| Factor | Sub-weight | Direction | Source |
|--------|-----------|-----------|--------|
| Revenue Growth | 40% | Higher is better | `info['revenueGrowth']` |
| Earnings Growth | 35% | Higher is better | `info['earningsGrowth']` |
| Quarterly Earnings Growth | 25% | Higher is better | `info['earningsQuarterlyGrowth']` |

### Value Score (15% of composite)

| Factor | Sub-weight | Direction | Clipping |
|--------|-----------|-----------|----------|
| P/E Ratio | 35% | Lower is better (reversed) | Capped at 100x |
| PEG Ratio | 35% | Lower is better (reversed) | Capped at 5x |
| P/S Ratio | 30% | Lower is better (reversed) | Capped at 20x |

### Momentum Score (15% of composite)

| Factor | Sub-weight | Direction |
|--------|-----------|-----------|
| 12-1 Month Momentum | 70% | Higher is better |
| Volatility | 30% | Lower is better (reversed) |

### Normalization Method

All factors undergo:
1. **Winsorization** at 5th/95th percentiles (clips outliers)
2. **Z-score standardization** (mean=0, std=1)
3. **NaN handling** via median fill (fallback to 0 if all NaN)
4. **Zero-std protection** returns 0 for all values when std=0

---

## Configuration Parameters

```python
CONFIG = {
    'START_DATE': '2021-01-01',
    'END_DATE': datetime.now(),
    'TOP_N': 20,                    # Portfolio size
    'REBALANCE_FREQ': 'Q',         # Q=quarterly, M=monthly
    'MAX_SECTOR_WEIGHT': 0.30,     # 30% sector cap
    'MIN_MARKET_CAP': 300e6,       # $300M floor
    'MAX_MARKET_CAP': 10e9,        # $10B ceiling
    'MIN_AVG_VOLUME': 100000,      # 100K shares/day
    'INITIAL_BALANCE': 100000,     # $100K starting capital
    'BATCH_SIZE': 50,              # API batch size
    'SLEEP_TIME': 1,               # Rate limiting (seconds)
    'CACHE_DAYS': 7,               # Cache validity
}
```

---

## Caching System

Two-layer smart cache reduces runtime from 30-60 minutes to seconds:

| Cache | File | Shared | Contents | Invalidation |
|-------|------|--------|----------|-------------|
| Fundamental | `stock_data_cache.pkl` | Yes (backtest + portfolio gen) | Market cap, financials, ratios | 7-day expiry |
| Price History | `price_history_cache.pkl` | No (backtest only) | Historical OHLCV data | 7-day expiry + date range change |

---

## Output Files

Each backtest run creates a timestamped folder `results_YYYYMMDD_HHMMSS/`:

| File | Purpose |
|------|---------|
| `current_portfolio.csv` | Actionable buy list with share counts and prices |
| `backtest_results.png` | 4-panel chart (returns, drawdown, monthly heatmap, sectors) |
| `period_summary.png` | 4-panel quarterly dashboard |
| `backtest_daily_returns.csv` | Daily strategy vs benchmark returns |
| `factor_scores.csv` | All stocks with factor scores |
| `rebalancing_history.csv` | Holdings at each rebalance date |
| `rebalancing_details.csv` | Position-level entry/exit P&L |
| `period_summary.csv` | Quarterly aggregated metrics |
| `strategy_config.json` | Configuration snapshot for reproducibility |

---

## Strengths

1. **Complete end-to-end system** - Data collection through portfolio management in a single codebase
2. **Smart shared caching** - Fundamental cache shared between scripts; price cache is date-range aware with automatic invalidation
3. **Robust data handling** - Winsorization, NaN median fill, string-to-numeric coercion, timezone stripping, zero-std protection
4. **Comprehensive analytics** - Position-level P&L, quarterly summaries, win/loss analysis, sector concentration tracking
5. **Dual interface** - CLI for automation, Tkinter GUI for interactive use
6. **Portfolio versioning** - Tag, snapshot, compare, and track actual portfolio holdings over time
7. **Sector constraints** - Prevents over-concentration (30% max per sector)
8. **Configurable factor weights** - GUI allows saving/loading configuration presets as JSON
9. **API resilience** - Batch processing with sleep delays; graceful handling of yfinance failures
10. **Reproducibility** - Saves `strategy_config.json` with each backtest run

---

## Weaknesses and Risks

### Methodological

1. **Survivorship bias** - Uses current Russell 2000 constituents for historical backtests. Stocks that were delisted, acquired, or dropped from the index are excluded, which inflates backtest returns.

2. **No transaction costs** - The backtest tracks turnover but does not deduct commissions, slippage, or market impact. For a 20-stock small-cap portfolio rebalanced quarterly, this could be significant (estimated 0.5-1.5% per rebalance depending on position sizes).

3. **Look-ahead bias risk** - Fundamental data from `yfinance` represents the *most recent* values, not point-in-time data. The backtest uses current fundamentals to score stocks at historical rebalance dates, which is a form of look-ahead bias.

4. **No risk management** - No stop-losses, no position sizing based on volatility, no drawdown-based deleveraging. Equal-weight allocation ignores stock-level risk differences.

5. **Equal-weight benchmark** - The benchmark is an equal-weight portfolio of the entire universe, not a standard index like IWM. This makes outperformance harder to compare with real-world benchmarks.

### Technical

6. **No test suite** - Zero unit tests, integration tests, or data validation tests. Factor calculations and backtest logic are untested.

7. **No logging framework** - Relies entirely on `print()` statements. No log levels, no file logging, no structured output.

8. **Hardcoded factor sub-weights** - While top-level factor weights are configurable via the GUI, the sub-weights within each factor category (e.g., ROE 25%, ROA 20%) are hardcoded in `calculate_composite_score()`.

9. **Generic exception handling** - Most try/except blocks catch `Exception` broadly, which can silently mask bugs.

10. **GUI-subprocess coupling** - The GUI passes parameters to `russell2000_backtest.py` via command-line arguments and reads results from the filesystem. This is fragile - if the backtest script's CLI interface changes, the GUI breaks silently.

11. **No input validation for factor weights** - The GUI configuration tab does not verify that factor weights sum to 100%.

12. **CSV-based persistence** - The portfolio manager writes to CSV on every operation (add, edit, remove). No batched writes, no file locking, no concurrent access protection.

### Data Quality

13. **yfinance data reliability** - yfinance is an unofficial API scraping Yahoo Finance. Data can be stale, inconsistent, or unavailable. Some fields return strings instead of numbers (handled via `pd.to_numeric(..., errors='coerce')`).

14. **Limited fundamental history** - yfinance provides current fundamentals only, not time-series. Historical factor values cannot be reconstructed accurately.

15. **Small sample in prototype** - `smallcap_backtest.py` uses only 50 hardcoded tickers, many of which (LULU, DECK) are now mid/large-cap. This makes it an unreliable prototype.

---

## Comparison: Production vs Prototype

| Feature | `russell2000_backtest.py` | `smallcap_backtest.py` |
|---------|--------------------------|------------------------|
| Universe | 1,959 stocks | 50 hardcoded |
| Caching | Dual (fundamental + price) | None |
| Data collection | Batched with rate limiting | Sequential |
| Normalization | Winsorized z-score | Basic z-score |
| Sector constraints | Yes (30% max) | No |
| Portfolio shares/pricing | Yes | No |
| Config snapshot | Yes (`strategy_config.json`) | No |
| Run time (first) | 30-60 min | 5-10 min |
| Run time (cached) | Seconds | 5-10 min (no cache) |

---

## GUI Features Summary

The Tkinter GUI (`gui_app.py`, 2,056 lines) provides four tabs:

1. **Backtest Tab** - Configure date range, rebalance frequency, initial balance; run backtest with real-time output streaming
2. **Results Tab** - Browse results folders, load CSVs into sortable tables, view charts, export to Excel
3. **Configuration Tab** - Adjust factor weights and constraints, save/load presets, Korean + English documentation
4. **Portfolio Tab** - Track actual holdings, refresh prices, calculate P&L, tag versions, compare snapshots

---

## Potential Improvements

### High Priority
- Add point-in-time fundamental data (or document the look-ahead bias clearly in results)
- Implement transaction cost modeling (even a simple flat-rate deduction per rebalance)
- Add basic unit tests for factor calculations and scoring logic
- Validate factor weight sums in GUI (must equal 100%)

### Medium Priority
- Replace `print()` with Python `logging` module
- Make sub-factor weights configurable alongside top-level weights
- Add IWM (Russell 2000 ETF) as a benchmark option
- Implement simple risk management (volatility-based position sizing or max drawdown circuit breaker)

### Lower Priority
- Add type hints throughout the codebase
- Replace generic `except Exception` with specific exception types
- Implement file locking or database backend for portfolio manager
- Update `smallcap_backtest.py` ticker list (several are no longer small-cap)
- Add data quality checks (flag stocks with stale or suspicious fundamental data)

---

## Performance Metrics Computed

The backtest calculates these metrics for both strategy and benchmark:

| Metric | Formula |
|--------|---------|
| Total Return | `(final / initial) - 1` |
| Annual Return | `(1 + total_return)^(252/trading_days) - 1` |
| Annual Volatility | `daily_returns.std() × sqrt(252)` |
| Sharpe Ratio | `annual_return / annual_volatility` |
| Sortino Ratio | `annual_return / downside_deviation` |
| Max Drawdown | `max peak-to-trough decline` |
| Calmar Ratio | `annual_return / abs(max_drawdown)` |
| Win Rate | `% of positive daily returns` |

---

*Analysis generated: 2026-01-31*
