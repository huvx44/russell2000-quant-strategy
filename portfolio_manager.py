"""
Portfolio Manager - Core Business Logic
========================================
Manages user's actual portfolio holdings with real-time tracking.

Features:
- Load/save portfolio holdings to CSV
- Track historical snapshots
- Fetch current prices via yfinance
- Calculate P&L and performance metrics
- Add/remove positions

Author: Portfolio Management System
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import time


class PortfolioManager:
    """
    Core portfolio management class for tracking holdings and performance
    """

    def __init__(self, portfolio_file='my_portfolio.csv', history_file='portfolio_history.csv'):
        """
        Initialize portfolio manager

        Args:
            portfolio_file: Path to CSV file storing current holdings
            history_file: Path to CSV file storing historical snapshots
        """
        self.portfolio_file = portfolio_file
        self.history_file = history_file
        self.holdings = pd.DataFrame()
        self.history = pd.DataFrame()
        self.last_refresh = None

        # Define column schemas
        self.holdings_columns = [
            'Ticker', 'Sector', 'Entry_Date', 'Entry_Price', 'Shares_Owned',
            'Cost_Basis', 'Current_Price', 'Current_Value', 'Unrealized_PL',
            'Unrealized_PL_Pct', 'Composite_Score', 'Quality_Score',
            'Growth_Score', 'Value_Score', 'Momentum_Score', 'Last_Updated'
        ]

        self.history_columns = [
            'Date', 'Total_Value', 'Total_Cost_Basis', 'Total_Unrealized_PL',
            'Total_Return_Pct', 'Num_Positions'
        ]

        # Auto-load on initialization
        self.load_portfolio()
        self.load_history()

    def load_portfolio(self):
        """
        Load portfolio holdings from CSV file

        Returns:
            pd.DataFrame: Holdings dataframe
        """
        if os.path.exists(self.portfolio_file):
            try:
                self.holdings = pd.read_csv(self.portfolio_file)
                print(f"Loaded portfolio: {len(self.holdings)} positions from {self.portfolio_file}")
            except Exception as e:
                print(f"Error loading portfolio: {e}")
                self.holdings = pd.DataFrame(columns=self.holdings_columns)
        else:
            print(f"No existing portfolio found. Starting fresh.")
            self.holdings = pd.DataFrame(columns=self.holdings_columns)

        return self.holdings

    def save_portfolio(self):
        """
        Save current holdings to CSV file
        """
        try:
            self.holdings.to_csv(self.portfolio_file, index=False)
            print(f"Portfolio saved: {len(self.holdings)} positions to {self.portfolio_file}")
            return True
        except Exception as e:
            print(f"Error saving portfolio: {e}")
            return False

    def load_history(self):
        """
        Load historical snapshots from CSV file

        Returns:
            pd.DataFrame: History dataframe
        """
        if os.path.exists(self.history_file):
            try:
                self.history = pd.read_csv(self.history_file)
                print(f"Loaded history: {len(self.history)} snapshots from {self.history_file}")
            except Exception as e:
                print(f"Error loading history: {e}")
                self.history = pd.DataFrame(columns=self.history_columns)
        else:
            print(f"No existing history found. Starting fresh.")
            self.history = pd.DataFrame(columns=self.history_columns)

        return self.history

    def append_snapshot(self, snapshot=None):
        """
        Append current portfolio state to history file

        Args:
            snapshot: Optional dict with snapshot data. If None, calculates from current holdings.

        Returns:
            bool: Success status
        """
        if snapshot is None:
            # Calculate snapshot from current holdings
            if len(self.holdings) == 0:
                print("No holdings to snapshot")
                return False

            summary = self.calculate_portfolio_summary()
            snapshot = {
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Total_Value': summary['total_value'],
                'Total_Cost_Basis': summary['total_cost_basis'],
                'Total_Unrealized_PL': summary['total_unrealized_pl'],
                'Total_Return_Pct': summary['total_return_pct'],
                'Num_Positions': summary['num_positions']
            }

        # Check if snapshot for today already exists
        today = datetime.now().strftime('%Y-%m-%d')
        if len(self.history) > 0 and today in self.history['Date'].values:
            # Update existing snapshot
            self.history.loc[self.history['Date'] == today] = pd.Series(snapshot)
            print(f"Updated snapshot for {today}")
        else:
            # Append new snapshot
            self.history = pd.concat([self.history, pd.DataFrame([snapshot])], ignore_index=True)
            print(f"Added new snapshot for {today}")

        # Save to file
        try:
            self.history.to_csv(self.history_file, index=False)
            print(f"History saved: {len(self.history)} snapshots")
            return True
        except Exception as e:
            print(f"Error saving history: {e}")
            return False

    def fetch_current_price(self, ticker):
        """
        Fetch latest price for a single ticker using yfinance

        Args:
            ticker: Stock symbol

        Returns:
            float: Current price, or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')

            if not hist.empty:
                return hist['Close'].iloc[-1]
            else:
                # Fallback to info dict
                info = stock.info
                price = info.get('currentPrice') or info.get('regularMarketPrice')
                return price
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
            return None

    def refresh_all_prices(self, progress_callback=None):
        """
        Update current prices for all holdings

        Args:
            progress_callback: Optional function(current, total, ticker) to report progress

        Returns:
            dict: Summary of refresh operation
        """
        if len(self.holdings) == 0:
            print("No holdings to refresh")
            return {'success': 0, 'failed': 0, 'total': 0}

        success_count = 0
        failed_count = 0
        total = len(self.holdings)

        print(f"\nRefreshing prices for {total} positions...")

        for idx, row in self.holdings.iterrows():
            ticker = row['Ticker']

            if progress_callback:
                progress_callback(idx + 1, total, ticker)

            # Fetch current price
            current_price = self.fetch_current_price(ticker)

            if current_price is not None:
                # Update holdings dataframe
                self.holdings.at[idx, 'Current_Price'] = current_price
                self.holdings.at[idx, 'Current_Value'] = current_price * row['Shares_Owned']
                self.holdings.at[idx, 'Unrealized_PL'] = self.holdings.at[idx, 'Current_Value'] - row['Cost_Basis']
                self.holdings.at[idx, 'Unrealized_PL_Pct'] = ((current_price / row['Entry_Price']) - 1) * 100
                self.holdings.at[idx, 'Last_Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                success_count += 1
            else:
                failed_count += 1
                print(f"  Failed to update {ticker}")

            # Rate limiting: small delay between requests
            time.sleep(0.3)

        self.last_refresh = datetime.now()

        print(f"\nRefresh complete: {success_count} succeeded, {failed_count} failed")

        return {
            'success': success_count,
            'failed': failed_count,
            'total': total,
            'timestamp': self.last_refresh
        }

    def calculate_position_metrics(self, row):
        """
        Calculate metrics for a single position

        Args:
            row: pd.Series representing one position

        Returns:
            dict: Calculated metrics
        """
        cost_basis = row['Entry_Price'] * row['Shares_Owned']
        current_value = row['Current_Price'] * row['Shares_Owned']
        unrealized_pl = current_value - cost_basis
        unrealized_pl_pct = ((row['Current_Price'] / row['Entry_Price']) - 1) * 100

        return {
            'Cost_Basis': cost_basis,
            'Current_Value': current_value,
            'Unrealized_PL': unrealized_pl,
            'Unrealized_PL_Pct': unrealized_pl_pct
        }

    def calculate_portfolio_summary(self):
        """
        Calculate overall portfolio metrics

        Returns:
            dict: Portfolio summary statistics
        """
        if len(self.holdings) == 0:
            return {
                'total_value': 0,
                'total_cost_basis': 0,
                'total_unrealized_pl': 0,
                'total_return_pct': 0,
                'num_positions': 0
            }

        total_cost_basis = self.holdings['Cost_Basis'].sum()
        total_value = self.holdings['Current_Value'].sum()
        total_unrealized_pl = total_value - total_cost_basis
        total_return_pct = ((total_value / total_cost_basis) - 1) * 100 if total_cost_basis > 0 else 0

        return {
            'total_value': total_value,
            'total_cost_basis': total_cost_basis,
            'total_unrealized_pl': total_unrealized_pl,
            'total_return_pct': total_return_pct,
            'num_positions': len(self.holdings)
        }

    def add_position(self, ticker, entry_date, entry_price, shares, sector='',
                    composite_score=0, quality_score=0, growth_score=0, value_score=0, momentum_score=0):
        """
        Add new position to holdings

        Args:
            ticker: Stock symbol
            entry_date: Purchase date (YYYY-MM-DD)
            entry_price: Purchase price per share
            shares: Number of shares
            sector: Industry sector
            composite_score: Overall factor score (optional)
            quality_score: Quality factor score (optional)
            growth_score: Growth factor score (optional)
            value_score: Value factor score (optional)
            momentum_score: Momentum factor score (optional)

        Returns:
            bool: Success status
        """
        # Validate inputs
        if shares <= 0 or entry_price <= 0:
            print("Error: Shares and entry price must be positive")
            return False

        # Check if ticker already exists
        if ticker in self.holdings['Ticker'].values:
            print(f"Warning: {ticker} already exists in portfolio. Consider updating instead.")
            return False

        # Fetch current price
        print(f"Fetching current price for {ticker}...")
        current_price = self.fetch_current_price(ticker)

        if current_price is None:
            print(f"Warning: Could not fetch current price for {ticker}. Using entry price as current.")
            current_price = entry_price

        # Calculate metrics
        cost_basis = entry_price * shares
        current_value = current_price * shares
        unrealized_pl = current_value - cost_basis
        unrealized_pl_pct = ((current_price / entry_price) - 1) * 100

        # Create new position
        new_position = {
            'Ticker': ticker,
            'Sector': sector,
            'Entry_Date': entry_date,
            'Entry_Price': entry_price,
            'Shares_Owned': shares,
            'Cost_Basis': cost_basis,
            'Current_Price': current_price,
            'Current_Value': current_value,
            'Unrealized_PL': unrealized_pl,
            'Unrealized_PL_Pct': unrealized_pl_pct,
            'Composite_Score': composite_score,
            'Quality_Score': quality_score,
            'Growth_Score': growth_score,
            'Value_Score': value_score,
            'Momentum_Score': momentum_score,
            'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Append to holdings
        self.holdings = pd.concat([self.holdings, pd.DataFrame([new_position])], ignore_index=True)

        print(f"Added {ticker}: {shares} shares @ ${entry_price:.2f}, Current: ${current_price:.2f}, P&L: ${unrealized_pl:.2f}")

        return True

    def remove_position(self, ticker):
        """
        Remove position from holdings

        Args:
            ticker: Stock symbol to remove

        Returns:
            bool: Success status
        """
        if ticker not in self.holdings['Ticker'].values:
            print(f"Error: {ticker} not found in portfolio")
            return False

        self.holdings = self.holdings[self.holdings['Ticker'] != ticker].reset_index(drop=True)
        print(f"Removed {ticker} from portfolio")

        return True

    def get_recommended_portfolio(self):
        """
        Load latest backtest recommendations from results_*/ folder

        Returns:
            pd.DataFrame: Recommended portfolio, or None if not found
        """
        import glob

        folders = glob.glob('results_*/')
        if not folders:
            print("No backtest results found")
            return None

        latest_folder = max(folders, key=os.path.getctime)
        portfolio_file = os.path.join(latest_folder, 'current_portfolio.csv')

        if not os.path.exists(portfolio_file):
            print(f"Portfolio file not found in {latest_folder}")
            return None

        try:
            recommended_df = pd.read_csv(portfolio_file)
            print(f"Loaded recommended portfolio from {portfolio_file}: {len(recommended_df)} positions")
            return recommended_df
        except Exception as e:
            print(f"Error loading recommended portfolio: {e}")
            return None

    def import_from_recommended(self, recommended_df, adjustments):
        """
        Import positions from recommended portfolio with user adjustments

        Args:
            recommended_df: DataFrame with recommended positions
            adjustments: Dict mapping ticker -> {'shares': int, 'price': float, 'date': str}

        Returns:
            int: Number of positions imported
        """
        imported_count = 0

        for idx, row in recommended_df.iterrows():
            ticker = row['Ticker']

            if ticker in adjustments:
                # Use user-adjusted values
                adj = adjustments[ticker]
                shares = adj.get('shares', row.get('Shares_to_Buy', 0))
                entry_price = adj.get('price', row.get('Current_Price', 0))
                entry_date = adj.get('date', datetime.now().strftime('%Y-%m-%d'))

                if shares > 0 and entry_price > 0:
                    success = self.add_position(
                        ticker=ticker,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        shares=shares,
                        sector=row.get('Sector', ''),
                        composite_score=row.get('Composite_Score', 0),
                        quality_score=row.get('Quality_Score', 0),
                        growth_score=row.get('Growth_Score', 0),
                        value_score=row.get('Value_Score', 0),
                        momentum_score=row.get('Momentum_Score', 0)
                    )

                    if success:
                        imported_count += 1

        print(f"\nImported {imported_count} positions from recommended portfolio")
        return imported_count


if __name__ == "__main__":
    # Test the portfolio manager
    print("Testing PortfolioManager...")

    pm = PortfolioManager()

    # Add test position
    pm.add_position(
        ticker='AAPL',
        entry_date='2026-01-01',
        entry_price=180.00,
        shares=10,
        sector='Technology'
    )

    # Display holdings
    print("\nCurrent Holdings:")
    print(pm.holdings[['Ticker', 'Entry_Price', 'Shares_Owned', 'Current_Price', 'Unrealized_PL']])

    # Calculate summary
    summary = pm.calculate_portfolio_summary()
    print("\nPortfolio Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Save
    pm.save_portfolio()

    # Take snapshot
    pm.append_snapshot()

    print("\nTest complete!")
