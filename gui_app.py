"""
Russell 2000 Quant Strategy - GUI Application
==============================================
Graphical interface for backtesting and results analysis

Usage:
    python gui_app.py

Features:
- Run historical backtests with configurable parameters
- View and analyze results (portfolio, quarterly summary, trades)
- Display performance charts
- Export to Excel
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import subprocess
import os
import pandas as pd
from datetime import datetime
import glob
from PIL import Image, ImageTk
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import portfolio manager
from portfolio_manager import PortfolioManager

class QuantStrategyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Russell 2000 Quant Strategy")
        self.root.geometry("1200x800")

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager()

        # Create tabs
        self.create_backtest_tab()
        self.create_results_tab()
        self.create_portfolio_tab()
        self.create_config_tab()

    def create_backtest_tab(self):
        """Backtest Tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📈 Backtest")

        # Title
        title = ttk.Label(tab, text="Run Historical Backtest", font=('Arial', 16, 'bold'))
        title.pack(pady=10)

        # Description
        desc = ttk.Label(tab, text="Analyze strategy performance over historical data with comprehensive analytics",
                        font=('Arial', 10))
        desc.pack(pady=5)

        # Control Frame
        control_frame = ttk.LabelFrame(tab, text="Options", padding=10)
        control_frame.pack(fill='x', padx=20, pady=10)

        # Refresh checkbox
        self.backtest_refresh_var = tk.BooleanVar(value=False)
        refresh_check = ttk.Checkbutton(control_frame, text="Force Refresh Data (ignore cache)",
                                       variable=self.backtest_refresh_var)
        refresh_check.pack(anchor='w', pady=5)

        # Date Range
        date_frame = ttk.Frame(control_frame)
        date_frame.pack(fill='x', pady=5)
        ttk.Label(date_frame, text="Start Date:").pack(side='left')
        self.backtest_start_var = tk.StringVar(value="2021-01-01")
        ttk.Entry(date_frame, textvariable=self.backtest_start_var, width=15).pack(side='left', padx=10)
        ttk.Label(date_frame, text="End Date:").pack(side='left', padx=(20,0))
        self.backtest_end_var = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        ttk.Entry(date_frame, textvariable=self.backtest_end_var, width=15).pack(side='left', padx=10)

        # Initial Balance
        balance_frame = ttk.Frame(control_frame)
        balance_frame.pack(fill='x', pady=5)
        ttk.Label(balance_frame, text="Initial Balance ($):").pack(side='left')
        self.backtest_balance_var = tk.StringVar(value="100000")
        ttk.Entry(balance_frame, textvariable=self.backtest_balance_var, width=15).pack(side='left', padx=10)
        ttk.Label(balance_frame, text="(Portfolio capital for backtest)",
                 font=('Arial', 9), foreground='gray').pack(side='left', padx=10)

        # Rebalance Frequency
        rebal_frame = ttk.Frame(control_frame)
        rebal_frame.pack(fill='x', pady=5)
        ttk.Label(rebal_frame, text="Rebalance Frequency:").pack(side='left')
        self.backtest_rebal_var = tk.StringVar(value="Q")
        ttk.Radiobutton(rebal_frame, text="Quarterly", variable=self.backtest_rebal_var,
                       value="Q").pack(side='left', padx=10)
        ttk.Radiobutton(rebal_frame, text="Monthly", variable=self.backtest_rebal_var,
                       value="M").pack(side='left')

        # Run Button
        self.backtest_run_btn = ttk.Button(control_frame, text="🚀 Run Backtest",
                                          command=self.run_backtest)
        self.backtest_run_btn.pack(pady=10)

        # Progress
        self.backtest_progress = ttk.Progressbar(tab, mode='indeterminate')
        self.backtest_progress.pack(fill='x', padx=20, pady=5)

        # Output
        output_frame = ttk.LabelFrame(tab, text="Output", padding=10)
        output_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.backtest_output = scrolledtext.ScrolledText(output_frame, height=15, font=('Courier', 10))
        self.backtest_output.pack(fill='both', expand=True)

        # Status
        self.backtest_status = ttk.Label(tab, text="Ready", font=('Arial', 10))
        self.backtest_status.pack(pady=5)

    def create_results_tab(self):
        """Results Viewer Tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Results")

        # Title
        title = ttk.Label(tab, text="View Results", font=('Arial', 16, 'bold'))
        title.pack(pady=10)

        # Controls
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill='x', padx=20, pady=10)

        ttk.Button(control_frame, text="📁 Load Latest Portfolio",
                  command=self.load_latest_portfolio).pack(side='left', padx=5)
        ttk.Button(control_frame, text="📁 Load Latest Backtest",
                  command=self.load_latest_backtest).pack(side='left', padx=5)
        ttk.Button(control_frame, text="🖼️ View Charts",
                  command=self.view_charts).pack(side='left', padx=5)
        ttk.Button(control_frame, text="💾 Export to Excel",
                  command=self.export_results).pack(side='left', padx=5)

        # Results Display
        results_frame = ttk.LabelFrame(tab, text="Data", padding=10)
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Create Treeview for tabular data
        self.results_tree = ttk.Treeview(results_frame, show='headings')
        self.results_tree.pack(side='left', fill='both', expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        # Status
        self.results_status = ttk.Label(tab, text="No results loaded", font=('Arial', 10))
        self.results_status.pack(pady=5)

    def create_config_tab(self):
        """Configuration Tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="⚙️ Configuration")

        # Title
        title = ttk.Label(tab, text="Strategy Configuration", font=('Arial', 16, 'bold'))
        title.pack(pady=10)

        # Config Frame
        config_frame = ttk.LabelFrame(tab, text="Parameters", padding=20)
        config_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Create scrollable frame
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Factor Weights
        ttk.Label(scrollable_frame, text="Factor Weights", font=('Arial', 12, 'bold')).pack(anchor='w', pady=5)

        weights_frame = ttk.Frame(scrollable_frame)
        weights_frame.pack(fill='x', pady=5)

        ttk.Label(weights_frame, text="Quality (%):", width=20).grid(row=0, column=0, sticky='w', pady=3)
        self.quality_weight = tk.StringVar(value="35")
        ttk.Entry(weights_frame, textvariable=self.quality_weight, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(weights_frame, text="Growth (%):", width=20).grid(row=1, column=0, sticky='w', pady=3)
        self.growth_weight = tk.StringVar(value="35")
        ttk.Entry(weights_frame, textvariable=self.growth_weight, width=10).grid(row=1, column=1, padx=5)

        ttk.Label(weights_frame, text="Value (%):", width=20).grid(row=2, column=0, sticky='w', pady=3)
        self.value_weight = tk.StringVar(value="15")
        ttk.Entry(weights_frame, textvariable=self.value_weight, width=10).grid(row=2, column=1, padx=5)

        ttk.Label(weights_frame, text="Momentum (%):", width=20).grid(row=3, column=0, sticky='w', pady=3)
        self.momentum_weight = tk.StringVar(value="15")
        ttk.Entry(weights_frame, textvariable=self.momentum_weight, width=10).grid(row=3, column=1, padx=5)

        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill='x', pady=10)

        # Constraints
        ttk.Label(scrollable_frame, text="Portfolio Constraints", font=('Arial', 12, 'bold')).pack(anchor='w', pady=5)

        constraints_frame = ttk.Frame(scrollable_frame)
        constraints_frame.pack(fill='x', pady=5)

        ttk.Label(constraints_frame, text="Max Sector Weight (%):", width=20).grid(row=0, column=0, sticky='w', pady=3)
        self.max_sector = tk.StringVar(value="30")
        ttk.Entry(constraints_frame, textvariable=self.max_sector, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(constraints_frame, text="Min Market Cap ($M):", width=20).grid(row=1, column=0, sticky='w', pady=3)
        self.min_mcap = tk.StringVar(value="300")
        ttk.Entry(constraints_frame, textvariable=self.min_mcap, width=10).grid(row=1, column=1, padx=5)

        ttk.Label(constraints_frame, text="Max Market Cap ($B):", width=20).grid(row=2, column=0, sticky='w', pady=3)
        self.max_mcap = tk.StringVar(value="10")
        ttk.Entry(constraints_frame, textvariable=self.max_mcap, width=10).grid(row=2, column=1, padx=5)

        ttk.Label(constraints_frame, text="Min Avg Volume:", width=20).grid(row=3, column=0, sticky='w', pady=3)
        self.min_volume = tk.StringVar(value="100000")
        ttk.Entry(constraints_frame, textvariable=self.min_volume, width=10).grid(row=3, column=1, padx=5)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Info
        info_text = """
Note: Configuration changes require modifying the script files directly.
These settings are for reference and planning. Future versions will support
dynamic configuration updates.
        """
        ttk.Label(tab, text=info_text, font=('Arial', 9), foreground='gray').pack(pady=10)

    def create_portfolio_tab(self):
        """Portfolio Manager Tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="💼 Portfolio")

        # Title
        title = ttk.Label(tab, text="Portfolio Manager", font=('Arial', 16, 'bold'))
        title.pack(pady=10)

        # Description
        desc = ttk.Label(tab, text="Track your actual holdings with real-time prices and performance",
                        font=('Arial', 10))
        desc.pack(pady=5)

        # Control Frame
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill='x', padx=20, pady=10)

        # Top row buttons
        button_frame1 = ttk.Frame(control_frame)
        button_frame1.pack(fill='x', pady=5)

        ttk.Button(button_frame1, text="📁 Load Recommended",
                  command=self.load_recommended_portfolio_handler).pack(side='left', padx=5)
        ttk.Button(button_frame1, text="💾 Import Portfolio",
                  command=self.import_portfolio_handler).pack(side='left', padx=5)
        ttk.Button(button_frame1, text="🔄 Refresh Prices",
                  command=self.refresh_prices_handler).pack(side='left', padx=5)
        ttk.Button(button_frame1, text="💾 Save",
                  command=self.save_portfolio_handler).pack(side='left', padx=5)

        # Bottom row buttons
        button_frame2 = ttk.Frame(control_frame)
        button_frame2.pack(fill='x', pady=5)

        ttk.Button(button_frame2, text="➕ Add Position",
                  command=self.add_position_handler).pack(side='left', padx=5)
        ttk.Button(button_frame2, text="✏️ Edit Selected",
                  command=self.edit_position_handler).pack(side='left', padx=5)
        ttk.Button(button_frame2, text="➖ Remove Selected",
                  command=self.remove_position_handler).pack(side='left', padx=5)
        ttk.Button(button_frame2, text="📸 Take Snapshot",
                  command=self.take_snapshot_handler).pack(side='left', padx=5)

        # Summary Panel
        summary_frame = ttk.LabelFrame(tab, text="Portfolio Summary", padding=10)
        summary_frame.pack(fill='x', padx=20, pady=10)

        # Summary labels (will be updated dynamically)
        self.portfolio_summary_labels = {}

        summary_grid = ttk.Frame(summary_frame)
        summary_grid.pack(fill='x')

        # Row 1
        row1 = ttk.Frame(summary_grid)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="Total Value:", font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        self.portfolio_summary_labels['total_value'] = ttk.Label(row1, text="$0.00", font=('Arial', 10))
        self.portfolio_summary_labels['total_value'].pack(side='left', padx=5)

        ttk.Label(row1, text="Cost Basis:", font=('Arial', 10, 'bold')).pack(side='left', padx=(20, 5))
        self.portfolio_summary_labels['cost_basis'] = ttk.Label(row1, text="$0.00", font=('Arial', 10))
        self.portfolio_summary_labels['cost_basis'].pack(side='left', padx=5)

        # Row 2
        row2 = ttk.Frame(summary_grid)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="Unrealized P&L:", font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        self.portfolio_summary_labels['unrealized_pl'] = ttk.Label(row2, text="$0.00 (0.00%)", font=('Arial', 10))
        self.portfolio_summary_labels['unrealized_pl'].pack(side='left', padx=5)

        ttk.Label(row2, text="Positions:", font=('Arial', 10, 'bold')).pack(side='left', padx=(20, 5))
        self.portfolio_summary_labels['num_positions'] = ttk.Label(row2, text="0", font=('Arial', 10))
        self.portfolio_summary_labels['num_positions'].pack(side='left', padx=5)

        # Row 3
        row3 = ttk.Frame(summary_grid)
        row3.pack(fill='x', pady=2)
        ttk.Label(row3, text="Last Updated:", font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        self.portfolio_summary_labels['last_updated'] = ttk.Label(row3, text="Never", font=('Arial', 10))
        self.portfolio_summary_labels['last_updated'].pack(side='left', padx=5)

        # Holdings Table Frame
        table_frame = ttk.LabelFrame(tab, text="Holdings", padding=10)
        table_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Create Treeview for holdings
        columns = ('Ticker', 'Sector', 'Entry_Date', 'Entry_Price', 'Shares', 'Cost_Basis',
                  'Current_Price', 'Current_Value', 'PL_Dollar', 'PL_Percent')

        self.portfolio_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)

        # Define column headings
        self.portfolio_tree.heading('Ticker', text='Ticker')
        self.portfolio_tree.heading('Sector', text='Sector')
        self.portfolio_tree.heading('Entry_Date', text='Entry Date')
        self.portfolio_tree.heading('Entry_Price', text='Entry $')
        self.portfolio_tree.heading('Shares', text='Shares')
        self.portfolio_tree.heading('Cost_Basis', text='Cost Basis')
        self.portfolio_tree.heading('Current_Price', text='Current $')
        self.portfolio_tree.heading('Current_Value', text='Current Value')
        self.portfolio_tree.heading('PL_Dollar', text='P&L ($)')
        self.portfolio_tree.heading('PL_Percent', text='P&L (%)')

        # Define column widths
        self.portfolio_tree.column('Ticker', width=60, anchor='center')
        self.portfolio_tree.column('Sector', width=100)
        self.portfolio_tree.column('Entry_Date', width=80, anchor='center')
        self.portfolio_tree.column('Entry_Price', width=70, anchor='e')
        self.portfolio_tree.column('Shares', width=60, anchor='e')
        self.portfolio_tree.column('Cost_Basis', width=90, anchor='e')
        self.portfolio_tree.column('Current_Price', width=70, anchor='e')
        self.portfolio_tree.column('Current_Value', width=100, anchor='e')
        self.portfolio_tree.column('PL_Dollar', width=90, anchor='e')
        self.portfolio_tree.column('PL_Percent', width=70, anchor='e')

        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.portfolio_tree.yview)
        self.portfolio_tree.configure(yscrollcommand=scrollbar.set)

        self.portfolio_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Bind double-click to edit position
        self.portfolio_tree.bind('<Double-1>', lambda e: self.edit_position_handler())

        # Charts Panel (placeholder for Phase 6)
        charts_frame = ttk.LabelFrame(tab, text="Charts", padding=10)
        charts_frame.pack(fill='both', expand=False, padx=20, pady=10)

        # Create notebook for chart tabs
        self.portfolio_charts_notebook = ttk.Notebook(charts_frame)
        self.portfolio_charts_notebook.pack(fill='both', expand=True)

        # Chart tabs (empty for now, will be populated in Phase 6)
        self.chart_tab_performance = ttk.Frame(self.portfolio_charts_notebook)
        self.portfolio_charts_notebook.add(self.chart_tab_performance, text="Performance")

        self.chart_tab_sector = ttk.Frame(self.portfolio_charts_notebook)
        self.portfolio_charts_notebook.add(self.chart_tab_sector, text="Sector Allocation")

        self.chart_tab_positions = ttk.Frame(self.portfolio_charts_notebook)
        self.portfolio_charts_notebook.add(self.chart_tab_positions, text="Position Details")

        # Placeholder labels in chart tabs
        ttk.Label(self.chart_tab_performance, text="Performance chart will appear here",
                 font=('Arial', 10), foreground='gray').pack(expand=True)
        ttk.Label(self.chart_tab_sector, text="Sector allocation chart will appear here",
                 font=('Arial', 10), foreground='gray').pack(expand=True)
        ttk.Label(self.chart_tab_positions, text="Position details chart will appear here",
                 font=('Arial', 10), foreground='gray').pack(expand=True)

        # Status label
        self.portfolio_status = ttk.Label(tab, text="Ready", font=('Arial', 10))
        self.portfolio_status.pack(pady=5)

        # Setup column sorting
        self.sort_column = None
        self.sort_reverse = False

        # Initial update
        self.update_portfolio_display()

    def run_backtest(self):
        """Run backtest in separate thread"""
        def run():
            self.backtest_run_btn.config(state='disabled')
            self.backtest_progress.start()
            self.backtest_status.config(text="Running... (this may take several minutes)")
            self.backtest_output.delete(1.0, tk.END)

            try:
                cmd = ['python', 'russell2000_backtest.py']
                if self.backtest_refresh_var.get():
                    cmd.append('--refresh')

                # Add initial balance
                try:
                    balance = float(self.backtest_balance_var.get())
                    cmd.extend(['--initial-balance', str(balance)])
                except ValueError:
                    messagebox.showerror("Error", "Invalid initial balance. Please enter a number.")
                    self.backtest_progress.stop()
                    self.backtest_run_btn.config(state='normal')
                    return

                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, bufsize=1, universal_newlines=True)

                for line in process.stdout:
                    self.backtest_output.insert(tk.END, line)
                    self.backtest_output.see(tk.END)
                    self.root.update_idletasks()

                process.wait()

                if process.returncode == 0:
                    self.backtest_status.config(text="✅ Completed Successfully")
                    messagebox.showinfo("Success", "Backtest completed successfully!\nCheck results_*/ folder")
                else:
                    self.backtest_status.config(text="❌ Error occurred")
                    messagebox.showerror("Error", "Backtest failed. Check output for details.")

            except Exception as e:
                self.backtest_output.insert(tk.END, f"\n\nError: {str(e)}\n")
                self.backtest_status.config(text="❌ Error occurred")
                messagebox.showerror("Error", str(e))
            finally:
                self.backtest_progress.stop()
                self.backtest_run_btn.config(state='normal')

        threading.Thread(target=run, daemon=True).start()

    def load_latest_portfolio(self):
        """Load latest portfolio CSV from backtest results"""
        try:
            folders = glob.glob('results_*/')
            if not folders:
                messagebox.showwarning("No Results", "No backtest results found. Run a backtest first.")
                return

            latest_folder = max(folders, key=os.path.getctime)
            portfolio_file = os.path.join(latest_folder, 'current_portfolio.csv')

            if not os.path.exists(portfolio_file):
                messagebox.showwarning("No Portfolio", "Portfolio file not found in results folder.")
                return

            df = pd.read_csv(portfolio_file)
            self.display_dataframe(df, f"Portfolio: {portfolio_file}")
            self.results_status.config(text=f"Loaded: {portfolio_file}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load portfolio: {str(e)}")

    def load_latest_backtest(self):
        """Load latest backtest results"""
        try:
            folders = glob.glob('results_*/')
            if not folders:
                messagebox.showwarning("No Results", "No backtest results found. Run a backtest first.")
                return

            latest_folder = max(folders, key=os.path.getctime)

            # Ask user which file to load
            files = {
                'Current Portfolio': 'current_portfolio.csv',
                'Quarterly Summary': 'quarterly_summary.csv',
                'Rebalancing Details': 'rebalancing_details.csv',
                'Factor Scores': 'factor_scores.csv'
            }

            dialog = tk.Toplevel(self.root)
            dialog.title("Select File")
            dialog.geometry("300x200")

            ttk.Label(dialog, text="Select file to view:", font=('Arial', 12)).pack(pady=10)

            for name, filename in files.items():
                btn = ttk.Button(dialog, text=name,
                                command=lambda f=filename, d=dialog: self.load_backtest_file(latest_folder, f, d))
                btn.pack(pady=5, padx=20, fill='x')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load results: {str(e)}")

    def load_backtest_file(self, folder, filename, dialog):
        """Load specific backtest file"""
        try:
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)
            self.display_dataframe(df, f"{folder}/{filename}")
            self.results_status.config(text=f"Loaded: {filepath}")
            dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def display_dataframe(self, df, title):
        """Display dataframe in treeview"""
        # Clear existing
        self.results_tree.delete(*self.results_tree.get_children())

        # Configure columns
        self.results_tree['columns'] = list(df.columns)
        for col in df.columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)

        # Add data
        for idx, row in df.iterrows():
            values = [row[col] for col in df.columns]
            self.results_tree.insert('', tk.END, values=values)

    def view_charts(self):
        """Open chart viewer"""
        try:
            folders = glob.glob('results_*/')
            if not folders:
                messagebox.showwarning("No Results", "No backtest results found. Run a backtest first.")
                return

            latest_folder = max(folders, key=os.path.getctime)

            # Find PNG files
            charts = glob.glob(os.path.join(latest_folder, '*.png'))
            if not charts:
                messagebox.showwarning("No Charts", "No charts found in results folder.")
                return

            # Create chart viewer window
            viewer = tk.Toplevel(self.root)
            viewer.title("Charts Viewer")
            viewer.geometry("1000x800")

            # Create notebook for multiple charts
            chart_notebook = ttk.Notebook(viewer)
            chart_notebook.pack(fill='both', expand=True, padx=10, pady=10)

            for chart_path in sorted(charts):
                chart_name = os.path.basename(chart_path).replace('.png', '').replace('_', ' ').title()

                # Create tab
                tab = ttk.Frame(chart_notebook)
                chart_notebook.add(tab, text=chart_name)

                # Open and display image
                try:
                    img = Image.open(chart_path)
                    img.thumbnail((950, 750), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)

                    label = ttk.Label(tab, image=photo)
                    label.image = photo  # Keep reference
                    label.pack(pady=10)
                except Exception as e:
                    ttk.Label(tab, text=f"Error loading image: {str(e)}").pack(pady=20)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to view charts: {str(e)}")

    def export_results(self):
        """Export results to Excel"""
        try:
            # Ask for save location
            filepath = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile=f"quant_strategy_export_{datetime.now().strftime('%Y%m%d')}.xlsx"
            )

            if not filepath:
                return

            # Find latest results
            folders = glob.glob('results_*/')
            if not folders:
                messagebox.showwarning("No Results", "No backtest results found.")
                return

            latest_folder = max(folders, key=os.path.getctime)

            # Load all CSV files
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                csv_files = glob.glob(os.path.join(latest_folder, '*.csv'))

                for csv_file in csv_files:
                    sheet_name = os.path.basename(csv_file).replace('.csv', '')[:31]  # Excel limit
                    df = pd.read_csv(csv_file)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            messagebox.showinfo("Success", f"Exported to:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")

    # ===================================================================
    # Portfolio Manager Methods
    # ===================================================================

    def update_portfolio_display(self):
        """Update portfolio summary and holdings table"""
        # Update summary labels
        summary = self.portfolio_manager.calculate_portfolio_summary()

        self.portfolio_summary_labels['total_value'].config(
            text=f"${summary['total_value']:,.2f}"
        )
        self.portfolio_summary_labels['cost_basis'].config(
            text=f"${summary['total_cost_basis']:,.2f}"
        )

        # Color-code P&L
        pl_color = 'green' if summary['total_unrealized_pl'] >= 0 else 'red'
        self.portfolio_summary_labels['unrealized_pl'].config(
            text=f"${summary['total_unrealized_pl']:,.2f} ({summary['total_return_pct']:.2f}%)",
            foreground=pl_color
        )

        self.portfolio_summary_labels['num_positions'].config(
            text=str(summary['num_positions'])
        )

        # Update last updated timestamp
        if len(self.portfolio_manager.holdings) > 0:
            last_updated = self.portfolio_manager.holdings['Last_Updated'].iloc[0]
            self.portfolio_summary_labels['last_updated'].config(text=last_updated)
        else:
            self.portfolio_summary_labels['last_updated'].config(text="Never")

        # Update holdings table
        self.update_holdings_table()

        # Update status
        self.portfolio_status.config(text=f"Last refreshed: {summary['num_positions']} positions")

        # Update charts
        self.update_portfolio_charts()

    def update_holdings_table(self):
        """Refresh holdings Treeview with current data"""
        # Clear existing rows
        for item in self.portfolio_tree.get_children():
            self.portfolio_tree.delete(item)

        # Populate with current holdings
        for idx, row in self.portfolio_manager.holdings.iterrows():
            values = (
                row['Ticker'],
                row['Sector'],
                row['Entry_Date'],
                f"${row['Entry_Price']:.2f}",
                int(row['Shares_Owned']),
                f"${row['Cost_Basis']:,.2f}",
                f"${row['Current_Price']:.2f}",
                f"${row['Current_Value']:,.2f}",
                f"${row['Unrealized_PL']:,.2f}",
                f"{row['Unrealized_PL_Pct']:.2f}%"
            )

            # Insert row with tag for coloring
            tag = 'profit' if row['Unrealized_PL'] >= 0 else 'loss'
            self.portfolio_tree.insert('', tk.END, values=values, tags=(tag,))

        # Configure row colors
        self.portfolio_tree.tag_configure('profit', foreground='green')
        self.portfolio_tree.tag_configure('loss', foreground='red')

        # Setup sorting if not already done
        if not hasattr(self, 'sorting_setup'):
            self.setup_column_sorting()
            self.sorting_setup = True

    def setup_column_sorting(self):
        """Enable column sorting by clicking headers"""
        # Sorting state
        self.sort_column = None
        self.sort_reverse = False

        def sort_by_column(col):
            """Sort treeview by column"""
            # Get all data
            data = [(self.portfolio_tree.set(child, col), child) for child in self.portfolio_tree.get_children('')]

            # Determine sort order
            if self.sort_column == col:
                self.sort_reverse = not self.sort_reverse
            else:
                self.sort_reverse = False
                self.sort_column = col

            # Sort data
            try:
                # Try numeric sort
                data.sort(key=lambda x: float(x[0].replace('$', '').replace(',', '').replace('%', '')),
                         reverse=self.sort_reverse)
            except:
                # Fall back to string sort
                data.sort(key=lambda x: x[0], reverse=self.sort_reverse)

            # Rearrange items
            for index, (_, child) in enumerate(data):
                self.portfolio_tree.move(child, '', index)

        # Bind click event to all column headers
        for col in self.portfolio_tree['columns']:
            self.portfolio_tree.heading(col, command=lambda c=col: sort_by_column(c))

    def load_recommended_portfolio_handler(self):
        """Load recommended portfolio from latest backtest results"""
        # Get recommended portfolio
        recommended_df = self.portfolio_manager.get_recommended_portfolio()

        if recommended_df is None:
            messagebox.showerror("Error", "No backtest results found.\nPlease run a backtest first.")
            return

        # Open import dialog
        self.open_import_dialog(recommended_df)

    def import_portfolio_handler(self):
        """Import existing portfolio from CSV file"""
        self.portfolio_manager.load_portfolio()
        self.update_portfolio_display()
        messagebox.showinfo("Success", f"Loaded {len(self.portfolio_manager.holdings)} positions")

    def refresh_prices_handler(self):
        """Refresh current prices for all holdings"""
        if len(self.portfolio_manager.holdings) == 0:
            messagebox.showwarning("No Holdings", "No positions to refresh")
            return

        # Run in separate thread to avoid blocking GUI
        def refresh_thread():
            # Create progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Refreshing Prices")
            progress_dialog.geometry("400x150")
            progress_dialog.transient(self.root)
            progress_dialog.grab_set()

            ttk.Label(progress_dialog, text="Updating prices...", font=('Arial', 12, 'bold')).pack(pady=10)

            # Progress bar
            progress_var = tk.IntVar()
            progress_bar = ttk.Progressbar(progress_dialog, variable=progress_var, maximum=100, length=350)
            progress_bar.pack(pady=10)

            # Status label
            status_label = ttk.Label(progress_dialog, text="Starting...", font=('Arial', 10))
            status_label.pack(pady=5)

            def progress_callback(current, total, ticker):
                """Update progress bar and label"""
                percent = int((current / total) * 100)
                progress_var.set(percent)
                status_label.config(text=f"Updating {ticker} ({current}/{total})...")
                progress_dialog.update()

            # Refresh prices
            result = self.portfolio_manager.refresh_all_prices(progress_callback=progress_callback)

            # Update display
            self.update_portfolio_display()

            # Close progress dialog
            progress_dialog.destroy()

            # Show result
            msg = f"Price refresh complete!\n\n"
            msg += f"✓ {result['success']} succeeded\n"
            if result['failed'] > 0:
                msg += f"✗ {result['failed']} failed\n"
            msg += f"\nTotal: {result['total']} positions"

            messagebox.showinfo("Refresh Complete", msg)

        # Start thread
        threading.Thread(target=refresh_thread, daemon=True).start()

    def save_portfolio_handler(self):
        """Save portfolio to CSV"""
        success = self.portfolio_manager.save_portfolio()
        if success:
            messagebox.showinfo("Success", f"Portfolio saved to {self.portfolio_manager.portfolio_file}")
        else:
            messagebox.showerror("Error", "Failed to save portfolio")

    def add_position_handler(self):
        """Open dialog to manually add a position"""
        self.open_add_position_dialog()

    def edit_position_handler(self):
        """Open dialog to edit selected position"""
        selected = self.portfolio_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a position to edit")
            return

        # Get ticker from selected row
        item = self.portfolio_tree.item(selected[0])
        ticker = item['values'][0]

        # Get full position data from portfolio manager
        position = self.portfolio_manager.holdings[
            self.portfolio_manager.holdings['Ticker'] == ticker
        ]

        if len(position) == 0:
            messagebox.showerror("Error", f"Position {ticker} not found")
            return

        position = position.iloc[0]

        # Open edit dialog
        self.open_edit_position_dialog(position)

    def remove_position_handler(self):
        """Remove selected position from portfolio"""
        selected = self.portfolio_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a position to remove")
            return

        # Get ticker from selected row
        item = self.portfolio_tree.item(selected[0])
        ticker = item['values'][0]

        # Confirm deletion
        confirm = messagebox.askyesno("Confirm Removal",
                                     f"Remove {ticker} from portfolio?")
        if confirm:
            success = self.portfolio_manager.remove_position(ticker)
            if success:
                self.update_portfolio_display()
                messagebox.showinfo("Success", f"Removed {ticker} from portfolio")

    def update_portfolio_charts(self):
        """Update all charts in the charts panel"""
        self.plot_performance_chart()
        self.plot_sector_allocation()
        self.plot_position_details()

    def plot_performance_chart(self):
        """Plot historical performance chart"""
        # Clear existing
        for widget in self.chart_tab_performance.winfo_children():
            widget.destroy()

        if len(self.portfolio_manager.history) == 0:
            ttk.Label(self.chart_tab_performance, text="No historical data yet. Take snapshots to see performance over time.",
                     font=('Arial', 10), foreground='gray').pack(expand=True)
            return

        # Create matplotlib figure
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Plot data
        history = self.portfolio_manager.history
        dates = pd.to_datetime(history['Date'])
        values = history['Total_Value']
        cost_basis = history['Total_Cost_Basis']

        ax.plot(dates, values, marker='o', linewidth=2, markersize=6, color='#2E86AB', label='Portfolio Value')
        ax.plot(dates, cost_basis, linestyle='--', linewidth=1.5, color='gray', alpha=0.7, label='Cost Basis')

        # Fill area between lines
        ax.fill_between(dates, values, cost_basis,
                        where=(values >= cost_basis), color='green', alpha=0.2, label='Profit')
        ax.fill_between(dates, values, cost_basis,
                        where=(values < cost_basis), color='red', alpha=0.2, label='Loss')

        ax.set_title('Portfolio Performance Over Time', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_tab_performance)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_sector_allocation(self):
        """Plot sector allocation pie chart"""
        # Clear existing
        for widget in self.chart_tab_sector.winfo_children():
            widget.destroy()

        if len(self.portfolio_manager.holdings) == 0:
            ttk.Label(self.chart_tab_sector, text="No positions to display",
                     font=('Arial', 10), foreground='gray').pack(expand=True)
            return

        # Group by sector
        holdings = self.portfolio_manager.holdings
        sector_values = holdings.groupby('Sector')['Current_Value'].sum()

        if len(sector_values) == 0:
            ttk.Label(self.chart_tab_sector, text="No sector data available",
                     font=('Arial', 10), foreground='gray').pack(expand=True)
            return

        # Create matplotlib figure
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Pie chart
        colors = plt.cm.Set3(range(len(sector_values)))
        wedges, texts, autotexts = ax.pie(sector_values, labels=sector_values.index,
                                           autopct='%1.1f%%', startangle=90, colors=colors)

        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('Portfolio Allocation by Sector', fontweight='bold')
        ax.axis('equal')

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_tab_sector)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_position_details(self):
        """Plot position details bar chart"""
        # Clear existing
        for widget in self.chart_tab_positions.winfo_children():
            widget.destroy()

        if len(self.portfolio_manager.holdings) == 0:
            ttk.Label(self.chart_tab_positions, text="No positions to display",
                     font=('Arial', 10), foreground='gray').pack(expand=True)
            return

        # Sort by P&L percentage
        holdings = self.portfolio_manager.holdings.sort_values('Unrealized_PL_Pct', ascending=False)

        # Create matplotlib figure
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Bar chart
        tickers = holdings['Ticker']
        pl_pct = holdings['Unrealized_PL_Pct']

        colors = ['green' if x >= 0 else 'red' for x in pl_pct]
        bars = ax.bar(range(len(tickers)), pl_pct, color=colors, alpha=0.7)

        ax.set_title('P&L by Position', fontweight='bold')
        ax.set_xlabel('Ticker')
        ax.set_ylabel('P&L (%)')
        ax.set_xticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_tab_positions)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def take_snapshot_handler(self):
        """Take snapshot of current portfolio for historical tracking"""
        if len(self.portfolio_manager.holdings) == 0:
            messagebox.showwarning("No Holdings", "No positions to snapshot")
            return

        # Take snapshot
        success = self.portfolio_manager.append_snapshot()

        if success:
            # Show summary
            summary = self.portfolio_manager.calculate_portfolio_summary()
            today = datetime.now().strftime('%Y-%m-%d')

            msg = f"Snapshot saved for {today}\n\n"
            msg += f"Total Value: ${summary['total_value']:,.2f}\n"
            msg += f"Cost Basis: ${summary['total_cost_basis']:,.2f}\n"
            msg += f"P&L: ${summary['total_unrealized_pl']:,.2f} ({summary['total_return_pct']:.2f}%)\n"
            msg += f"Positions: {summary['num_positions']}\n\n"
            msg += f"Total snapshots: {len(self.portfolio_manager.history)}"

            messagebox.showinfo("Snapshot Saved", msg)
        else:
            messagebox.showerror("Error", "Failed to save snapshot")

    def open_import_dialog(self, recommended_df):
        """
        Open dialog to import recommended positions with adjustments

        Args:
            recommended_df: DataFrame with recommended positions from backtest
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Import Recommended Portfolio")
        dialog.geometry("900x600")

        # Title
        ttk.Label(dialog, text="Adjust Recommended Positions", font=('Arial', 14, 'bold')).pack(pady=10)

        # Instructions
        instructions = ("Edit the 'Actual Shares' and 'Actual Price' columns below.\n"
                       "Only positions with shares > 0 will be imported.")
        ttk.Label(dialog, text=instructions, font=('Arial', 10)).pack(pady=5)

        # Create frame for table
        table_frame = ttk.Frame(dialog)
        table_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Create Treeview for editing
        columns = ('Ticker', 'Sector', 'Rec_Shares', 'Rec_Price', 'Act_Shares', 'Act_Price', 'Entry_Date')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)

        # Column headings
        tree.heading('Ticker', text='Ticker')
        tree.heading('Sector', text='Sector')
        tree.heading('Rec_Shares', text='Rec. Shares')
        tree.heading('Rec_Price', text='Rec. Price')
        tree.heading('Act_Shares', text='Actual Shares')
        tree.heading('Act_Price', text='Actual Price')
        tree.heading('Entry_Date', text='Entry Date')

        # Column widths
        tree.column('Ticker', width=60, anchor='center')
        tree.column('Sector', width=100)
        tree.column('Rec_Shares', width=80, anchor='e')
        tree.column('Rec_Price', width=80, anchor='e')
        tree.column('Act_Shares', width=100, anchor='e')
        tree.column('Act_Price', width=100, anchor='e')
        tree.column('Entry_Date', width=100, anchor='center')

        # Populate with recommended positions
        today = datetime.now().strftime('%Y-%m-%d')
        for idx, row in recommended_df.iterrows():
            rec_shares = int(row.get('Shares_to_Buy', 0))
            rec_price = row.get('Current_Price', 0)

            values = (
                row['Ticker'],
                row.get('Sector', ''),
                rec_shares,
                f"${rec_price:.2f}",
                rec_shares,  # Default to recommended
                f"${rec_price:.2f}",  # Default to recommended
                today
            )
            tree.insert('', tk.END, values=values)

        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Edit entry (simple approach: double-click to edit)
        ttk.Label(dialog, text="Tip: Double-click a row to edit actual shares and price",
                 font=('Arial', 9), foreground='gray').pack(pady=5)

        # Store editable data
        editable_data = {}
        for item_id in tree.get_children():
            item = tree.item(item_id)
            ticker = item['values'][0]
            editable_data[ticker] = {
                'rec_shares': item['values'][2],
                'rec_price': float(item['values'][3].replace('$', '')),
                'act_shares': item['values'][2],
                'act_price': float(item['values'][5].replace('$', '')),
                'entry_date': item['values'][6],
                'sector': item['values'][1]
            }

        def on_double_click(event):
            """Handle double-click to edit row"""
            selected = tree.selection()
            if not selected:
                return

            item = tree.item(selected[0])
            ticker = item['values'][0]

            # Open edit dialog
            edit_dialog = tk.Toplevel(dialog)
            edit_dialog.title(f"Edit {ticker}")
            edit_dialog.geometry("400x250")

            ttk.Label(edit_dialog, text=f"Edit Position: {ticker}", font=('Arial', 12, 'bold')).pack(pady=10)

            # Form
            form_frame = ttk.Frame(edit_dialog)
            form_frame.pack(padx=20, pady=10)

            ttk.Label(form_frame, text="Actual Shares:").grid(row=0, column=0, sticky='w', pady=5)
            shares_var = tk.StringVar(value=str(editable_data[ticker]['act_shares']))
            shares_entry = ttk.Entry(form_frame, textvariable=shares_var, width=15)
            shares_entry.grid(row=0, column=1, padx=10, pady=5)

            ttk.Label(form_frame, text="Actual Price ($):").grid(row=1, column=0, sticky='w', pady=5)
            price_var = tk.StringVar(value=f"{editable_data[ticker]['act_price']:.2f}")
            price_entry = ttk.Entry(form_frame, textvariable=price_var, width=15)
            price_entry.grid(row=1, column=1, padx=10, pady=5)

            ttk.Label(form_frame, text="Entry Date:").grid(row=2, column=0, sticky='w', pady=5)
            date_var = tk.StringVar(value=editable_data[ticker]['entry_date'])
            date_entry = ttk.Entry(form_frame, textvariable=date_var, width=15)
            date_entry.grid(row=2, column=1, padx=10, pady=5)

            def save_edit():
                try:
                    shares = int(shares_var.get())
                    price = float(price_var.get())
                    date = date_var.get()

                    # Update editable data
                    editable_data[ticker]['act_shares'] = shares
                    editable_data[ticker]['act_price'] = price
                    editable_data[ticker]['entry_date'] = date

                    # Update tree display
                    tree.item(selected[0], values=(
                        ticker,
                        editable_data[ticker]['sector'],
                        editable_data[ticker]['rec_shares'],
                        f"${editable_data[ticker]['rec_price']:.2f}",
                        shares,
                        f"${price:.2f}",
                        date
                    ))

                    edit_dialog.destroy()
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid numbers for shares and price")

            ttk.Button(form_frame, text="Save", command=save_edit).grid(row=3, column=0, columnspan=2, pady=10)

        tree.bind('<Double-1>', on_double_click)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        def import_positions():
            """Import positions with user adjustments"""
            adjustments = {}

            for ticker, data in editable_data.items():
                if data['act_shares'] > 0:
                    adjustments[ticker] = {
                        'shares': data['act_shares'],
                        'price': data['act_price'],
                        'date': data['entry_date']
                    }

            if not adjustments:
                messagebox.showwarning("No Positions", "No positions to import (all have 0 shares)")
                return

            # Import using portfolio manager
            imported_count = self.portfolio_manager.import_from_recommended(recommended_df, adjustments)

            # Update display
            self.update_portfolio_display()

            # Close dialog
            dialog.destroy()

            messagebox.showinfo("Success", f"Imported {imported_count} positions to portfolio")

        ttk.Button(button_frame, text="Import", command=import_positions, width=15).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy, width=15).pack(side='left', padx=5)

    def open_add_position_dialog(self):
        """Open dialog to manually add a position"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Position")
        dialog.geometry("400x350")

        ttk.Label(dialog, text="Add New Position", font=('Arial', 14, 'bold')).pack(pady=10)

        # Form
        form_frame = ttk.Frame(dialog)
        form_frame.pack(padx=20, pady=10)

        ttk.Label(form_frame, text="Ticker:").grid(row=0, column=0, sticky='w', pady=5)
        ticker_var = tk.StringVar()
        ticker_entry = ttk.Entry(form_frame, textvariable=ticker_var, width=20)
        ticker_entry.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(form_frame, text="Sector:").grid(row=1, column=0, sticky='w', pady=5)
        sector_var = tk.StringVar()
        sector_entry = ttk.Entry(form_frame, textvariable=sector_var, width=20)
        sector_entry.grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(form_frame, text="Entry Date:").grid(row=2, column=0, sticky='w', pady=5)
        date_var = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        date_entry = ttk.Entry(form_frame, textvariable=date_var, width=20)
        date_entry.grid(row=2, column=1, padx=10, pady=5)

        ttk.Label(form_frame, text="Entry Price ($):").grid(row=3, column=0, sticky='w', pady=5)
        price_var = tk.StringVar()
        price_entry = ttk.Entry(form_frame, textvariable=price_var, width=20)
        price_entry.grid(row=3, column=1, padx=10, pady=5)

        ttk.Label(form_frame, text="Shares:").grid(row=4, column=0, sticky='w', pady=5)
        shares_var = tk.StringVar()
        shares_entry = ttk.Entry(form_frame, textvariable=shares_var, width=20)
        shares_entry.grid(row=4, column=1, padx=10, pady=5)

        # Status label
        status_label = ttk.Label(dialog, text="", font=('Arial', 9), foreground='blue')
        status_label.pack(pady=5)

        def add_position():
            """Add position with validation"""
            try:
                ticker = ticker_var.get().strip().upper()
                sector = sector_var.get().strip()
                entry_date = date_var.get().strip()
                entry_price = float(price_var.get())
                shares = int(shares_var.get())

                if not ticker:
                    messagebox.showerror("Invalid Input", "Ticker is required")
                    return

                if shares <= 0 or entry_price <= 0:
                    messagebox.showerror("Invalid Input", "Shares and price must be positive")
                    return

                # Add position
                status_label.config(text=f"Adding {ticker}...")
                dialog.update()

                success = self.portfolio_manager.add_position(
                    ticker=ticker,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    shares=shares,
                    sector=sector
                )

                if success:
                    # Update display
                    self.update_portfolio_display()

                    # Close dialog
                    dialog.destroy()

                    messagebox.showinfo("Success", f"Added {ticker} to portfolio")
                else:
                    status_label.config(text="Failed to add position", foreground='red')

            except ValueError as e:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for price and shares")

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Add", command=add_position, width=15).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy, width=15).pack(side='left', padx=5)

    def open_edit_position_dialog(self, position):
        """
        Open dialog to edit an existing position

        Args:
            position: pd.Series with current position data
        """
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit Position - {position['Ticker']}")
        dialog.geometry("400x350")

        ttk.Label(dialog, text=f"Edit Position: {position['Ticker']}", font=('Arial', 14, 'bold')).pack(pady=10)

        # Form
        form_frame = ttk.Frame(dialog)
        form_frame.pack(padx=20, pady=10)

        # Ticker (read-only, display only)
        ttk.Label(form_frame, text="Ticker:").grid(row=0, column=0, sticky='w', pady=5)
        ticker_label = ttk.Label(form_frame, text=position['Ticker'], font=('Arial', 10, 'bold'))
        ticker_label.grid(row=0, column=1, padx=10, pady=5, sticky='w')

        ttk.Label(form_frame, text="Sector:").grid(row=1, column=0, sticky='w', pady=5)
        sector_var = tk.StringVar(value=position['Sector'])
        sector_entry = ttk.Entry(form_frame, textvariable=sector_var, width=20)
        sector_entry.grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(form_frame, text="Entry Date:").grid(row=2, column=0, sticky='w', pady=5)
        date_var = tk.StringVar(value=position['Entry_Date'])
        date_entry = ttk.Entry(form_frame, textvariable=date_var, width=20)
        date_entry.grid(row=2, column=1, padx=10, pady=5)

        ttk.Label(form_frame, text="Entry Price ($):").grid(row=3, column=0, sticky='w', pady=5)
        price_var = tk.StringVar(value=f"{position['Entry_Price']:.2f}")
        price_entry = ttk.Entry(form_frame, textvariable=price_var, width=20)
        price_entry.grid(row=3, column=1, padx=10, pady=5)

        ttk.Label(form_frame, text="Shares:").grid(row=4, column=0, sticky='w', pady=5)
        shares_var = tk.StringVar(value=str(int(position['Shares_Owned'])))
        shares_entry = ttk.Entry(form_frame, textvariable=shares_var, width=20)
        shares_entry.grid(row=4, column=1, padx=10, pady=5)

        # Status label
        status_label = ttk.Label(dialog, text="", font=('Arial', 9), foreground='blue')
        status_label.pack(pady=5)

        def update_position():
            """Update position with validation"""
            try:
                ticker = position['Ticker']
                sector = sector_var.get().strip()
                entry_date = date_var.get().strip()
                entry_price = float(price_var.get())
                shares = int(shares_var.get())

                if shares <= 0 or entry_price <= 0:
                    messagebox.showerror("Invalid Input", "Shares and price must be positive")
                    return

                # Update position in portfolio manager
                status_label.config(text=f"Updating {ticker}...")
                dialog.update()

                success = self.portfolio_manager.update_position(
                    ticker=ticker,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    shares=shares,
                    sector=sector
                )

                if success:
                    # Update display
                    self.update_portfolio_display()

                    # Close dialog
                    dialog.destroy()

                    messagebox.showinfo("Success", f"Updated {ticker}")
                else:
                    status_label.config(text="Failed to update position", foreground='red')

            except ValueError as e:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for price and shares")

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Update", command=update_position, width=15).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy, width=15).pack(side='left', padx=5)


def main():
    root = tk.Tk()
    app = QuantStrategyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
