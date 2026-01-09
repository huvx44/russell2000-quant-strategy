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

    def load_recommended_portfolio_handler(self):
        """Load recommended portfolio from latest backtest results"""
        messagebox.showinfo("Coming Soon", "This feature will be implemented in Phase 3:\n"
                           "- Load current_portfolio.csv from latest results folder\n"
                           "- Open dialog to adjust actual shares/prices\n"
                           "- Import to portfolio manager")

    def import_portfolio_handler(self):
        """Import existing portfolio from CSV file"""
        self.portfolio_manager.load_portfolio()
        self.update_portfolio_display()
        messagebox.showinfo("Success", f"Loaded {len(self.portfolio_manager.holdings)} positions")

    def refresh_prices_handler(self):
        """Refresh current prices for all holdings"""
        messagebox.showinfo("Coming Soon", "This feature will be implemented in Phase 4:\n"
                           "- Fetch current prices via yfinance\n"
                           "- Update P&L calculations\n"
                           "- Show progress bar")

    def save_portfolio_handler(self):
        """Save portfolio to CSV"""
        success = self.portfolio_manager.save_portfolio()
        if success:
            messagebox.showinfo("Success", f"Portfolio saved to {self.portfolio_manager.portfolio_file}")
        else:
            messagebox.showerror("Error", "Failed to save portfolio")

    def add_position_handler(self):
        """Open dialog to manually add a position"""
        messagebox.showinfo("Coming Soon", "This feature will be implemented in Phase 3:\n"
                           "- Open dialog for manual position entry\n"
                           "- Fields: Ticker, Entry Date, Entry Price, Shares, Sector\n"
                           "- Fetch current price and add to portfolio")

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

    def take_snapshot_handler(self):
        """Take snapshot of current portfolio for historical tracking"""
        messagebox.showinfo("Coming Soon", "This feature will be implemented in Phase 5:\n"
                           "- Calculate current portfolio metrics\n"
                           "- Append snapshot to portfolio_history.csv\n"
                           "- Update historical charts")


def main():
    root = tk.Tk()
    app = QuantStrategyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
