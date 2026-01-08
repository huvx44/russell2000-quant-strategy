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

        # Create tabs
        self.create_backtest_tab()
        self.create_results_tab()
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


def main():
    root = tk.Tk()
    app = QuantStrategyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
