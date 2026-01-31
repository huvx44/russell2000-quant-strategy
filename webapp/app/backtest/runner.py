"""
Backtest subprocess runner. Manages execution and output streaming.
"""
import subprocess
import os
import threading


class BacktestRunner:
    """Manages a single backtest subprocess."""

    def __init__(self):
        self.process = None
        self.running = False
        self.output_lines = []
        self._lock = threading.Lock()

    def start(self, core_dir, data_dir, config):
        """
        Start the backtest subprocess.

        Args:
            core_dir: Path to directory containing russell2000_backtest.py
            data_dir: Path to data directory for results/cache
            config: Dict with backtest parameters
        """
        if self.running:
            raise RuntimeError('A backtest is already running')

        script = os.path.join(core_dir, 'russell2000_backtest.py')
        if not os.path.exists(script):
            raise FileNotFoundError(f'Backtest script not found: {script}')

        cmd = ['python', script]

        if config.get('refresh'):
            cmd.append('--refresh')

        # Map config keys to CLI args
        arg_map = {
            'initial_balance': '--initial-balance',
            'rebalance_freq': '--rebalance-freq',
            'start_date': '--start-date',
            'end_date': '--end-date',
            'top_n': '--top-n',
            'max_sector_weight': '--max-sector-weight',
            'min_market_cap': '--min-market-cap',
            'max_market_cap': '--max-market-cap',
            'min_avg_volume': '--min-avg-volume',
            'quality_weight': '--quality-weight',
            'growth_weight': '--growth-weight',
            'value_weight': '--value-weight',
            'momentum_weight': '--momentum-weight',
        }

        for key, flag in arg_map.items():
            val = config.get(key)
            if val is not None:
                cmd.extend([flag, str(val)])

        with self._lock:
            self.output_lines = []
            self.running = True

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=core_dir,
        )

        return self.process

    def read_output(self):
        """Generator that yields lines from the subprocess stdout."""
        if not self.process:
            return

        try:
            for line in self.process.stdout:
                with self._lock:
                    self.output_lines.append(line)
                yield line
        finally:
            self.process.wait()
            with self._lock:
                self.running = False

    def stop(self):
        """Kill the running subprocess."""
        if self.process and self.running:
            self.process.kill()
            with self._lock:
                self.running = False

    @property
    def return_code(self):
        if self.process:
            return self.process.returncode
        return None
