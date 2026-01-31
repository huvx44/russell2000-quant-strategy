"""
SocketIO event handlers for backtest streaming.
"""
from flask import current_app
from flask_socketio import emit
from flask_login import current_user
from app import socketio
from app.backtest.runner import BacktestRunner

# Single runner instance (one backtest at a time)
runner = BacktestRunner()


@socketio.on('start_backtest')
def handle_start_backtest(data):
    if not current_user.is_authenticated:
        emit('backtest_error', {'error': 'Not authenticated'})
        return

    if runner.running:
        emit('backtest_error', {'error': 'A backtest is already running'})
        return

    core_dir = current_app.config['CORE_DIR']
    data_dir = current_app.config['DATA_DIR']

    config = {
        'refresh': data.get('refresh', False),
        'start_date': data.get('start_date', '2021-01-01'),
        'end_date': data.get('end_date', ''),
        'initial_balance': data.get('initial_balance', '100000'),
        'rebalance_freq': data.get('rebalance_freq', 'Q'),
        'top_n': data.get('top_n', '20'),
        'max_sector_weight': data.get('max_sector_weight', '0.30'),
        'min_market_cap': data.get('min_market_cap', '300000000'),
        'max_market_cap': data.get('max_market_cap', '10000000000'),
        'min_avg_volume': data.get('min_avg_volume', '100000'),
        'quality_weight': data.get('quality_weight', '0.35'),
        'growth_weight': data.get('growth_weight', '0.35'),
        'value_weight': data.get('value_weight', '0.15'),
        'momentum_weight': data.get('momentum_weight', '0.15'),
    }

    emit('backtest_output', {'line': 'Starting backtest...\n'})

    def run_backtest():
        try:
            runner.start(core_dir, data_dir, config)
            for line in runner.read_output():
                socketio.emit('backtest_output', {'line': line}, namespace='/')

            rc = runner.return_code
            if rc == 0:
                socketio.emit('backtest_complete', {'success': True, 'message': 'Backtest completed successfully'}, namespace='/')
            else:
                socketio.emit('backtest_complete', {'success': False, 'message': f'Backtest failed (exit code {rc})'}, namespace='/')
        except Exception as e:
            socketio.emit('backtest_error', {'error': str(e)}, namespace='/')

    socketio.start_background_task(run_backtest)


@socketio.on('stop_backtest')
def handle_stop_backtest():
    if not current_user.is_authenticated:
        return
    runner.stop()
    emit('backtest_output', {'line': '\n--- Backtest stopped by user ---\n'})
    emit('backtest_complete', {'success': False, 'message': 'Backtest stopped by user'})
