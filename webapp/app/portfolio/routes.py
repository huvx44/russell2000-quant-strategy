"""
Portfolio management routes: CRUD operations, price refresh, charts, tags.
"""
import os
import io
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import render_template, request, jsonify, current_app, Response
from flask_login import login_required
from app.portfolio import bp

# Lazy-initialized portfolio manager
_pm = None


def get_pm():
    """Get or create the PortfolioManager instance with correct paths."""
    global _pm
    if _pm is None:
        import sys
        core_dir = current_app.config['CORE_DIR']
        data_dir = current_app.config['DATA_DIR']
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)
        from portfolio_manager import PortfolioManager
        _pm = PortfolioManager(
            portfolio_file=os.path.join(data_dir, 'my_portfolio.csv'),
            history_file=os.path.join(data_dir, 'portfolio_history.csv'),
        )
        # Override tags dir/file to use data_dir
        _pm.tags_dir = os.path.join(data_dir, 'tagged_portfolios')
        _pm.tags_index_file = os.path.join(data_dir, 'portfolio_tags.csv')
        os.makedirs(_pm.tags_dir, exist_ok=True)
    return _pm


@bp.route('/')
@login_required
def index():
    pm = get_pm()
    summary = pm.calculate_portfolio_summary()
    holdings = pm.holdings.to_dict('records') if len(pm.holdings) > 0 else []
    tags = pm.get_all_tags()
    tags_list = tags.to_dict('records') if len(tags) > 0 else []

    return render_template('portfolio/index.html',
                           summary=summary,
                           holdings=holdings,
                           tags=tags_list)


@bp.route('/holdings')
@login_required
def get_holdings():
    pm = get_pm()
    summary = pm.calculate_portfolio_summary()
    holdings = pm.holdings.to_dict('records') if len(pm.holdings) > 0 else []
    return jsonify({'summary': summary, 'holdings': holdings})


@bp.route('/refresh', methods=['POST'])
@login_required
def refresh_prices():
    pm = get_pm()
    if len(pm.holdings) == 0:
        return jsonify({'error': 'No holdings to refresh'}), 400

    result = pm.refresh_all_prices()
    pm.save_portfolio()
    return jsonify({
        'success': result['success'],
        'failed': result['failed'],
        'total': result['total'],
    })


@bp.route('/add', methods=['POST'])
@login_required
def add_position():
    pm = get_pm()
    data = request.get_json()

    ticker = data.get('ticker', '').strip().upper()
    if not ticker:
        return jsonify({'error': 'Ticker is required'}), 400

    try:
        entry_price = float(data.get('entry_price', 0))
        shares = int(data.get('shares', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid price or shares'}), 400

    if entry_price <= 0 or shares <= 0:
        return jsonify({'error': 'Price and shares must be positive'}), 400

    success = pm.add_position(
        ticker=ticker,
        entry_date=data.get('entry_date', ''),
        entry_price=entry_price,
        shares=shares,
        sector=data.get('sector', ''),
    )

    if success:
        pm.save_portfolio()
        return jsonify({'success': True, 'message': f'Added {ticker}'})
    else:
        return jsonify({'error': f'Failed to add {ticker}. It may already exist.'}), 400


@bp.route('/edit/<ticker>', methods=['POST'])
@login_required
def edit_position(ticker):
    pm = get_pm()
    data = request.get_json()

    kwargs = {}
    if 'entry_date' in data:
        kwargs['entry_date'] = data['entry_date']
    if 'entry_price' in data:
        kwargs['entry_price'] = float(data['entry_price'])
    if 'shares' in data:
        kwargs['shares'] = int(data['shares'])
    if 'sector' in data:
        kwargs['sector'] = data['sector']

    success = pm.update_position(ticker, **kwargs)
    if success:
        pm.save_portfolio()
        return jsonify({'success': True, 'message': f'Updated {ticker}'})
    return jsonify({'error': f'{ticker} not found'}), 404


@bp.route('/remove/<ticker>', methods=['POST'])
@login_required
def remove_position(ticker):
    pm = get_pm()
    success = pm.remove_position(ticker)
    if success:
        pm.save_portfolio()
        return jsonify({'success': True, 'message': f'Removed {ticker}'})
    return jsonify({'error': f'{ticker} not found'}), 404


@bp.route('/save', methods=['POST'])
@login_required
def save():
    pm = get_pm()
    success = pm.save_portfolio()
    if success:
        return jsonify({'success': True, 'message': 'Portfolio saved'})
    return jsonify({'error': 'Failed to save'}), 500


@bp.route('/snapshot', methods=['POST'])
@login_required
def snapshot():
    pm = get_pm()
    if len(pm.holdings) == 0:
        return jsonify({'error': 'No holdings to snapshot'}), 400

    success = pm.append_snapshot()
    if success:
        summary = pm.calculate_portfolio_summary()
        return jsonify({'success': True, 'summary': summary})
    return jsonify({'error': 'Failed to create snapshot'}), 500


@bp.route('/import-recommended', methods=['POST'])
@login_required
def import_recommended():
    pm = get_pm()
    core_dir = current_app.config['CORE_DIR']

    # Find latest results folder
    folders = glob.glob(os.path.join(core_dir, 'results_*/'))
    if not folders:
        return jsonify({'error': 'No backtest results found'}), 404

    latest = max(folders, key=os.path.getmtime)
    portfolio_file = os.path.join(latest, 'current_portfolio.csv')
    if not os.path.exists(portfolio_file):
        return jsonify({'error': 'No portfolio file in latest results'}), 404

    recommended = pd.read_csv(portfolio_file)
    positions = recommended.to_dict('records')

    return jsonify({
        'success': True,
        'folder': os.path.basename(latest),
        'positions': positions,
    })


@bp.route('/import-execute', methods=['POST'])
@login_required
def import_execute():
    pm = get_pm()
    data = request.get_json()
    adjustments = data.get('adjustments', {})

    core_dir = current_app.config['CORE_DIR']
    folders = glob.glob(os.path.join(core_dir, 'results_*/'))
    if not folders:
        return jsonify({'error': 'No results found'}), 404

    latest = max(folders, key=os.path.getmtime)
    portfolio_file = os.path.join(latest, 'current_portfolio.csv')
    recommended = pd.read_csv(portfolio_file)

    count = pm.import_from_recommended(recommended, adjustments)
    pm.save_portfolio()

    return jsonify({'success': True, 'imported': count})


# --- Tags ---

@bp.route('/tags')
@login_required
def list_tags():
    pm = get_pm()
    tags = pm.get_all_tags()
    return jsonify(tags.to_dict('records') if len(tags) > 0 else [])


@bp.route('/tags/create', methods=['POST'])
@login_required
def create_tag():
    pm = get_pm()
    data = request.get_json()
    name = data.get('name', '').strip()
    description = data.get('description', '')

    if not name:
        return jsonify({'error': 'Tag name is required'}), 400

    tag_id = pm.save_tagged_portfolio(name, description)
    if tag_id:
        return jsonify({'success': True, 'tag_id': tag_id})
    return jsonify({'error': 'Failed to create tag'}), 500


@bp.route('/tags/<tag_id>/load', methods=['POST'])
@login_required
def load_tag(tag_id):
    pm = get_pm()
    holdings = pm.load_tagged_portfolio(tag_id)
    if holdings is not None:
        pm.holdings = holdings
        pm.save_portfolio()
        return jsonify({'success': True, 'count': len(holdings)})
    return jsonify({'error': 'Tag not found'}), 404


@bp.route('/tags/<tag_id>/delete', methods=['POST'])
@login_required
def delete_tag(tag_id):
    pm = get_pm()
    success = pm.delete_tagged_portfolio(tag_id)
    if success:
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to delete tag'}), 404


@bp.route('/tags/compare', methods=['POST'])
@login_required
def compare_tags():
    pm = get_pm()
    data = request.get_json()
    tag1 = data.get('tag1')  # None means current portfolio
    tag2 = data.get('tag2')

    result = pm.compare_portfolios(tag1, tag2)
    if result:
        return jsonify(result)
    return jsonify({'error': 'Failed to compare'}), 400


# --- Charts ---

@bp.route('/chart/sector')
@login_required
def chart_sector():
    pm = get_pm()
    if len(pm.holdings) == 0:
        return _empty_chart('No positions')

    holdings = pm.holdings
    sector_values = holdings.groupby('Sector')['Current_Value'].sum()

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    fig.patch.set_facecolor('#2b2f36')
    ax.set_facecolor('#2b2f36')

    colors = plt.cm.Set3(range(len(sector_values)))
    wedges, texts, autotexts = ax.pie(
        sector_values, labels=sector_values.index,
        autopct='%1.1f%%', startangle=90, colors=colors,
        textprops={'color': '#e0e0e0', 'fontsize': 8}
    )
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(7)
    ax.set_title('Sector Allocation', color='#e0e0e0', fontweight='bold')

    return _fig_to_response(fig)


@bp.route('/chart/pnl')
@login_required
def chart_pnl():
    pm = get_pm()
    if len(pm.holdings) == 0:
        return _empty_chart('No positions')

    holdings = pm.holdings.sort_values('Unrealized_PL_Pct', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    fig.patch.set_facecolor('#2b2f36')
    ax.set_facecolor('#2b2f36')

    colors = ['#28a745' if x >= 0 else '#dc3545' for x in holdings['Unrealized_PL_Pct']]
    ax.bar(range(len(holdings)), holdings['Unrealized_PL_Pct'], color=colors, alpha=0.8)
    ax.set_xticks(range(len(holdings)))
    ax.set_xticklabels(holdings['Ticker'], rotation=45, ha='right', fontsize=8, color='#e0e0e0')
    ax.axhline(y=0, color='#8b949e', linewidth=0.8)
    ax.set_ylabel('P&L %', color='#e0e0e0')
    ax.set_title('P&L by Position', color='#e0e0e0', fontweight='bold')
    ax.tick_params(colors='#e0e0e0')
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()

    return _fig_to_response(fig)


@bp.route('/chart/performance')
@login_required
def chart_performance():
    pm = get_pm()
    if len(pm.history) == 0:
        return _empty_chart('No snapshots yet')

    history = pm.history
    dates = pd.to_datetime(history['Date'])
    values = history['Total_Value']
    cost = history['Total_Cost_Basis']

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    fig.patch.set_facecolor('#2b2f36')
    ax.set_facecolor('#2b2f36')

    ax.plot(dates, values, marker='o', linewidth=2, markersize=5, color='#58a6ff', label='Portfolio Value')
    ax.plot(dates, cost, linestyle='--', linewidth=1.5, color='#8b949e', alpha=0.7, label='Cost Basis')
    ax.fill_between(dates, values, cost, where=(values >= cost), color='#28a745', alpha=0.15)
    ax.fill_between(dates, values, cost, where=(values < cost), color='#dc3545', alpha=0.15)
    ax.set_title('Performance Over Time', color='#e0e0e0', fontweight='bold')
    ax.set_ylabel('Value ($)', color='#e0e0e0')
    ax.legend(facecolor='#2b2f36', edgecolor='#3a3f47', labelcolor='#e0e0e0')
    ax.tick_params(colors='#e0e0e0')
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()

    return _fig_to_response(fig)


def _fig_to_response(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype='image/png')


def _empty_chart(message):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    fig.patch.set_facecolor('#2b2f36')
    ax.set_facecolor('#2b2f36')
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha='center', va='center',
            fontsize=14, color='#8b949e')
    ax.set_xticks([])
    ax.set_yticks([])
    return _fig_to_response(fig)
