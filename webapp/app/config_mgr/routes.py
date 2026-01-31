"""
Configuration management routes: view/edit strategy params, save/load presets.
"""
import os
import json
import glob
from flask import render_template, request, jsonify, current_app, flash, redirect, url_for
from flask_login import login_required
from app.config_mgr import bp

DEFAULT_CONFIG = {
    'quality_weight': 35,
    'growth_weight': 35,
    'value_weight': 15,
    'momentum_weight': 15,
    'top_n': 20,
    'max_sector_weight': 30,
    'min_market_cap': 300,
    'max_market_cap': 10,
    'min_avg_volume': 100000,
    'rebalance_freq': 'Q',
    'initial_balance': 100000,
}


def get_presets_dir():
    data_dir = current_app.config['DATA_DIR']
    presets_dir = os.path.join(data_dir, 'config_presets')
    os.makedirs(presets_dir, exist_ok=True)
    return presets_dir


@bp.route('/')
@login_required
def index():
    # Load available presets
    presets_dir = get_presets_dir()
    preset_files = glob.glob(os.path.join(presets_dir, '*.json'))
    presets = []
    for f in sorted(preset_files):
        name = os.path.basename(f).replace('.json', '')
        presets.append(name)

    # Load configs from results folders (strategy_config.json)
    core_dir = current_app.config['CORE_DIR']
    results_configs = []
    for folder in sorted(glob.glob(os.path.join(core_dir, 'results_*')), reverse=True):
        config_file = os.path.join(folder, 'strategy_config.json')
        if os.path.exists(config_file):
            results_configs.append(os.path.basename(folder))

    return render_template('config/index.html',
                           config=DEFAULT_CONFIG,
                           presets=presets,
                           results_configs=results_configs)


@bp.route('/save', methods=['POST'])
@login_required
def save_preset():
    data = request.get_json()
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Preset name is required'}), 400

    # Sanitize filename
    safe_name = ''.join(c for c in name if c.isalnum() or c in '-_ ')
    if not safe_name:
        return jsonify({'error': 'Invalid preset name'}), 400

    config = data.get('config', {})
    presets_dir = get_presets_dir()
    filepath = os.path.join(presets_dir, f'{safe_name}.json')

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    return jsonify({'success': True, 'message': f'Preset "{safe_name}" saved'})


@bp.route('/load/<name>')
@login_required
def load_preset(name):
    presets_dir = get_presets_dir()
    filepath = os.path.join(presets_dir, f'{name}.json')

    if not os.path.exists(filepath):
        return jsonify({'error': 'Preset not found'}), 404

    with open(filepath, 'r') as f:
        config = json.load(f)

    return jsonify(config)


@bp.route('/load-from-results/<folder_name>')
@login_required
def load_from_results(folder_name):
    core_dir = current_app.config['CORE_DIR']
    config_file = os.path.join(core_dir, folder_name, 'strategy_config.json')

    if not os.path.exists(config_file) or not folder_name.startswith('results_'):
        return jsonify({'error': 'Config not found'}), 404

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Map to our form field names
    mapped = {
        'quality_weight': int(float(config.get('QUALITY_WEIGHT', config.get('quality_weight', 0.35))) * 100),
        'growth_weight': int(float(config.get('GROWTH_WEIGHT', config.get('growth_weight', 0.35))) * 100),
        'value_weight': int(float(config.get('VALUE_WEIGHT', config.get('value_weight', 0.15))) * 100),
        'momentum_weight': int(float(config.get('MOMENTUM_WEIGHT', config.get('momentum_weight', 0.15))) * 100),
        'top_n': config.get('TOP_N', config.get('top_n', 20)),
        'max_sector_weight': int(float(config.get('MAX_SECTOR_WEIGHT', config.get('max_sector_weight', 0.30))) * 100),
        'min_market_cap': int(float(config.get('MIN_MARKET_CAP', config.get('min_market_cap', 300e6))) / 1e6),
        'max_market_cap': int(float(config.get('MAX_MARKET_CAP', config.get('max_market_cap', 10e9))) / 1e9),
        'min_avg_volume': config.get('MIN_AVG_VOLUME', config.get('min_avg_volume', 100000)),
        'rebalance_freq': config.get('REBALANCE_FREQ', config.get('rebalance_freq', 'Q')),
        'initial_balance': config.get('INITIAL_BALANCE', config.get('initial_balance', 100000)),
    }

    return jsonify(mapped)


@bp.route('/delete/<name>', methods=['POST'])
@login_required
def delete_preset(name):
    presets_dir = get_presets_dir()
    filepath = os.path.join(presets_dir, f'{name}.json')

    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'success': True, 'message': f'Preset "{name}" deleted'})

    return jsonify({'error': 'Preset not found'}), 404
