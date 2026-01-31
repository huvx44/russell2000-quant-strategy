"""
Results viewer routes: browse results folders, view CSV data, serve charts.
"""
import os
import glob
import json
import pandas as pd
from flask import render_template, send_from_directory, current_app, jsonify, abort
from flask_login import login_required
from app.results import bp


def get_results_folders():
    """Get all results_* folders sorted by modification time (newest first)."""
    core_dir = current_app.config['CORE_DIR']
    pattern = os.path.join(core_dir, 'results_*')
    folders = glob.glob(pattern)
    folders.sort(key=os.path.getmtime, reverse=True)
    return [(os.path.basename(f), os.path.getmtime(f)) for f in folders if os.path.isdir(f)]


@bp.route('/')
@login_required
def index():
    folders = get_results_folders()
    return render_template('results/list.html', folders=folders)


@bp.route('/<folder_name>')
@login_required
def detail(folder_name):
    core_dir = current_app.config['CORE_DIR']
    folder_path = os.path.join(core_dir, folder_name)

    if not os.path.isdir(folder_path) or not folder_name.startswith('results_'):
        abort(404)

    # List CSV files
    csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
    csv_names = [os.path.basename(f) for f in csv_files]

    # List chart images
    chart_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    chart_names = [os.path.basename(f) for f in chart_files]

    # Load strategy config if available
    config_file = os.path.join(folder_path, 'strategy_config.json')
    strategy_config = None
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            strategy_config = json.load(f)

    return render_template('results/detail.html',
                           folder=folder_name,
                           csv_files=csv_names,
                           chart_files=chart_names,
                           strategy_config=strategy_config)


@bp.route('/<folder_name>/csv/<filename>')
@login_required
def view_csv(folder_name, filename):
    core_dir = current_app.config['CORE_DIR']
    filepath = os.path.join(core_dir, folder_name, filename)

    if not os.path.exists(filepath) or not folder_name.startswith('results_'):
        abort(404)

    try:
        df = pd.read_csv(filepath)
        # Limit to 500 rows for browser performance
        truncated = len(df) > 500
        if truncated:
            df = df.head(500)

        columns = list(df.columns)
        rows = df.values.tolist()

        return jsonify({
            'columns': columns,
            'rows': rows,
            'total_rows': len(df) if not truncated else f'500 of {len(df)}+',
            'filename': filename,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@bp.route('/<folder_name>/chart/<filename>')
@login_required
def serve_chart(folder_name, filename):
    core_dir = current_app.config['CORE_DIR']
    folder_path = os.path.join(core_dir, folder_name)

    if not folder_name.startswith('results_'):
        abort(404)

    return send_from_directory(folder_path, filename, mimetype='image/png')


@bp.route('/<folder_name>/export')
@login_required
def export_excel(folder_name):
    """Export all CSV files in a results folder as a single Excel workbook."""
    core_dir = current_app.config['CORE_DIR']
    folder_path = os.path.join(core_dir, folder_name)

    if not os.path.isdir(folder_path) or not folder_name.startswith('results_'):
        abort(404)

    import io
    output = io.BytesIO()

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for csv_file in csv_files:
            sheet = os.path.basename(csv_file).replace('.csv', '')[:31]
            df = pd.read_csv(csv_file)
            df.to_excel(writer, sheet_name=sheet, index=False)

    output.seek(0)

    from flask import send_file
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'{folder_name}.xlsx'
    )
