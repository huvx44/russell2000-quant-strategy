"""
Flask application factory for Russell 2000 Quant Strategy Web App
"""
import os
from datetime import datetime
from flask import Flask
from flask_socketio import SocketIO
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
socketio = SocketIO()
login_manager = LoginManager()


def create_app(config_class=None):
    app = Flask(__name__)

    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(
        os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')),
        'app.db'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Data directory for all persistent files
    app.config['DATA_DIR'] = os.environ.get(
        'DATA_DIR',
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    )

    # Core scripts directory (where backtest scripts live)
    app.config['CORE_DIR'] = os.environ.get(
        'CORE_DIR',
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )

    # WebAuthn settings
    app.config['RP_ID'] = os.environ.get('RP_ID', 'localhost')
    app.config['RP_NAME'] = os.environ.get('RP_NAME', 'Quant Strategy')
    app.config['RP_ORIGIN'] = os.environ.get('RP_ORIGIN', 'http://localhost:5000')

    # Session settings
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

    if config_class:
        app.config.from_object(config_class)

    # Ensure data directory exists
    os.makedirs(app.config['DATA_DIR'], exist_ok=True)

    # Initialize extensions
    db.init_app(app)
    socketio.init_app(app, async_mode='threading')
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'

    # Register blueprints
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    from app.backtest import bp as backtest_bp
    app.register_blueprint(backtest_bp, url_prefix='/backtest')

    from app.results import bp as results_bp
    app.register_blueprint(results_bp, url_prefix='/results')

    from app.config_mgr import bp as config_bp
    app.register_blueprint(config_bp, url_prefix='/config')

    from app.portfolio import bp as portfolio_bp
    app.register_blueprint(portfolio_bp, url_prefix='/portfolio')

    # Template filters
    @app.template_filter('timestamp_format')
    def timestamp_format(ts):
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')

    # Root redirect
    @app.route('/')
    def index():
        from flask import redirect, url_for
        from flask_login import current_user
        if current_user.is_authenticated:
            return redirect(url_for('backtest.index'))
        return redirect(url_for('auth.login'))

    # Create database tables
    with app.app_context():
        from app.auth.models import User, WebAuthnCredential
        db.create_all()

        # Create default admin user if none exists
        if User.query.count() == 0:
            admin_password = os.environ.get('ADMIN_PASSWORD', 'admin')
            admin = User(username='admin')
            admin.set_password(admin_password)
            db.session.add(admin)
            db.session.commit()

    return app
