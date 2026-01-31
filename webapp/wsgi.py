"""
WSGI entry point for the application.

Usage:
    gunicorn --worker-class=gthread --workers=1 --threads=4 -b 0.0.0.0:5000 wsgi:app

    Or for development:
    python wsgi.py
"""
from app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
