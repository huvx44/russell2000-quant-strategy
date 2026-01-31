from flask import Blueprint

bp = Blueprint('config_mgr', __name__)

from app.config_mgr import routes  # noqa: E402, F401
