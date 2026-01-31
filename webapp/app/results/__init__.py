from flask import Blueprint

bp = Blueprint('results', __name__)

from app.results import routes  # noqa: E402, F401
