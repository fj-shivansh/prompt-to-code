"""
API routes module
"""
from flask import Blueprint

# Create blueprints
prompt_bp = Blueprint('prompt', __name__, url_prefix='/api')
database_bp = Blueprint('database', __name__, url_prefix='/api')
condition_bp = Blueprint('condition', __name__, url_prefix='/api')
nav_bp = Blueprint('nav', __name__, url_prefix='/api')

def register_routes(app):
    """Register all route blueprints with the Flask app"""
    from . import prompt_routes
    from . import database_routes
    from . import condition_routes
    from . import nav_routes

    app.register_blueprint(prompt_bp)
    app.register_blueprint(database_bp)
    app.register_blueprint(condition_bp)
    app.register_blueprint(nav_bp)
