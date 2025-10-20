#!/usr/bin/env python3
"""
Flask Backend API for Prompt-to-Code Testing System
Modular version with clean separation of concerns
"""

from flask import Flask
from flask_cors import CORS
from werkzeug.serving import run_simple
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import configuration
from backend.config.settings import Settings

# Import database manager
from backend.database.manager import DatabaseManager

# Import services
from backend.services.prompt_refiner import PromptRefiner
from backend.services.nav_calculator import NAVCalculator

# Import utils
from backend.utils.process_manager import ProcessManager

# Import main system
from backend.services.system_orchestrator import PromptToCodeSystem

# Create Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
try:
    # Change to parent directory temporarily to ensure PromptToCodeSystem works correctly
    original_cwd = os.getcwd()
    os.chdir('..')
    system = PromptToCodeSystem()
    os.chdir(original_cwd)

    db_manager = DatabaseManager(Settings.DB_PATH)
    prompt_refiner = PromptRefiner()
    nav_calculator = NAVCalculator()
    process_manager = ProcessManager()

    print("✓ All components initialized successfully")

except ValueError as e:
    print(f"Warning: {e}")
    system = None
    db_manager = DatabaseManager(Settings.DB_PATH)
    prompt_refiner = None
    nav_calculator = NAVCalculator()
    process_manager = ProcessManager()


# Import and initialize routes
from backend.routes import register_routes
from backend.routes.prompt_routes import init_prompt_routes
from backend.routes.database_routes import init_database_routes
from backend.routes.condition_routes import init_condition_routes
from backend.routes.nav_routes import init_nav_routes

# Initialize route dependencies
init_prompt_routes(system, prompt_refiner, process_manager)
init_database_routes(db_manager)
init_condition_routes(system, db_manager)
init_nav_routes(nav_calculator)

# Register all blueprints
register_routes(app)

print("✓ All routes registered")


if __name__ == '__main__':
    print("="*60)
    print("Flask Backend API - Prompt-to-Code Testing System")
    print("="*60)
    print(f"Database: {Settings.DB_PATH}")
    print(f"System initialized: {system is not None}")
    print(f"Prompt refiner initialized: {prompt_refiner is not None}")
    print("="*60)

    # Use werkzeug directly to have more control over file watching
    run_simple(
        Settings.FLASK_HOST,
        Settings.FLASK_PORT,
        app,
        use_reloader=True,
        use_debugger=True,
        extra_files=[],
        exclude_patterns=Settings.EXCLUDE_PATTERNS
    )
