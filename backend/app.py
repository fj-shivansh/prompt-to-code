#!/usr/bin/env python3
"""
Flask Backend API for Prompt-to-Code Testing System
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import sqlite3
import json
import os
import sys

# Load environment variables from .env file
load_dotenv('../.env')

sys.path.append('..')
from main import PromptToCodeSystem, DatabaseManager

app = Flask(__name__)
CORS(app)

# Get absolute path to database
DB_PATH = os.path.abspath('../historical_data.db')

# Initialize the prompt-to-code system
try:
    # Change to parent directory temporarily to ensure PromptToCodeSystem works correctly
    original_cwd = os.getcwd()
    os.chdir('..')
    system = PromptToCodeSystem()
    os.chdir(original_cwd)
    
    db_manager = DatabaseManager(DB_PATH)
except ValueError as e:
    print(f"Warning: {e}")
    system = None
    db_manager = DatabaseManager(DB_PATH)

@app.route('/api/process_prompt', methods=['POST'])
def process_prompt():
    if not system:
        return jsonify({
            'error': 'System not initialized. Please set GEMINI_API_KEY environment variable.'
        }), 500
    
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    
    if not prompt:
        return jsonify({'error': 'Prompt cannot be empty'}), 400
    
    try:
        # Change to parent directory to execute task (for database access)
        original_cwd = os.getcwd()
        os.chdir('..')
        result = system.process_task(prompt)
        os.chdir(original_cwd)
        
        # Debug logging
        print("=== DEBUG: Process Task Result ===")
        print(f"Result keys: {list(result.keys())}")
        
        if 'error' in result:
            print(f"ERROR in result: {result['error']}")
            return jsonify({'error': result['error']}), 500
        
        # Debug test result
        if 'test_result' in result:
            print(f"Test result success: {result['test_result'].success}")
            print(f"Test result raw result: {repr(result['test_result'].result)}")
            print(f"Test result error: {result['test_result'].error}")
            print(f"Test result execution time: {result['test_result'].execution_time}")
        
        # Debug generation
        if 'generation' in result:
            print(f"Generated code length: {len(result['generation'].code)}")
            print(f"Generated explanation: {result['generation'].explanation[:200]}...")
        
        response = {
            'success': True,
            'code': result['generation'].code,
            'explanation': result['generation'].explanation,
            'requirements': result['generation'].requirements or [],
            'result': result['test_result'].result if result['test_result'].success else None,
            'execution_time': result['test_result'].execution_time,
            'success_rate': result['analytics']['summary']['success_rate'],
            'error': result['test_result'].error if not result['test_result'].success else None,
            'analytics': result['analytics']
        }
        
        # Debug final response
        print(f"=== DEBUG: Final Response ===")
        print(f"Response result: {repr(response['result'])}")
        print(f"Response error: {response['error']}")
        print("===============================")
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/database_data')
def get_database_data():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    sort_by = request.args.get('sort_by', 'Date')
    sort_order = request.args.get('sort_order', 'DESC')
    ticker_filter = request.args.get('ticker', '')
    
    # Validate sort parameters
    valid_sort_columns = ['Date', 'Ticker', 'Adj_Close']
    valid_sort_orders = ['ASC', 'DESC']
    
    if sort_by not in valid_sort_columns:
        sort_by = 'Date'
    if sort_order.upper() not in valid_sort_orders:
        sort_order = 'DESC'
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Build WHERE clause
        where_clause = "WHERE Adj_Close IS NOT NULL"
        params = []
        
        if ticker_filter:
            where_clause += " AND Ticker = ?"
            params.append(ticker_filter)
        
        # Get total count with filter
        count_query = f"SELECT COUNT(*) FROM stock_data {where_clause}"
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        
        # Get paginated and sorted data
        offset = (page - 1) * per_page
        data_query = f"""
            SELECT Date, Ticker, Adj_Close 
            FROM stock_data 
            {where_clause}
            ORDER BY {sort_by} {sort_order}
            LIMIT ? OFFSET ?
        """
        cursor.execute(data_query, params + [per_page, offset])
        
        rows = cursor.fetchall()
        conn.close()
        
        data = [
            {"Date": row[0], "Ticker": row[1], "Adj_Close": row[2]}
            for row in rows
        ]
        
        return jsonify({
            'data': data,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page,
            'sort_by': sort_by,
            'sort_order': sort_order,
            'ticker_filter': ticker_filter
        })
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/api/tickers')
def get_unique_tickers():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT Ticker 
            FROM stock_data 
            WHERE Adj_Close IS NOT NULL 
            ORDER BY Ticker
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        tickers = [row[0] for row in rows]
        
        return jsonify({'tickers': tickers})
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'system_ready': system is not None})

if __name__ == '__main__':
    # Configure Flask to ignore generated_code.py for auto-reload
    import os
    from werkzeug.serving import run_simple
    
    # Use werkzeug directly to have more control over file watching
    run_simple('0.0.0.0', 5000, app, use_reloader=True, use_debugger=True,
               extra_files=[], exclude_patterns=['**/generated_code.py'])