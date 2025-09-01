#!/usr/bin/env python3
"""
Flask Backend API for Prompt-to-Code Testing System
"""

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

from dotenv import load_dotenv
import sqlite3
import json
import os
import sys
import pandas as pd
import numpy as np

from werkzeug.serving import run_simple
import google.generativeai as genai
from typing import Optional
import threading
import time
from collections import deque

# Load environment variables from .env file
load_dotenv('../.env')

sys.path.append('..')
from main import PromptToCodeSystem, DatabaseManager

class PromptRefiner:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client for prompt refinement.
        """
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Use existing API key logic from environment
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                # Try multiple API keys
                api_keys_str = os.getenv("GEMINI_API_KEYS")
                if api_keys_str:
                    api_keys = [key.strip() for key in api_keys_str.split(',')]
                    api_key = api_keys[0]  # Use first key for refinement
            
            if not api_key:
                raise ValueError("Gemini API key not provided")
            genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel("gemini-1.5-flash")
    
    def refine_prompt(self, user_request: str) -> str:
        """
        Refine a vague user request into a detailed, structured, production-ready prompt.
        """
        system_prompt = f"""
                        You are an expert financial analyst and prompt engineer. 
                        Transform the following user request into a **concise, step-by-step instruction** specifying:

                        1. Calculations required
                        2. Output columns and what each should contain

                        Focus ONLY on calculations and column mapping. 
                        Do NOT include extra instructions, handling missing data, or formatting advice.
                        User Request:
                        {user_request}
                    """

        response = self.model.generate_content(system_prompt)
        return response.text.strip()

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
    prompt_refiner = PromptRefiner()
except ValueError as e:
    print(f"Warning: {e}")
    system = None
    db_manager = DatabaseManager(DB_PATH)
    prompt_refiner = None

def generate_status_updates(original_prompt):
    """Generator function for Server-Sent Events with detailed progress tracking"""
    try:
        yield f"data: {json.dumps({'type': 'init', 'message': 'Initializing...', 'progress': {'step': 1, 'total': 6}})}\n\n"
        
        if not system:
            yield f"data: {json.dumps({'type': 'error', 'message': 'System not initialized'})}\n\n"
            return
        
        # Complete process with restart mechanism
        max_complete_restarts = 2
        complete_restart_attempt = 0
        
        while complete_restart_attempt <= max_complete_restarts:
            complete_restart_attempt += 1
            
            yield f"data: {json.dumps({'type': 'attempt_start', 'message': f'Starting Complete Attempt {complete_restart_attempt}', 'attempt': complete_restart_attempt, 'max_attempts': max_complete_restarts + 1, 'progress': {'step': 2, 'total': 4}})}\n\n"
            
            # Process the prompt directly
            yield f"data: {json.dumps({'type': 'generating', 'message': 'Generating and executing code...', 'progress': {'step': 3, 'total': 4}})}\n\n"
            
            original_cwd = os.getcwd()
            os.chdir('..')
            
            # Process the actual task
            result = system.process_task(original_prompt)
            
            os.chdir(original_cwd)
            
            # Get actual retry count from result
            actual_retries = result['analytics']['generation_info']['retry_attempts'] if result and 'analytics' in result and 'generation_info' in result['analytics'] else 1
            
            # Show execution summary
            yield f"data: {json.dumps({'type': 'execution_complete', 'message': f'Code execution completed ({actual_retries} retries used)', 'retry': actual_retries, 'max_retries': 5, 'attempt': complete_restart_attempt})}\n\n"
            
            # Show final attempt status
            if result and 'error' not in result and result.get('test_result') and result['test_result'].success:
                yield f"data: {json.dumps({'type': 'attempt_success', 'message': f'Complete Attempt {complete_restart_attempt} succeeded!', 'attempt': complete_restart_attempt, 'progress': {'step': 4, 'total': 4}})}\n\n"
                
                # Send final result
                max_retries = result['analytics']['generation_info']['max_retries'] if 'analytics' in result and 'generation_info' in result['analytics'] else 5
                
                response = {
                    'success': True,
                    'original_prompt': original_prompt,
                    'refined_prompt': original_prompt,
                    'prompt_was_refined': False,
                    'complete_restart_attempts': complete_restart_attempt,
                    'max_complete_restarts': max_complete_restarts + 1,
                    'execution_retry_attempts': actual_retries,
                    'max_execution_retries': max_retries,
                    'total_retries_used': (complete_restart_attempt - 1) * max_retries + actual_retries,
                    'had_complete_restarts': complete_restart_attempt > 1,
                    'code': result['generation'].code,
                    'explanation': result['generation'].explanation,
                    'requirements': result['generation'].requirements or [],
                    'result': result['test_result'].result if result['test_result'].success else None,
                    'execution_time': result['test_result'].execution_time,
                    'success_rate': result['analytics']['summary']['success_rate'],
                    'error': result['test_result'].error if not result['test_result'].success else None,
                    'analytics': result['analytics']
                }
                
                yield f"data: {json.dumps({'type': 'final_result', 'data': response})}\n\n"
                break
            else:
                # This complete attempt failed
                error_msg = result.get('error', 'Unknown error') if result else 'System error'
                if result and result.get('test_result'):
                    error_msg = result['test_result'].error
                
                yield f"data: {json.dumps({'type': 'retry_failed', 'message': f'All {actual_retries} retries failed', 'retry': actual_retries, 'attempt': complete_restart_attempt})}\n\n"
                yield f"data: {json.dumps({'type': 'attempt_failed', 'message': f'Complete Attempt {complete_restart_attempt} failed after {actual_retries} retries', 'attempt': complete_restart_attempt, 'error': error_msg[:200]})}\n\n"
                
                if complete_restart_attempt <= max_complete_restarts:
                    yield f"data: {json.dumps({'type': 'restarting', 'message': f'Restarting completely... ({max_complete_restarts + 1 - complete_restart_attempt} attempts remaining)', 'progress': {'step': 2, 'total': 4}})}\n\n"
                    continue
                else:
                    max_retries = 5  # Default value for error calculation
                    yield f"data: {json.dumps({'type': 'final_error', 'message': 'All complete attempts failed', 'total_attempts': complete_restart_attempt, 'total_retries': complete_restart_attempt * max_retries})}\n\n"
                    break
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'final_error', 'message': f'System error: {str(e)}'})}\n\n"

@app.route('/api/process_prompt_stream', methods=['POST'])
def process_prompt_stream():
    """Server-Sent Events endpoint for real-time status updates"""
    data = request.get_json()
    original_prompt = data.get('prompt', '').strip()
    
    if not original_prompt:
        return jsonify({'error': 'Prompt cannot be empty'}), 400
    
    def event_stream():
        for update in generate_status_updates(original_prompt):
            yield update
    
    return Response(event_stream(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})

@app.route('/api/process_prompt', methods=['POST'])
def process_prompt():
    if not system:
        return jsonify({
            'error': 'System not initialized. Please set GEMINI_API_KEY (single key) or GEMINI_API_KEYS (comma-separated multiple keys) environment variable.'
        }), 500
    
    data = request.get_json()
    original_prompt = data.get('prompt', '').strip()
    
    if not original_prompt:
        return jsonify({'error': 'Prompt cannot be empty'}), 400
    
    try:
        # Complete process with restart mechanism
        max_complete_restarts = 2  # Allow 2 complete restarts (3 total attempts)
        complete_restart_attempt = 0
        
        while complete_restart_attempt <= max_complete_restarts:
            complete_restart_attempt += 1
            
            print(f"=== COMPLETE ATTEMPT {complete_restart_attempt}/{max_complete_restarts + 1} ===")
            
            # Process the prompt directly
            # Change to parent directory to execute task (for database access)
            original_cwd = os.getcwd()
            os.chdir('..')
            result = system.process_task(original_prompt)
            os.chdir(original_cwd)
            
            # Check if this attempt was successful
            if 'error' not in result and result.get('test_result') and result['test_result'].success:
                print(f"=== SUCCESS ON COMPLETE ATTEMPT {complete_restart_attempt} ===")
                break
            else:
                # This complete attempt failed
                error_msg = result.get('error', 'Unknown error')
                if result.get('test_result'):
                    error_msg = result['test_result'].error
                
                print(f"=== COMPLETE ATTEMPT {complete_restart_attempt} FAILED ===")
                print(f"Error: {error_msg}")
                
                if complete_restart_attempt <= max_complete_restarts:
                    print(f"=== PERFORMING COMPLETE RESTART ({max_complete_restarts + 1 - complete_restart_attempt} attempts remaining) ===")
                    print("Starting fresh: new prompt refinement + new code generation + new execution")
                    continue
                else:
                    print(f"=== ALL COMPLETE ATTEMPTS FAILED ===")
                    break
        
        # Debug logging
        print("=== DEBUG: Process Task Result ===")
        # print(f"Result keys: {list(result.keys())}")
        
        if 'error' in result:
            print(f"ERROR in result: {result['error']}")
            return jsonify({'error': result['error']}), 500
        
        # Debug test result
        if 'test_result' in result:
            print(f"Test result success: {result['test_result'].success}")
            # print(f"Test result raw result: {repr(result['test_result'].result)}")
            print(f"Test result error: {result['test_result'].error}")
            print(f"Test result execution time: {result['test_result'].execution_time}")
        
        # Debug generation
        if 'generation' in result:
            print(f"Generated code length: {len(result['generation'].code)}")
            print(f"Generated explanation: {result['generation'].explanation[:200]}...")
        
        # Extract retry information from the analytics
        retry_attempts = result['analytics']['generation_info']['retry_attempts'] if 'generation_info' in result['analytics'] else 1
        max_retries = result['analytics']['generation_info']['max_retries'] if 'generation_info' in result['analytics'] else 5
        
        response = {
            'success': True,
            'original_prompt': original_prompt,
            'refined_prompt': original_prompt,
            'prompt_was_refined': False,
            'complete_restart_attempts': complete_restart_attempt,
            'max_complete_restarts': max_complete_restarts + 1,
            'execution_retry_attempts': retry_attempts,
            'max_execution_retries': max_retries,
            'total_retries_used': (complete_restart_attempt - 1) * max_retries + retry_attempts,
            'had_complete_restarts': complete_restart_attempt > 1,
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
        # print(f"Response result: {repr(response['result'])}")
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

@app.route('/api/csv_data')
def get_csv_data():
    csv_path = os.path.abspath('../output.csv')
    
    if not os.path.exists(csv_path):
        return jsonify({'error': 'No CSV output file found. Run a prompt first.'}), 404
    
    # Check file size to ensure it's not empty or still being written
    try:
        file_size = os.path.getsize(csv_path)
        if file_size == 0:
            return jsonify({'error': 'CSV file is empty or still being written. Please try again.'}), 404
    except OSError:
        return jsonify({'error': 'Error accessing CSV file.'}), 404
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    sort_by = request.args.get('sort_by', 'Date')  # Default to Date sorting
    sort_order = request.args.get('sort_order', 'DESC')  # Default to DESC (latest first)
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Get total count
        total = len(df)
        
        # Apply sorting - default to Date DESC if Date column exists
        if sort_by in df.columns:
            ascending = sort_order.upper() == 'ASC'
            df = df.sort_values(by=sort_by, ascending=ascending)
        elif 'Date' in df.columns and not sort_by:
            # Fallback: if no sort specified but Date column exists, sort by Date DESC
            df = df.sort_values(by='Date', ascending=False)
            sort_by = 'Date'
            sort_order = 'DESC'
        
        # Apply pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_df = df.iloc[start_idx:end_idx]
        
        # Convert to list of dictionaries and handle NaN values
        data = paginated_df.replace({np.nan: None}).to_dict('records')
        
        return jsonify({
            'data': data,
            'columns': df.columns.tolist(),
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page,
            'sort_by': sort_by,
            'sort_order': sort_order
        })
    
    except Exception as e:
        return jsonify({'error': f'Error reading CSV: {str(e)}'}), 500

@app.route('/api/download_csv')
def download_csv():
    csv_path = os.path.abspath('../output.csv')
    
    if not os.path.exists(csv_path):
        return jsonify({'error': 'No CSV output file found. Run a prompt first.'}), 404
    
    return send_file(csv_path, as_attachment=True, download_name='output.csv')

@app.route('/api/refine_prompt', methods=['POST'])
def refine_prompt():
    """Standalone endpoint for prompt refinement"""
    if not prompt_refiner:
        return jsonify({
            'error': 'Prompt refiner not initialized. Please set GEMINI_API_KEY environment variable.'
        }), 500
    
    data = request.get_json()
    original_prompt = data.get('prompt', '').strip()
    
    if not original_prompt:
        return jsonify({'error': 'Prompt cannot be empty'}), 400
    
    try:
        refined_prompt = prompt_refiner.refine_prompt(original_prompt)
        
        return jsonify({
            'success': True,
            'original_prompt': original_prompt,
            'refined_prompt': refined_prompt,
            'was_refined': refined_prompt != original_prompt
        })
    
    except Exception as e:
        return jsonify({'error': f'Prompt refinement failed: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'system_ready': system is not None})

if __name__ == '__main__':
    # Configure Flask to ignore generated_code.py for auto-reload
    
    # Use werkzeug directly to have more control over file watching
    run_simple('0.0.0.0', 5000, app, use_reloader=True, use_debugger=True,
               extra_files=[], exclude_patterns=['**/generated_code.py'])