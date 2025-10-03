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
import signal
import subprocess
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

# Load environment variables from .env file
load_dotenv('../.env')

sys.path.append('..')
from main import PromptToCodeSystem, DatabaseManager, TestResult

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

        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def refine_prompt(self, user_request: str) -> str:
        """
        Refine a vague user request into a detailed, structured, production-ready prompt.
        """
        system_prompt = f"""
 You are an expert financial analyst and prompt engineer. 
                        Transform the following user request into a **concise, step-by-step instruction** specifying:

                        1. Calculations required
                        2. Output columns and what each should contain

                        🚨 CRITICAL REQUIREMENT: The output CSV MUST ALWAYS contain these 5 database columns AS THE FIRST 5 COLUMNS:
                        Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct
                        
                        Any additional calculated columns should come AFTER these 5 mandatory columns.
                        DO NOT rename or exclude these 5 database columns - they are required for NAV calculations.

                        Focus ONLY on calculations and column mapping. 
                        Do NOT include extra instructions, handling missing data, or formatting advice.
                        User Request:
                        {user_request}
                    """

        response = self.model.generate_content(system_prompt)
        return response.text.strip()

    def generate_condition_suggestions(self, columns: list, sample_data: dict = None) -> list:
        """
        Generate profitable trading condition suggestions based on available columns.
        """
        print(f"[generate_condition_suggestions] Called with {len(columns)} columns")
        print(f"[generate_condition_suggestions] Columns: {columns}")
        print(f"[generate_condition_suggestions] Sample data provided: {sample_data is not None}")

        system_prompt = f"""
You are an expert quantitative trading analyst. Based on the following CSV columns from a stock trading dataset, generate 5 PROFITABLE trading condition prompts that could be used to create signals for a trading strategy.

Available Columns: {', '.join(columns)}
Sample Data (first row): {sample_data if sample_data else 'Not available'}

REQUIREMENTS:
1. Each condition should be a clear, actionable trading rule
2. Focus on conditions that have potential for profitability based on technical analysis
3. Each condition should result in a Signal column (1 for True/Buy, 0 for False/No action)
4. Be specific about thresholds and comparisons
5. Return ONLY a JSON array of condition strings, no explanations

IMPORTANT: The conditions should work with the available columns only. Don't reference columns that don't exist.

Example format (return only the JSON array):
[
  "Create Signal=1 when RSI < 30 and Daily_Gain_Pct < -2, otherwise Signal=0",
  "Create Signal=1 when MA_5 > MA_20 and volume is above average, otherwise Signal=0"
]

Generate 5 profitable condition prompts as a JSON array:
"""

        print(f"[generate_condition_suggestions] Sending request to Gemini...")
        try:
            response = self.model.generate_content(system_prompt)
            print(f"[generate_condition_suggestions] Received response from Gemini")
            print(f"[generate_condition_suggestions] Response text: {response.text[:200]}...")

            import json
            # Extract JSON from response
            text = response.text.strip()
            print(f"[generate_condition_suggestions] Looking for JSON array in response...")

            # Try to find JSON array in the response
            if '[' in text and ']' in text:
                start = text.index('[')
                end = text.rindex(']') + 1
                json_str = text[start:end]
                print(f"[generate_condition_suggestions] Extracted JSON: {json_str[:200]}...")

                conditions = json.loads(json_str)
                print(f"[generate_condition_suggestions] Successfully parsed {len(conditions)} conditions")
                result = conditions[:5]  # Ensure max 5 conditions
                print(f"[generate_condition_suggestions] Returning {len(result)} conditions")
                return result
            else:
                print(f"[generate_condition_suggestions] WARNING: No JSON array found in response")
                print(f"[generate_condition_suggestions] Full response: {text}")
                return []
        except Exception as e:
            print(f"[generate_condition_suggestions] ERROR: {str(e)}")
            print(f"[generate_condition_suggestions] Exception type: {type(e).__name__}")
            import traceback
            print(f"[generate_condition_suggestions] Traceback: {traceback.format_exc()}")
            return []

app = Flask(__name__)
CORS(app)

# Get absolute path to database
DB_PATH = os.path.abspath('../historical_data_500_tickers_with_gains.db')

# Global variables for process management
current_process = None
stop_requested = False
process_lock = threading.Lock()

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

def generate_status_updates(original_prompt, filters=None):
    """Generator function for Server-Sent Events with detailed progress tracking"""
    global stop_requested, current_process

    # Extract filter information for response
    selected_tickers = filters.get('selected_tickers', []) if filters else []
    ticker_count = filters.get('ticker_count', '10') if filters else '10'
    start_date = filters.get('start_date') if filters else None
    end_date = filters.get('end_date') if filters else None

    try:
        stop_requested = False
        yield f"data: {json.dumps({'type': 'init', 'message': 'Initializing...', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
        
        if not system:
            yield f"data: {json.dumps({'type': 'error', 'message': 'System not initialized'})}\n\n"
            return
        
        # Check for stop request
        if stop_requested:
            yield f"data: {json.dumps({'type': 'user_stopped', 'message': 'Processing stopped by user'})}\n\n"
            return
        
        # Complete process with restart mechanism
        max_complete_restarts = 1  # 1 restart (2 total attempts)
        max_error_attempts = 2     # 2 error attempts (3 total per restart)
        complete_restart_attempt = 0
        
        while complete_restart_attempt <= max_complete_restarts:
            complete_restart_attempt += 1
            
            # Check for stop request before starting attempt
            if stop_requested:
                yield f"data: {json.dumps({'type': 'user_stopped', 'message': 'Processing stopped by user'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'attempt_start', 'message': f'Starting attempt {complete_restart_attempt} of {max_complete_restarts + 1}', 'attempt': complete_restart_attempt, 'max_attempts': max_complete_restarts + 1, 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
            
            # Process the prompt directly
            yield f"data: {json.dumps({'type': 'generating', 'message': 'Generating and executing code...', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
            
            # Check for stop request before processing
            if stop_requested:
                yield f"data: {json.dumps({'type': 'user_stopped', 'message': 'Processing stopped by user'})}\n\n"
                return
            
            original_cwd = os.getcwd()
            os.chdir('..')
            
            # Process the actual task
            with process_lock:
                if stop_requested:
                    os.chdir(original_cwd)
                    yield f"data: {json.dumps({'type': 'user_stopped', 'message': 'Processing stopped by user'})}\n\n"
                    return
                    
                # Set a callback to track subprocess
                def set_current_process(proc):
                    global current_process
                    current_process = proc
                
                # Set the process callback on the code executor
                system.code_executor.process_callback = set_current_process
                
                # Monkey patch the system's code executor to check for stop requests
                original_execute = system.code_executor.execute_code
                def tracked_execute(code, filename=None):
                    if stop_requested:
                        return TestResult(success=False, result=None, execution_time=0, error="Processing stopped by user")
                    return original_execute(code, filename)
                
                system.code_executor.execute_code = tracked_execute
                # Create a queue to pass updates in real-time
                from queue import Queue
                import threading
                update_queue = Queue()
                result_container = [None]
                
                def streaming_callback(update_str):
                    if stop_requested:
                        return
                    update_queue.put(update_str)
                
                def run_processing():
                    try:
                        result_container[0] = system.process_task_streaming(original_prompt, max_complete_restarts, max_error_attempts, progress_callback=streaming_callback, filters=filters)
                        update_queue.put("PROCESSING_COMPLETE")
                    except Exception as e:
                        update_queue.put(f"ERROR:{str(e)}")
                
                # Start processing in a separate thread
                processing_thread = threading.Thread(target=run_processing)
                processing_thread.start()
                
                # Stream updates in real-time
                while True:
                    try:
                        update = update_queue.get(timeout=1.0)
                        if update == "PROCESSING_COMPLETE":
                            break
                        elif update.startswith("ERROR:"):
                            yield f"data: {json.dumps({'type': 'error', 'message': update[6:], 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
                            break
                        elif update.startswith('data: '):
                            try:
                                data = json.loads(update[6:].split('\n')[0])
                                data['timestamp'] = time.strftime('%H:%M:%S')
                                yield f"data: {json.dumps(data)}\n\n"
                            except:
                                pass
                    except:
                        # Timeout - check if processing is still alive
                        if not processing_thread.is_alive():
                            break
                        continue
                
                processing_thread.join()
                result = result_container[0]
                
                # Clean up
                system.code_executor.execute_code = original_execute  # Restore original method
                system.code_executor.process_callback = None
                current_process = None
            
            os.chdir(original_cwd)
            
            # Get actual retry count from result
            actual_retries = result['analytics']['generation_info']['retry_attempts'] if result and 'analytics' in result and 'generation_info' in result['analytics'] else 1
            
            # Get token information from result
            tokens = None
            if 'generation' in result and hasattr(result['generation'], 'tokens') and result['generation'].tokens:
                tokens = result['generation'].tokens
            
            # Show execution summary
            execution_msg = {
                'type': 'execution_complete', 
                'message': f'Code execution completed (retry {actual_retries} of 3)', 
                'retry': actual_retries, 
                'max_retries': 3, 
                'attempt': complete_restart_attempt, 
                'timestamp': time.strftime('%H:%M:%S')
            }
            if tokens:
                execution_msg['tokens'] = tokens
            yield f"data: {json.dumps(execution_msg)}\n\n"
            
            # Show final attempt status
            if result and 'error' not in result and result.get('test_result') and result['test_result'].success:
                yield f"data: {json.dumps({'type': 'attempt_success', 'message': f'Attempt {complete_restart_attempt} succeeded!', 'attempt': complete_restart_attempt, 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
                
                # Generate condition suggestions based on output CSV
                suggested_conditions = []
                if prompt_refiner:
                    try:
                        print("=" * 80)
                        print("CONDITION SUGGESTIONS GENERATION START")
                        print("=" * 80)
                        yield f"data: {json.dumps({'type': 'generating_suggestions', 'message': 'Generating profitable condition suggestions...', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"

                        # Read output CSV to get columns
                        output_csv_path = os.path.abspath('../output.csv')
                        print(f"Looking for output CSV at: {output_csv_path}")
                        print(f"CSV exists: {os.path.exists(output_csv_path)}")

                        if os.path.exists(output_csv_path):
                            print("Reading CSV file...")
                            df = pd.read_csv(output_csv_path)
                            columns = df.columns.tolist()
                            sample_data = df.iloc[0].to_dict() if len(df) > 0 else None

                            print(f"CSV columns: {columns}")
                            print(f"Sample data: {sample_data}")
                            print(f"Calling generate_condition_suggestions...")

                            suggested_conditions = prompt_refiner.generate_condition_suggestions(columns, sample_data)

                            print(f"Suggested conditions received: {suggested_conditions}")
                            print(f"Number of suggestions: {len(suggested_conditions)}")

                            yield f"data: {json.dumps({'type': 'suggestions_generated', 'message': f'Generated {len(suggested_conditions)} condition suggestions', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
                        else:
                            print(f"WARNING: output.csv not found at {output_csv_path}")
                            yield f"data: {json.dumps({'type': 'suggestions_error', 'message': 'output.csv not found', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"

                        print("=" * 80)
                        print("CONDITION SUGGESTIONS GENERATION END")
                        print("=" * 80)
                    except Exception as e:
                        print(f"ERROR generating condition suggestions: {str(e)}")
                        print(f"Exception type: {type(e).__name__}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        yield f"data: {json.dumps({'type': 'suggestions_error', 'message': f'Failed to generate suggestions: {str(e)}', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
                else:
                    print("WARNING: prompt_refiner is None, cannot generate suggestions")

                # Send final result
                max_retries = result['analytics']['generation_info']['max_retries'] if 'analytics' in result and 'generation_info' in result['analytics'] else (max_complete_restarts + 1) * (max_error_attempts + 1)

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
                    'analytics': result['analytics'],
                    'selected_tickers': selected_tickers,
                    'ticker_count': ticker_count,
                    'date_range': {'start_date': start_date, 'end_date': end_date} if start_date and end_date else None,
                    'suggested_conditions': suggested_conditions
                }
                
                yield f"data: {json.dumps({'type': 'final_result', 'data': response})}\n\n"
                yield f"data: {json.dumps({'type': 'connection_close'})}\n\n"
                return
            else:
                # This complete attempt failed
                error_msg = result.get('error', 'Unknown error') if result else 'System error'
                if result and result.get('test_result'):
                    error_msg = result['test_result'].error
                
                yield f"data: {json.dumps({'type': 'retry_failed', 'message': f'All retries failed (retry {actual_retries} of 3)', 'retry': actual_retries, 'attempt': complete_restart_attempt, 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
                yield f"data: {json.dumps({'type': 'attempt_failed', 'message': f'Attempt {complete_restart_attempt} failed after {actual_retries} retries', 'attempt': complete_restart_attempt, 'error': error_msg[:200], 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
                
                if complete_restart_attempt <= max_complete_restarts:
                    yield f"data: {json.dumps({'type': 'restarting', 'message': f'Restarting... ({max_complete_restarts + 1 - complete_restart_attempt} attempts remaining)', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
                    continue
                else:
                    max_retries = (max_complete_restarts + 1) * (max_error_attempts + 1)  # Default value for error calculation
                    yield f"data: {json.dumps({'type': 'final_error', 'message': 'All attempts failed', 'total_attempts': complete_restart_attempt, 'total_retries': complete_restart_attempt * max_retries, 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
                    yield f"data: {json.dumps({'type': 'connection_close'})}\n\n"
                    return
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'final_error', 'message': f'System error: {str(e)}', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
        yield f"data: {json.dumps({'type': 'connection_close'})}\n\n"

@app.route('/api/process_prompt_stream', methods=['POST'])
def process_prompt_stream():
    """Server-Sent Events endpoint for real-time status updates"""
    data = request.get_json()
    original_prompt = data.get('prompt', '').strip()
    
    # Extract filter parameters and pre-generate random tickers
    ticker_count = data.get('ticker_count', '10')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    # Pre-generate random tickers so both LLMs use the SAME set
    selected_tickers = []
    if ticker_count != 'all':
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get random tickers with date filtering
            ticker_query = """
                SELECT DISTINCT Ticker 
                FROM stock_data 
                WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL
            """
            ticker_params = []
            
            if start_date and end_date:
                ticker_query += " AND Date >= ? AND Date <= ?"
                ticker_params.extend([start_date, end_date])
            
            ticker_query += " ORDER BY RANDOM() LIMIT ?"
            ticker_params.append(int(ticker_count))
            
            cursor.execute(ticker_query, ticker_params)
            selected_tickers = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            print(f"Pre-generated {len(selected_tickers)} random tickers for both LLMs: {selected_tickers}")
            
        except Exception as e:
            print(f"Error pre-generating tickers: {e}")
            selected_tickers = []
    
    filters = {
        'ticker_count': ticker_count,
        'start_date': start_date,
        'end_date': end_date,
        'selected_tickers': selected_tickers  # Both LLMs will use these SAME tickers
    }
    
    if not original_prompt:
        return jsonify({'error': 'Prompt cannot be empty'}), 400
    
    def event_stream():
        for update in generate_status_updates(original_prompt, filters):
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
        max_complete_restarts = 1  # Allow 1 complete restart (2 total attempts)
        max_error_attempts = 2     # Allow 2 error attempts per restart (3 total per restart)
        complete_restart_attempt = 0
        
        while complete_restart_attempt <= max_complete_restarts:
            complete_restart_attempt += 1
            
            print(f"=== COMPLETE ATTEMPT {complete_restart_attempt}/{max_complete_restarts + 1} ===")
            
            # Process the prompt directly
            # Change to parent directory to execute task (for database access)
            original_cwd = os.getcwd()
            os.chdir('..')
            result = system.process_task(original_prompt, max_complete_restarts, max_error_attempts)
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
        max_retries = result['analytics']['generation_info']['max_retries'] if 'generation_info' in result['analytics'] else (max_complete_restarts + 1) * (max_error_attempts + 1)
        
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
    valid_sort_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']
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
            SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct 
            FROM stock_data 
            {where_clause}
            ORDER BY {sort_by} {sort_order}
            LIMIT ? OFFSET ?
        """
        cursor.execute(data_query, params + [per_page, offset])
        
        rows = cursor.fetchall()
        conn.close()
        
        data = [
            {"Date": row[0], "Ticker": row[1], "Adj_Close": row[2], "Daily_Gain_Pct": row[3], "Forward_Gain_Pct": row[4]}
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
            WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL
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

@app.route('/api/stop_processing', methods=['POST'])
def stop_processing():
    """Endpoint to stop current processing"""
    global stop_requested, current_process
    
    try:
        with process_lock:
            stop_requested = True
            
            # If there's a current subprocess, terminate it
            if current_process and current_process.poll() is None:
                current_process.terminate()
                try:
                    current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    current_process.kill()
                    current_process.wait()
        
        return jsonify({
            'success': True,
            'message': 'Stop request sent successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to stop processing: {str(e)}'
        }), 500

def calculate_nav_long_only(df,initial_amount=100000,amount_to_invest=1,max_position_each_ticker=1):

    df['Date'] = pd.to_datetime(df['Date'])


    df = df.sort_values('Date').reset_index(drop=True)

    nav_list = [initial_amount]
    date_list = []

    # Group by Date (automatically gives unique dates in ascending order)
    for date, daily_df in df.groupby('Date', sort=True):
        
        daily_df = daily_df[daily_df["Signal"]==1]
        count = len(daily_df)
        if count > 0:
            total_forward_gain = daily_df['Forward_Gain_Pct'].sum()
            
            percentage_of_each_ticker = 1/count
            percentage_of_each_ticker = min(percentage_of_each_ticker,max_position_each_ticker)
            avg_forward_gain = total_forward_gain*percentage_of_each_ticker
        else:
            avg_forward_gain = 0
        date_list.append(date)
        nav_list.append(nav_list[-1]*(1+amount_to_invest*avg_forward_gain))

    nav_list = nav_list[:-1]
    

    nav_df = pd.DataFrame({"Date":date_list,"NAV":nav_list})
    start_val = nav_df["NAV"].iloc[0]
    end_val = nav_df["NAV"].iloc[-1]

    annual_return = ((end_val/start_val)**(250/nav_df.shape[0])-1)
    annual_return = float(round(annual_return * 100, 2))
    print(nav_df.shape[0])

    rolling_max = nav_df["NAV"].cummax()
    drawdown = (nav_df["NAV"] - rolling_max) / rolling_max
    max_drawdown = float(round(-drawdown.min()*100,2))
    ratio = float(round((annual_return/max_drawdown)*100,2))
    return nav_df,annual_return,max_drawdown,ratio

def generate_nav_graph(nav_df):
    """Generate a base64-encoded NAV graph image"""
    plt.figure(figsize=(12, 6))
    plt.plot(nav_df['Date'], nav_df['NAV'], linewidth=2, color='#2196F3')
    plt.title('Portfolio NAV Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('NAV ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    graph_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return graph_base64

def generate_condition_code(condition_prompt: str, output_file: str = "condition_output.csv") -> str:
    """Generate Python code for condition evaluation using existing CSV data"""
    
    # Get CSV headers information and actual column names
    csv_headers = ""
    actual_columns = []
    
    # Try multiple possible paths for output.csv
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    possible_paths = [
        os.path.abspath('../output.csv'),
        os.path.abspath('./output.csv'),
        os.path.abspath('output.csv'),
        os.path.join(current_dir, 'output.csv'),
        os.path.join(current_dir, '..', 'output.csv')
    ]
    
    print(f"Will try these paths for output.csv:")
    for path in possible_paths:
        print(f"  - {path} (exists: {os.path.exists(path)})")
    
    csv_found = False
    for output_csv_path in possible_paths:
        try:
            print(f"Checking for CSV at: {output_csv_path}")
            if os.path.exists(output_csv_path):
                df = pd.read_csv(output_csv_path)
                actual_columns = df.columns.tolist()
                csv_headers = f"Available columns in output.csv: {', '.join(actual_columns)}"
                
                # Add sample data for better context
                if len(df) > 0:
                    sample_row = df.iloc[0].to_dict()
                    sample_values = {k: v for k, v in sample_row.items()}
                    csv_headers += f"\n\nSample row for reference: {sample_values}"
                csv_found = True
                print(f"CSV found at: {output_csv_path}")
                print(f"Columns found: {actual_columns}")
                break
        except Exception as e:
            print(f"Error reading CSV at {output_csv_path}: {str(e)}")
            continue
    
    if not csv_found:
        print("ERROR: Could not find output.csv at any expected location!")
        print("Available files in current directory:")
        try:
            files = os.listdir(current_dir)
            for f in files:
                if f.endswith('.csv'):
                    print(f"  - {f}")
        except Exception as e:
            print(f"  Error listing files: {e}")
        
        csv_headers = "ERROR: output.csv not found. Cannot determine available columns."
        actual_columns = ["Date", "Ticker", "Adj_Close", "Daily_Gain_Pct", "Forward_Gain_Pct"]
        print(f"Using default columns: {actual_columns}")
    
    condition_prompt_template = f"""
You are an expert Python developer. Generate CLEAN, EXECUTABLE Python code that:

1. Reads the CSV file 'output.csv'
2. Interprets the condition: "{condition_prompt}"
3. Creates a 'Signal' column (1 for True, 0 for False)
4. Saves result to '{output_file}'

AVAILABLE COLUMNS IN output.csv:
{csv_headers}

🚨 CRITICAL: The available columns are EXACTLY: {actual_columns}
🚨 DO NOT use any columns not in this list: {actual_columns}
🚨 DO NOT assume columns like '10_Day_MA', '5_Day_MA', 'MA10', 'MA5' exist unless they are in the list above

IMPORTANT CODE STYLE REQUIREMENTS:
- Write CLEAN Python code suitable for programmatic execution
- DO NOT include shebang lines (#!/usr/bin/env python3)
- DO NOT include docstring headers or module comments
- DO NOT use print() statements for user output
- DO NOT use exit() or sys.exit()
- DO NOT include try/except blocks for file operations
- Write simple, direct pandas operations
- Only import required libraries (pandas)

Required output format:
- Must include ALL original columns: {actual_columns}
- Plus new 'Signal' column
- Final column order: {actual_columns + ['Signal']}

🚨🚨🚨 MANDATORY DATE FORMAT REQUIREMENT 🚨🚨🚨:
- The Date column MUST be in format "YYYY-MM-DD" (e.g., "2025-09-11")
- DO NOT include timestamps or time portions (no "00:00:00")
- Use this exact code before saving: df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
- This ensures both LLMs produce identical date formats

🚨🚨🚨 MANDATORY SORTING REQUIREMENT 🚨🚨🚨:
- The output CSV MUST be sorted by Date in DESCENDING order (latest first)
- Use this exact code before saving: df = df.sort_values('Date', ascending=False)

Expected response (JSON):
{{
    "code": "your complete Python code as string",
    "explanation": "brief explanation",
    "requirements": ["pandas"]
}}

CONDITION: {condition_prompt}
"""

    try:
        # Log the full prompt being sent to the LLM
        print("=" * 80)
        print("CONDITION PROMPT BEING SENT TO LLM:")
        print("=" * 80)
        print(condition_prompt_template)
        print("=" * 80)
        
        # Reuse existing GeminiClient from the system
        if system and system.gemini_client:
            generation = system.gemini_client.generate_code(condition_prompt_template)
            print("=" * 80)
            print("LLM GENERATED CODE:")
            print("=" * 80)
            print(generation.code)
            print("=" * 80)
            return generation
        else:
            raise ValueError("System not initialized")
    except Exception as e:
        raise ValueError(f"Failed to generate condition code: {str(e)}")

@app.route('/api/process_condition', methods=['POST'])
def process_condition():
    """Process condition prompt using existing CSV output"""
    if not system:
        return jsonify({
            'error': 'System not initialized. Please set GEMINI_API_KEY environment variable.'
        }), 500
    
    data = request.get_json()
    condition_prompt = data.get('condition', '').strip()
    
    # Extract filter parameters and pre-generate random tickers
    ticker_count = data.get('ticker_count', '10')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    # Pre-generate random tickers so both LLMs use the SAME set
    selected_tickers = []
    if ticker_count != 'all':
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get random tickers with date filtering
            ticker_query = """
                SELECT DISTINCT Ticker 
                FROM stock_data 
                WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL
            """
            ticker_params = []
            
            if start_date and end_date:
                ticker_query += " AND Date >= ? AND Date <= ?"
                ticker_params.extend([start_date, end_date])
            
            ticker_query += " ORDER BY RANDOM() LIMIT ?"
            ticker_params.append(int(ticker_count))
            
            cursor.execute(ticker_query, ticker_params)
            selected_tickers = [row[0] for row in cursor.fetchall()]
            conn.close()
            
        except Exception as e:
            print(f"Error pre-generating tickers for condition: {e}")
            selected_tickers = []
    
    filters = {
        'ticker_count': ticker_count,
        'start_date': start_date,
        'end_date': end_date,
        'selected_tickers': selected_tickers
    }
    
    if not condition_prompt:
        return jsonify({'error': 'Condition prompt cannot be empty'}), 400
    
    # Check if output.csv exists
    output_csv_path = os.path.abspath('../output.csv')
    if not os.path.exists(output_csv_path):
        return jsonify({'error': 'No output.csv found. Please run data generation first.'}), 400
    
    try:
        # Use parallel processing for condition evaluation
        original_cwd = os.getcwd()
        os.chdir('..')
        
        result = system.process_condition_parallel(condition_prompt, max_complete_restarts=1, max_error_attempts=2)
        
        os.chdir(original_cwd)
        
        if "error" in result:
            return jsonify({'error': result['error']}), 500
        
        # Extract retry information from the analytics
        retry_attempts = result['analytics']['generation_info']['retry_attempts'] if 'analytics' in result and 'generation_info' in result['analytics'] else 1
        max_retries = result['analytics']['generation_info']['max_retries'] if 'analytics' in result and 'generation_info' in result['analytics'] else 3
        
        response = {
            'success': True,
            'condition_prompt': condition_prompt,
            'code': result['generation'].code,
            'explanation': result['generation'].explanation,
            'requirements': result['generation'].requirements or [],
            'result': result['test_result'].result if result['test_result'].success else None,
            'execution_time': result['test_result'].execution_time,
            'retry_attempts': retry_attempts,
            'max_retries': max_retries,
            'parallel_mode': True,
            'successful_llms': result['analytics']['generation_info'].get('successful_llms', 1),
            'csvs_identical': result['analytics']['generation_info'].get('csvs_identical', False),
            'analytics': result['analytics'],
            'selected_tickers': selected_tickers,
            'ticker_count': ticker_count,
            'date_range': {'start_date': start_date, 'end_date': end_date} if start_date and end_date else None
        }
        
        return jsonify(response)
            
    except Exception as e:
        return jsonify({'error': f'Condition processing failed: {str(e)}'}), 500

@app.route('/api/condition_csv_data')
def get_condition_csv_data():
    """Get condition CSV data with pagination and sorting"""
    csv_path = os.path.abspath('../condition_output.csv')
    
    if not os.path.exists(csv_path):
        return jsonify({'error': 'No condition output file found. Process a condition first.'}), 404
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    sort_by = request.args.get('sort_by', 'Date')
    sort_order = request.args.get('sort_order', 'DESC')
    
    try:
        df = pd.read_csv(csv_path)
        total = len(df)
        
        # Apply sorting
        if sort_by in df.columns:
            ascending = sort_order.upper() == 'ASC'
            df = df.sort_values(by=sort_by, ascending=ascending)
        elif 'Date' in df.columns:
            df = df.sort_values(by='Date', ascending=False)
            sort_by = 'Date'
            sort_order = 'DESC'
        
        # Apply pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_df = df.iloc[start_idx:end_idx]
        
        # Convert to list of dictionaries
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
        return jsonify({'error': f'Error reading condition CSV: {str(e)}'}), 500

@app.route('/api/download_condition_csv')
def download_condition_csv():
    """Download condition output CSV file"""
    csv_path = os.path.abspath('../condition_output.csv')
    
    if not os.path.exists(csv_path):
        return jsonify({'error': 'No condition output file found.'}), 404
    
    return send_file(csv_path, as_attachment=True, download_name='condition_output.csv')

@app.route('/api/calculate_nav', methods=['POST'])
def calculate_nav():
    """Calculate NAV based on condition results with signals"""
    # Check if condition_output.csv exists
    csv_path = os.path.abspath('../condition_output.csv')
    
    if not os.path.exists(csv_path):
        return jsonify({'error': 'No condition output file found. Process a condition first.'}), 404
    
    try:
        data = request.get_json()
        initial_amount = data.get('initial_amount', 100000)
        amount_to_invest = data.get('amount_to_invest', 1)
        max_position_each_ticker = data.get('max_position_each_ticker', 1)
        
        # Read condition CSV data
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct', 'Signal']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({
                'error': f'Missing required columns: {missing_columns}. Ensure condition processing generated all required columns.'
            }), 400
        
        # Calculate NAV
        nav_df, annual_return, max_drawdown, ratio = calculate_nav_long_only(df, initial_amount, amount_to_invest, max_position_each_ticker)
        
        if nav_df.empty:
            return jsonify({
                'error': 'No signals found in the data. Make sure your condition generates signals (Signal=1).'
            }), 400
        
        # Generate graph
        graph_base64 = generate_nav_graph(nav_df)
        
        # Convert NAV data for frontend
        nav_data = nav_df.to_dict('records')
        for record in nav_data:
            record['Date'] = record['Date'].strftime('%Y-%m-%d') if hasattr(record['Date'], 'strftime') else str(record['Date'])
        
        # Calculate performance metrics
        initial_nav = nav_data[0]['NAV'] if nav_data else initial_amount
        final_nav = nav_data[-1]['NAV'] if nav_data else initial_amount
        total_return = ((final_nav - initial_nav) / initial_nav) * 100
        
        return jsonify({
            'success': True,
            'nav_data': nav_data,
            'graph_base64': graph_base64,
            'metrics': {
                'initial_amount': initial_amount,
                'final_nav': final_nav,
                'total_return_pct': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'ratio': ratio,
                'total_signals': len(nav_data),
                'investment_multiplier': amount_to_invest,
                'max_position_each_ticker': max_position_each_ticker
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'NAV calculation failed: {str(e)}'}), 500

@app.route('/api/compare_csv', methods=['POST'])
def compare_csv():
    """Compare the two generated CSV files and get similarity analysis"""
    if not system:
        return jsonify({
            'error': 'System not initialized. Please set GEMINI_API_KEY environment variable.'
        }), 500
    
    try:
        # Change to parent directory to access CSV files
        original_cwd = os.getcwd()
        os.chdir('..')
        
        result = system.compare_csv_files()
        
        os.chdir(original_cwd)
        
        if "error" in result:
            return jsonify({'error': result['error']}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'CSV comparison failed: {str(e)}'}), 500

@app.route('/api/date_range')
def get_available_date_range():
    """Get min/max dates available in database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MIN(Date) as min_date, MAX(Date) as max_date 
            FROM stock_data 
            WHERE Adj_Close IS NOT NULL
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        return jsonify({
            'min_date': result[0],
            'max_date': result[1]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/database_stats')
def get_database_stats():
    """Get database statistics for 500-ticker dataset"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get total stats
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT Ticker) as total_tickers,
                COUNT(*) as total_rows,
                MIN(Date) as min_date,
                MAX(Date) as max_date
            FROM stock_data 
            WHERE Adj_Close IS NOT NULL
        """)
        
        stats = cursor.fetchone()
        conn.close()
        
        return jsonify({
            'total_tickers': stats[0],
            'total_rows': stats[1],
            'min_date': stats[2],
            'max_date': stats[3],
            'avg_rows_per_ticker': stats[1] // stats[0] if stats[0] > 0 else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/random_tickers', methods=['POST'])
def get_random_tickers():
    """Optimized for 500-ticker database"""
    data = request.get_json()
    ticker_count = data.get('ticker_count', '10')  # Default to 10
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # First, get count of available tickers in date range
        count_query = """
            SELECT COUNT(DISTINCT Ticker) 
            FROM stock_data 
            WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL
        """
        count_params = []
        
        if start_date and end_date:
            count_query += " AND Date >= ? AND Date <= ?"
            count_params.extend([start_date, end_date])
        
        cursor.execute(count_query, count_params)
        available_count = cursor.fetchone()[0]
        
        # Adjust ticker_count if requesting more than available
        if ticker_count != 'all':
            requested_count = int(ticker_count)
            actual_count = min(requested_count, available_count)
        else:
            actual_count = available_count
        
        # Get random tickers efficiently
        ticker_query = """
            SELECT DISTINCT Ticker 
            FROM stock_data 
            WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL
        """
        ticker_params = []
        
        if start_date and end_date:
            ticker_query += " AND Date >= ? AND Date <= ?"
            ticker_params.extend([start_date, end_date])
        
        if ticker_count != 'all':
            ticker_query += " ORDER BY RANDOM() LIMIT ?"
            ticker_params.append(actual_count)
        else:
            ticker_query += " ORDER BY Ticker"
        
        cursor.execute(ticker_query, ticker_params)
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({
            'tickers': tickers,
            'count': len(tickers),
            'available_count': available_count,
            'requested_count': ticker_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'system_ready': system is not None})

if __name__ == '__main__':    
    # Use werkzeug directly to have more control over file watching
    run_simple('0.0.0.0', 5000, app, use_reloader=True, use_debugger=True,
               extra_files=[], exclude_patterns=[
                   '**/generated_code.py',
                   '**/condition_generated_code_llm1.py', 
                   '**/condition_generated_code_llm2.py',
                   '**/generated_code_llm1.py',
                   '**/generated_code_llm2.py',
                   '**/csv_comparison_code.py',
                   '**/output.csv',
                   '**/output_llm1.csv',
                   '**/output_llm2.csv',
                   '**/condition_output.csv',
                   '**/similarity_result.txt'
               ])