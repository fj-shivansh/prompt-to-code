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
DB_PATH = os.path.abspath('../historical_data_with_gains.db')

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

def generate_status_updates(original_prompt):
    """Generator function for Server-Sent Events with detailed progress tracking"""
    global stop_requested, current_process
    
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
        max_complete_restarts = 2
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
                def tracked_execute(code):
                    if stop_requested:
                        return TestResult(success=False, result=None, execution_time=0, error="Processing stopped by user")
                    return original_execute(code)
                
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
                        result_container[0] = system.process_task_streaming(original_prompt, progress_callback=streaming_callback)
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
                
                # Send final result
                max_retries = result['analytics']['generation_info']['max_retries'] if 'analytics' in result and 'generation_info' in result['analytics'] else 3
                
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
                
                yield f"data: {json.dumps({'type': 'retry_failed', 'message': f'All retries failed (retry {actual_retries} of 3)', 'retry': actual_retries, 'attempt': complete_restart_attempt, 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
                yield f"data: {json.dumps({'type': 'attempt_failed', 'message': f'Attempt {complete_restart_attempt} failed after {actual_retries} retries', 'attempt': complete_restart_attempt, 'error': error_msg[:200], 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
                
                if complete_restart_attempt <= max_complete_restarts:
                    yield f"data: {json.dumps({'type': 'restarting', 'message': f'Restarting... ({max_complete_restarts + 1 - complete_restart_attempt} attempts remaining)', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
                    continue
                else:
                    max_retries = 3  # Default value for error calculation
                    yield f"data: {json.dumps({'type': 'final_error', 'message': 'All attempts failed', 'total_attempts': complete_restart_attempt, 'total_retries': complete_restart_attempt * max_retries, 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
                    break
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'final_error', 'message': f'System error: {str(e)}', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"

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
        max_retries = result['analytics']['generation_info']['max_retries'] if 'generation_info' in result['analytics'] else 3
        
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

def calculate_nav_long_only(df, initial_amount=100000, amount_to_invest=1):
    """
    Calculate NAV (Net Asset Value) for a long-only strategy based on signals
    
    Args:
        df: DataFrame with columns Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct, Signal
        initial_amount: Initial investment amount (default 100000)
        amount_to_invest: Amount to invest multiplier (default 1)
    
    Returns:
        DataFrame with Date and NAV columns
    """
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter for signals and valid data
    df = df[(df['Signal'] == 1) & (df['Adj_Close'].notna())]
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    nav_list = [initial_amount]
    date_list = []
    
    # Group by Date (automatically gives unique dates in ascending order)
    for date, daily_df in df.groupby('Date', sort=True):
        count = len(daily_df)
        if count > 0:
            total_forward_gain = daily_df['Forward_Gain_Pct'].sum()
            avg_forward_gain = total_forward_gain / count
        else:
            avg_forward_gain = 0
        
        date_list.append(date)
        # Fixed syntax error: nav_list[-1] * (1 + amount_to_invest * avg_forward_gain)
        nav_list.append(nav_list[-1] * (1 + amount_to_invest * avg_forward_gain))
    
    # Remove the last NAV value (it's one extra)
    nav_list = nav_list[:-1]
    
    nav_df = pd.DataFrame({"Date": date_list, "NAV": nav_list})
    return nav_df

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

def generate_condition_code(condition_prompt: str) -> str:
    """Generate Python code for condition evaluation using existing CSV data"""
    
    # Get CSV headers information
    csv_headers = ""
    output_csv_path = os.path.abspath('../output.csv')
    try:
        if os.path.exists(output_csv_path):
            df = pd.read_csv(output_csv_path)
            columns_list = df.columns.tolist()
            csv_headers = f"Available columns in output.csv: {', '.join(columns_list)}"
            
            # Add sample data for better context
            if len(df) > 0:
                sample_row = df.iloc[0].to_dict()
                sample_values = {k: v for k, v in sample_row.items()}
                csv_headers += f"\n\nSample row for reference: {sample_values}"
        else:
            csv_headers = "Warning: output.csv not found. Assuming standard columns."
    except Exception as e:
        csv_headers = f"Could not read output.csv headers: {str(e)}"
    
    condition_prompt_template = f"""
You are an expert Python developer and data analyst. Generate COMPLETE, STANDALONE Python code that:

1. Reads the existing CSV file 'output.csv' (which contains calculated data from previous step)
2. Interprets the natural language condition: "{condition_prompt}"
3. Maps the condition to the correct column names from the CSV
4. Creates a new binary column called 'Signal' with values 1 (True) or 0 (False) based on the condition
5. Saves the result to 'condition_output.csv' with ALL original database columns (Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct) plus the new Signal column

CSV FILE INFO:
{csv_headers}

CONDITION INTERPRETATION GUIDELINES:
- The user provided: "{condition_prompt}"
- Map natural language terms to actual column names intelligently:
  * "10 day moving average" → look for columns like "MA10", "10_day_MA", "10_day_moving_avg", etc.
  * "5 day moving average" → look for columns like "MA5", "5_day_MA", "5_day_moving_avg", etc.
  * "price" or "stock price" → likely "Adj_Close"
  * "moving average", "MA", "average" → look for columns containing "MA", "avg", "average"
  * Be flexible with column name matching - use fuzzy matching if needed
- Convert natural language comparisons:
  * "greater than", "higher than", "above" → >
  * "less than", "lower than", "below" → <
  * "equal to", "equals" → ==
  * "greater than or equal", "at least" → >=
  * "less than or equal", "at most" → <=

IMPORTANT:
- Use pandas for all operations
- Handle missing values appropriately (treat as False for condition)  
- Include pd.set_option('display.max_rows', None)
- The condition should be evaluated row by row
- 🚨 MANDATORY: Preserve ALL original database columns with EXACT names (Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct) in the final output
- 🚨 DO NOT change column names: Use "Adj_Close" NOT "Close", "Ticker" NOT "Symbol", etc.
- Save final result sorted by Date DESC (latest first)
- If you can't find an exact column match, use the closest match and explain in the explanation
- Use EXACT column names from the CSV headers listed above after mapping
- The final condition_output.csv must contain all original columns plus the new Signal column
- 🚨 CRITICAL: Final CSV must have columns in this exact order: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct, Signal

🚨 MANDATORY TEMPLATE for final DataFrame:
```
# EXACT column order and names required:
final_df = df[['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct', 'Signal']]
final_df = final_df.sort_values('Date', ascending=False)
final_df.to_csv('condition_output.csv', index=False)
```

Expected response format (JSON):
{{
    "code": "your complete standalone Python code here as a string",
    "explanation": "your step-by-step explanation here as a string, including how you mapped the natural language to column names",
    "requirements": ["list", "of", "required", "packages", "if", "any"]
}}

NATURAL LANGUAGE CONDITION: {condition_prompt}
"""

    try:
        # Reuse existing GeminiClient from the system
        if system and system.gemini_client:
            generation = system.gemini_client.generate_code(condition_prompt_template)
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
    
    if not condition_prompt:
        return jsonify({'error': 'Condition prompt cannot be empty'}), 400
    
    # Check if output.csv exists
    output_csv_path = os.path.abspath('../output.csv')
    if not os.path.exists(output_csv_path):
        return jsonify({'error': 'No output.csv found. Please run data generation first.'}), 400
    
    try:
        # Implement retry mechanism like main code
        max_retries = 3
        generation = None
        test_result = None
        attempt = 0
        error_context = None
        
        while attempt < max_retries:
            attempt += 1
            print(f"\n=== Condition Attempt {attempt}/{max_retries} ===")
            
            try:
                # Generate condition evaluation code (with error context if retry)
                if attempt == 1:
                    generation = generate_condition_code(condition_prompt)
                else:
                    print(f"Retrying condition generation with error context: {error_context[:200]}...")
                    # Add error context to condition prompt for retry
                    retry_condition_prompt = f"{condition_prompt}\n\nPREVIOUS ATTEMPT FAILED WITH ERROR: {error_context}\n\nPlease fix the above error and try again."
                    generation = generate_condition_code(retry_condition_prompt)
                
                # Install requirements if needed
                if generation.requirements:
                    if not system.code_executor.install_requirements(generation.requirements):
                        error_context = 'Failed to install required packages'
                        continue
                
                # Execute condition code
                original_cwd = os.getcwd()
                os.chdir('..')
                
                # Write condition code to separate file
                condition_code_path = "condition_generated_code.py"
                with open(condition_code_path, "w") as f:
                    f.write(f'''#!/usr/bin/env python3
"""
Generated Condition Code Execution Module
"""

{generation.code}
''')
                
                # Execute condition code
                test_result = system.code_executor.execute_code(generation.code)
                os.chdir(original_cwd)
                
                if test_result.success:
                    print(f"=== Condition SUCCESS on attempt {attempt} ===")
                    return jsonify({
                        'success': True,
                        'condition_prompt': condition_prompt,
                        'code': generation.code,
                        'explanation': generation.explanation,
                        'requirements': generation.requirements or [],
                        'result': test_result.result,
                        'execution_time': test_result.execution_time,
                        'retry_attempts': attempt,
                        'max_retries': max_retries
                    })
                else:
                    error_context = test_result.error
                    print(f"=== Condition FAILED on attempt {attempt}: {error_context} ===")
                    if attempt == max_retries:
                        break
                    
            except Exception as e:
                error_context = str(e)
                print(f"=== Condition EXCEPTION on attempt {attempt}: {error_context} ===")
                if attempt == max_retries:
                    break
        
        # All attempts failed
        return jsonify({
            'error': f'Condition processing failed after {max_retries} attempts. Final error: {error_context}'
        }), 500
            
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
        nav_df = calculate_nav_long_only(df, initial_amount, amount_to_invest)
        
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
                'total_signals': len(nav_data),
                'investment_multiplier': amount_to_invest
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'NAV calculation failed: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'system_ready': system is not None})

if __name__ == '__main__':    
    # Use werkzeug directly to have more control over file watching
    run_simple('0.0.0.0', 5000, app, use_reloader=True, use_debugger=True,
               extra_files=[], exclude_patterns=['**/generated_code.py','**/condition_generated_code.py'])