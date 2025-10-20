"""
Prompt processing and streaming service
"""
import json
import time
import os
import pandas as pd


def generate_status_updates(original_prompt, filters, system, prompt_refiner, process_manager):
    """Generator function for Server-Sent Events with detailed progress tracking"""

    # Extract filter information for response
    selected_tickers = filters.get('selected_tickers', []) if filters else []
    ticker_count = filters.get('ticker_count', '10') if filters else '10'
    start_date = filters.get('start_date') if filters else None
    end_date = filters.get('end_date') if filters else None

    try:
        process_manager.stop_requested = False
        yield f"data: {json.dumps({'type': 'init', 'message': 'Initializing...', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"

        if not system:
            yield f"data: {json.dumps({'type': 'error', 'message': 'System not initialized'})}\n\n"
            return

        # Check for stop request
        if process_manager.stop_requested:
            yield f"data: {json.dumps({'type': 'user_stopped', 'message': 'Processing stopped by user'})}\n\n"
            return

        # Complete process with restart mechanism
        max_complete_restarts = 1  # 1 restart (2 total attempts)
        max_error_attempts = 2     # 2 error attempts (3 total per restart)
        complete_restart_attempt = 0

        while complete_restart_attempt <= max_complete_restarts:
            complete_restart_attempt += 1

            # Check for stop request before starting attempt
            if process_manager.stop_requested:
                yield f"data: {json.dumps({'type': 'user_stopped', 'message': 'Processing stopped by user'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'attempt_start', 'message': f'Starting attempt {complete_restart_attempt} of {max_complete_restarts + 1}', 'attempt': complete_restart_attempt, 'max_attempts': max_complete_restarts + 1, 'timestamp': time.strftime('%H:%M:%S')})}\n\n"

            # Process the prompt directly
            yield f"data: {json.dumps({'type': 'generating', 'message': 'Generating and executing code...', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"

            # Check for stop request before processing
            if process_manager.stop_requested:
                yield f"data: {json.dumps({'type': 'user_stopped', 'message': 'Processing stopped by user'})}\n\n"
                return

            original_cwd = os.getcwd()
            os.chdir('..')

            # Process the actual task
            with process_manager.process_lock:
                if process_manager.stop_requested:
                    os.chdir(original_cwd)
                    yield f"data: {json.dumps({'type': 'user_stopped', 'message': 'Processing stopped by user'})}\n\n"
                    return

                # Set a callback to track subprocess
                def set_current_process(proc):
                    process_manager.current_process = proc

                # Set the process callback on the code executor
                system.code_executor.process_callback = set_current_process

                # Monkey patch the system's code executor to check for stop requests
                original_execute = system.code_executor.execute_code
                def tracked_execute(code, filename=None):
                    if process_manager.stop_requested:
                        from ..models.base import TestResult
                        return TestResult(success=False, result=None, execution_time=0, error="Processing stopped by user")
                    return original_execute(code, filename)

                system.code_executor.execute_code = tracked_execute
                # Create a queue to pass updates in real-time
                from queue import Queue
                import threading
                update_queue = Queue()
                result_container = [None]

                def streaming_callback(update_str):
                    if process_manager.stop_requested:
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
                process_manager.current_process = None

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
