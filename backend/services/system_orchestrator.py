"""
Main system orchestrator for prompt-to-code processing
"""
import os
import json
from typing import Optional, Dict, Any
from ..database.manager import DatabaseManager
from .gemini_client import GeminiClient
from .code_executor import CodeExecutor
from .analytics import Analytics


class PromptToCodeSystem:
    """Main system orchestrator for all prompt-to-code operations"""

    def __init__(self, api_key: Optional[str] = None):
        self.db_manager = DatabaseManager()
        self.gemini_client = GeminiClient(api_key)
        self.code_executor = CodeExecutor()
        self.generations = []
        self.test_results = []

    def process_task_streaming(self, task: str, max_complete_restarts: int = 1,
                              max_error_attempts: int = 2, progress_callback=None, filters=None):
        """Process task with streaming progress updates using parallel processing"""

        if progress_callback:
            progress_callback(f"data: {json.dumps({'type': 'task_start', 'message': 'Starting parallel processing...'})}\n\n")

        if progress_callback:
            progress_callback(f"data: {json.dumps({'type': 'parallel_start', 'message': 'Running 2 LLMs in parallel...'})}\n\n")

        # Use the parallel processing method
        result = self.generate_and_execute_parallel(task, max_complete_restarts, max_error_attempts, filters, progress_callback)

        if "error" in result:
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'error', 'message': result['error']})}\n\n")
            return result

        # Update progress based on results
        if result.get("both_succeeded"):
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'comparison_complete', 'message': 'Both LLMs succeeded. Comparison completed.'})}\n\n")
        else:
            if progress_callback:
                selected_llm = result["analytics"]["generation_info"].get("selected_llm", "unknown")
                progress_callback(f"data: {json.dumps({'type': 'single_success', 'message': f'One LLM succeeded: {selected_llm}'})}\n\n")

        return result

    def process_task(self, task: str, max_complete_restarts: int = 1, max_error_attempts: int = 2) -> Dict[str, Any]:
        """Process a complete task with parallel LLM processing"""
        return self.generate_and_execute_parallel(task, max_complete_restarts, max_error_attempts)

    def generate_and_execute_parallel(self, task: str, max_complete_restarts: int = 1,
                                     max_error_attempts: int = 2, filters=None, progress_callback=None) -> Dict[str, Any]:
        """Generate code with 2 parallel LLM calls and execute both"""
        return self.parallel_process_generic(task, "prompt", max_complete_restarts, max_error_attempts, filters, progress_callback)

    def process_condition_parallel(self, task: str, max_complete_restarts: int = 1,
                                  max_error_attempts: int = 2, progress_callback=None) -> Dict[str, Any]:
        """Process condition with 2 parallel LLM calls"""
        return self.parallel_process_generic(task, "condition", max_complete_restarts, max_error_attempts, None, progress_callback)

    def parallel_process_generic(self, task: str, task_type: str = "prompt", max_complete_restarts: int = 1,
                                max_error_attempts: int = 2, filters=None, progress_callback=None) -> Dict[str, Any]:
        """
        Generic parallel processing for both prompts and conditions
        This is the main processing logic that runs 2 LLMs in parallel
        """
        # Check if parallel mode is enabled
        from ..config.settings import Settings

        if not Settings.ENABLE_PARALLEL_LLMS:
            print(f"⚠️  Parallel mode DISABLED - running single LLM")
            return self._single_llm_process(task, task_type, max_complete_restarts, max_error_attempts, filters, progress_callback)

        print(f"Starting parallel processing for {task_type}: {task}")

        if progress_callback:
            progress_callback(f"data: {json.dumps({'type': 'parallel_init', 'message': f'Initializing parallel {task_type} processing...'})}\n\n")

        # Configure based on task type
        if task_type == "condition":
            output_file1 = "condition_output_llm1.csv"
            output_file2 = "condition_output_llm2.csv"
            final_output = "condition_output.csv"
            filename_prefix = "condition_generated_code"
        else:  # prompt
            output_file1 = "output_llm1.csv"
            output_file2 = "output_llm2.csv"
            final_output = "output.csv"
            filename_prefix = "generated_code"

        # Load data for analytics (skip for condition processing)
        data_stats = {}
        if task_type == "prompt":
            print("Loading test data...")
            data = self.db_manager.get_all_data()
            data_stats = {
                "total_records": len(data),
                "unique_tickers": len(set(d["Ticker"] for d in data)),
                "date_range": {
                    "start": min(d["Date"] for d in data),
                    "end": max(d["Date"] for d in data)
                }
            }

        # Try up to 3 complete attempts to get identical CSVs
        max_identity_attempts = 3
        for identity_attempt in range(max_identity_attempts):
            print(f"\n=== Identity Attempt {identity_attempt + 1}/{max_identity_attempts} ===")
            results = {"llm1": None, "llm2": None}

            def process_llm(llm_id: str, output_file: str):
                """Process single LLM generation and execution"""
                print(f"Starting {llm_id} processing...")
                if progress_callback:
                    progress_callback(f"data: {json.dumps({'type': 'llm_processing', 'message': f'{llm_id.upper()} processing started', 'llm': llm_id})}\n\n")

                # Complete restart loop
                for complete_restart in range(max_complete_restarts + 1):
                    print(f"{llm_id} - Complete restart {complete_restart + 1}/{max_complete_restarts + 1}")
                    if progress_callback:
                        progress_callback(f"data: {json.dumps({'type': 'llm_restart', 'message': f'{llm_id.upper()} restart {complete_restart + 1}/{max_complete_restarts + 1}', 'llm': llm_id, 'restart': complete_restart + 1})}\n\n")

                    previous_error = None
                    previous_code = None

                    # Error attempt loop within each complete restart
                    for error_attempt in range(1, max_error_attempts + 2):
                        try:
                            total_attempt = complete_restart * (max_error_attempts + 1) + error_attempt
                            print(f"{llm_id} - Error attempt {error_attempt}/{max_error_attempts + 1} (total attempt {total_attempt})")
                            if progress_callback:
                                progress_callback(f"data: {json.dumps({'type': 'llm_attempt', 'message': f'{llm_id.upper()} attempt {error_attempt}/{max_error_attempts + 1}', 'llm': llm_id, 'attempt': error_attempt, 'total_attempt': total_attempt})}\n\n")

                            # Generate code
                            if progress_callback:
                                progress_callback(f"data: {json.dumps({'type': 'llm_generating', 'message': f'{llm_id.upper()} generating code...', 'llm': llm_id})}\n\n")

                            if task_type == "condition":
                                # Use condition generation
                                from ..utils.condition_code_generator import generate_condition_code
                                if error_attempt == 1:
                                    generation = generate_condition_code(task, self, output_file)
                                else:
                                    retry_prompt = f"{task}\n\nPREVIOUS ATTEMPT FAILED WITH ERROR: {previous_error}\n\nPlease fix the above error and try again."
                                    generation = generate_condition_code(retry_prompt, self, output_file)
                            else:
                                # Use regular prompt generation
                                if error_attempt == 1:
                                    ticker_filters = None
                                    date_filters = None

                                    if filters:
                                        if filters.get('selected_tickers') or filters.get('ticker_count'):
                                            ticker_filters = {
                                                'selected_tickers': filters.get('selected_tickers', []),
                                                'ticker_count': filters.get('ticker_count')
                                            }

                                        if filters.get('start_date') or filters.get('end_date'):
                                            date_filters = {
                                                'start_date': filters.get('start_date'),
                                                'end_date': filters.get('end_date')
                                            }

                                    generation = self.gemini_client.generate_code(
                                        task,
                                        output_file=output_file,
                                        ticker_filters=ticker_filters,
                                        date_filters=date_filters
                                    )
                                else:
                                    print(f"{llm_id} - Retrying with error context from previous attempt")
                                    ticker_filters = None
                                    date_filters = None

                                    if filters:
                                        if filters.get('selected_tickers') or filters.get('ticker_count'):
                                            ticker_filters = {
                                                'selected_tickers': filters.get('selected_tickers', []),
                                                'ticker_count': filters.get('ticker_count')
                                            }

                                        if filters.get('start_date') or filters.get('end_date'):
                                            date_filters = {
                                                'start_date': filters.get('start_date'),
                                                'end_date': filters.get('end_date')
                                            }

                                    generation = self.gemini_client.generate_code(
                                        task,
                                        error_context=previous_error,
                                        failed_code=previous_code,
                                        output_file=output_file,
                                        ticker_filters=ticker_filters,
                                        date_filters=date_filters
                                    )

                            # Install requirements if needed
                            if generation.requirements:
                                if progress_callback:
                                    progress_callback(f"data: {json.dumps({'type': 'llm_installing', 'message': f'{llm_id.upper()} installing requirements...', 'llm': llm_id})}\n\n")
                                if not self.code_executor.install_requirements(generation.requirements):
                                    print(f"{llm_id} - Failed to install requirements on attempt {total_attempt}")
                                    if progress_callback:
                                        progress_callback(f"data: {json.dumps({'type': 'llm_install_failed', 'message': f'{llm_id.upper()} failed to install requirements', 'llm': llm_id})}\n\n")
                                    continue

                            # Execute code
                            if progress_callback:
                                progress_callback(f"data: {json.dumps({'type': 'llm_executing', 'message': f'{llm_id.upper()} executing code...', 'llm': llm_id})}\n\n")

                            filename = f"{filename_prefix}_{llm_id}.py"
                            test_result = self.code_executor.execute_code(generation.code, filename)

                            if test_result.success:
                                if progress_callback:
                                    progress_callback(f"data: {json.dumps({'type': 'llm_success', 'message': f'{llm_id.upper()} succeeded on attempt {total_attempt}!', 'llm': llm_id, 'attempt': total_attempt})}\n\n")
                                print(f"{llm_id} - Success on total attempt {total_attempt}!")
                                results[llm_id] = {
                                    "generation": generation,
                                    "test_result": test_result,
                                    "attempts": total_attempt,
                                    "complete_restarts": complete_restart,
                                    "error_attempts": error_attempt
                                }
                                return
                            else:
                                if progress_callback:
                                    progress_callback(f"data: {json.dumps({'type': 'llm_failed', 'message': f'{llm_id.upper()} failed on attempt {total_attempt}', 'llm': llm_id, 'error': test_result.error[:100]})}\n\n")
                                print(f"{llm_id} - Failed on attempt {total_attempt}: {test_result.error}")
                                previous_error = test_result.error
                                previous_code = generation.code

                        except Exception as e:
                            total_attempt = complete_restart * (max_error_attempts + 1) + error_attempt
                            if progress_callback:
                                progress_callback(f"data: {json.dumps({'type': 'llm_exception', 'message': f'{llm_id.upper()} exception on attempt {total_attempt}', 'llm': llm_id, 'error': str(e)[:100]})}\n\n")
                            print(f"{llm_id} - Exception on attempt {total_attempt}: {str(e)}")
                            previous_error = str(e)
                            if 'generation' in locals():
                                previous_code = generation.code

                    # All error attempts for this complete restart failed
                    if complete_restart < max_complete_restarts:
                        if progress_callback:
                            progress_callback(f"data: {json.dumps({'type': 'llm_restart_needed', 'message': f'{llm_id.upper()} restart {complete_restart + 1} failed, starting fresh...', 'llm': llm_id})}\n\n")
                        print(f"{llm_id} - Complete restart {complete_restart + 1} failed, starting fresh...")
                    else:
                        if progress_callback:
                            progress_callback(f"data: {json.dumps({'type': 'llm_all_restarts_failed', 'message': f'{llm_id.upper()} all restarts failed', 'llm': llm_id})}\n\n")
                        print(f"{llm_id} - All complete restarts failed")

                if progress_callback:
                    progress_callback(f"data: {json.dumps({'type': 'llm_finished', 'message': f'{llm_id.upper()} finished (all attempts exhausted)', 'llm': llm_id, 'success': False})}\n\n")
                print(f"{llm_id} - All attempts failed")

            # Run both LLMs in parallel using threads
            import threading

            print("Running LLM1 and LLM2 in parallel...")
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'llm_parallel_start', 'message': 'Starting LLM1 and LLM2 in parallel...', 'identity_attempt': identity_attempt + 1})}\n\n")

            thread1 = threading.Thread(target=lambda: process_llm("llm1", output_file1))
            thread2 = threading.Thread(target=lambda: process_llm("llm2", output_file2))

            # Start thread1
            thread1.start()
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'llm_started', 'message': 'LLM1 started', 'llm': 'llm1'})}\n\n")

            # Stagger start by 0.5s to reduce API rate limiting impact
            import time
            time.sleep(0.5)

            # Start thread2
            thread2.start()
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'llm_started', 'message': 'LLM2 started (staggered)', 'llm': 'llm2'})}\n\n")

            thread1.join()
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'llm_completed', 'message': 'LLM1 completed', 'llm': 'llm1'})}\n\n")

            thread2.join()
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'llm_completed', 'message': 'LLM2 completed', 'llm': 'llm2'})}\n\n")

            print("Both LLMs completed")

            # Check if both succeeded
            successful_results = [r for r in results.values() if r is not None]

            if len(successful_results) == 2:
                # Both succeeded, check if CSVs are identical
                print("Both LLMs succeeded. Checking if CSV outputs are identical...")
                if progress_callback:
                    progress_callback(f"data: {json.dumps({'type': 'csv_comparison_start', 'message': 'Both LLMs succeeded. Comparing CSV outputs...'})}\n\n")

                csv1_path = output_file1
                csv2_path = output_file2

                try:
                    import pandas as pd
                    df1 = pd.read_csv(csv1_path)
                    df2 = pd.read_csv(csv2_path)

                    # Check if CSVs are identical
                    are_identical = df1.equals(df2)

                    if are_identical:
                        if progress_callback:
                            progress_callback(f"data: {json.dumps({'type': 'csv_identical', 'message': f'CSVs are identical! (attempt {identity_attempt + 1})'})}\n\n")
                        print(f"SUCCESS! CSVs are identical on identity attempt {identity_attempt + 1}")
                        # Use the first result and mark as identical
                        final_generation = results["llm1"]["generation"]
                        final_test_result = results["llm1"]["test_result"]

                        self.copy_to_output_csv(csv1_path, final_output)

                        analytics = Analytics.analyze_results([final_generation], [final_test_result], data_stats)
                        analytics["generation_info"] = {
                            "retry_attempts": results["llm1"]["attempts"],
                            "max_retries": (max_complete_restarts + 1) * (max_error_attempts + 1),
                            "complete_restarts": results["llm1"].get("complete_restarts", 0),
                            "max_complete_restarts": max_complete_restarts,
                            "error_attempts": results["llm1"].get("error_attempts", 1),
                            "max_error_attempts": max_error_attempts,
                            "parallel_mode": True,
                            "successful_llms": 2,
                            "total_llms": 2,
                            "identity_attempts": identity_attempt + 1,
                            "max_identity_attempts": max_identity_attempts,
                            "csvs_identical": True,
                            "selected_llm": "llm1",
                            "task_type": task_type
                        }

                        return {
                            "generation": final_generation,
                            "test_result": final_test_result,
                            "analytics": analytics,
                            "parallel_results": results,
                            "csvs_identical": True,
                            "identity_attempt": identity_attempt + 1,
                            "task_type": task_type
                        }
                    else:
                        if progress_callback:
                            progress_callback(f"data: {json.dumps({'type': 'csv_not_identical', 'message': f'CSVs are NOT identical (attempt {identity_attempt + 1})'})}\n\n")

                        print(f"CSVs are NOT identical on identity attempt {identity_attempt + 1}")
                        if identity_attempt + 1 < max_identity_attempts:
                            if progress_callback:
                                progress_callback(f"data: {json.dumps({'type': 'retry_identity', 'message': f'Restarting for identity attempt {identity_attempt + 2}...'})}\n\n")
                            print(f"Restarting both LLMs for identity attempt {identity_attempt + 2}...")
                            continue
                        else:
                            if progress_callback:
                                progress_callback(f"data: {json.dumps({'type': 'identity_failed', 'message': 'Max identity attempts reached. Using comparison logic.'})}\n\n")
                            print("Maximum identity attempts reached. Falling back to comparison logic.")
                            break

                except Exception as e:
                    print(f"Error comparing CSVs: {str(e)}. Falling back to comparison logic.")
                    break
            else:
                # Not both succeeded, exit identity attempts and handle normally
                print(f"Only {len(successful_results)} LLM(s) succeeded. Exiting identity attempts.")
                break

        # Fallback logic - handle cases where not both succeeded or CSVs differ
        print("\nFalling back to existing CSV comparison logic...")
        successful_results = [r for r in results.values() if r is not None]

        if len(successful_results) == 0:
            return {"error": "Both LLM calls failed", "task_type": task_type}
        elif len(successful_results) == 1:
            # One succeeded, use it
            successful_result = successful_results[0]

            # Determine which LLM succeeded and copy its CSV to output.csv
            if results["llm1"] is not None:
                selected_file = output_file1
                selected_llm = "llm1"
            else:
                selected_file = output_file2
                selected_llm = "llm2"

            self.copy_to_output_csv(selected_file, final_output)

            analytics = Analytics.analyze_results([successful_result["generation"]], [successful_result["test_result"]], data_stats)
            analytics["generation_info"] = {
                "retry_attempts": successful_result["attempts"],
                "max_retries": (max_complete_restarts + 1) * (max_error_attempts + 1),
                "complete_restarts": successful_result.get("complete_restarts", 0),
                "max_complete_restarts": max_complete_restarts,
                "error_attempts": successful_result.get("error_attempts", 1),
                "max_error_attempts": max_error_attempts,
                "parallel_mode": True,
                "successful_llms": 1,
                "total_llms": 2,
                "identity_attempts": max_identity_attempts,
                "max_identity_attempts": max_identity_attempts,
                "csvs_identical": False,
                "selected_llm": selected_llm,
                "task_type": task_type
            }

            return {
                "generation": successful_result["generation"],
                "test_result": successful_result["test_result"],
                "analytics": analytics,
                "parallel_results": results,
                "task_type": task_type
            }
        else:
            # Both succeeded but CSVs differ - use first one as fallback
            print("Both LLMs succeeded but CSVs differ. Using LLM1 result.")

            final_generation = results["llm1"]["generation"]
            final_test_result = results["llm1"]["test_result"]

            self.copy_to_output_csv(output_file1, final_output)

            analytics = Analytics.analyze_results([final_generation], [final_test_result], data_stats)
            analytics["generation_info"] = {
                "retry_attempts": max(r["attempts"] for r in successful_results),
                "max_retries": (max_complete_restarts + 1) * (max_error_attempts + 1),
                "complete_restarts": max(r.get("complete_restarts", 0) for r in successful_results),
                "max_complete_restarts": max_complete_restarts,
                "error_attempts": max(r.get("error_attempts", 1) for r in successful_results),
                "max_error_attempts": max_error_attempts,
                "parallel_mode": True,
                "successful_llms": 2,
                "total_llms": 2,
                "identity_attempts": max_identity_attempts,
                "max_identity_attempts": max_identity_attempts,
                "csvs_identical": False,
                "selected_llm": "llm1",
                "task_type": task_type
            }

            return {
                "generation": final_generation,
                "test_result": final_test_result,
                "analytics": analytics,
                "parallel_results": results,
                "both_succeeded": True,
                "task_type": task_type
            }

    def copy_to_output_csv(self, source_file: str, destination_file: str = "output.csv"):
        """Copy the selected CSV to the specified output file in BOTH backend and project root"""
        import shutil
        try:
            if os.path.exists(source_file):
                # Hardcoded paths - copy to BOTH locations
                project_root_path = f"../{destination_file}"
                backend_path = f"{destination_file}"

                # Copy to project root (one level up)
                shutil.copy2(source_file, project_root_path)
                print(f"Copied {source_file} to {project_root_path}")

                # Copy to backend folder (current directory)
                shutil.copy2(source_file, backend_path)
                print(f"Copied {source_file} to {backend_path}")
            else:
                print(f"Warning: {source_file} not found, cannot copy")
        except Exception as e:
            print(f"Error copying {source_file}: {str(e)}")

    def compare_csv_files(self) -> Dict[str, Any]:
        """Compare the two generated CSV files"""
        import pandas as pd

        csv1_path = "output_llm1.csv"
        csv2_path = "output_llm2.csv"

        if not os.path.exists(csv1_path) or not os.path.exists(csv2_path):
            return {"error": "One or both CSV files not found"}

        try:
            df1 = pd.read_csv(csv1_path)
            df2 = pd.read_csv(csv2_path)

            comparison_info = {
                "file1": csv1_path,
                "file2": csv2_path,
                "file1_shape": df1.shape,
                "file2_shape": df2.shape,
                "file1_columns": df1.columns.tolist(),
                "file2_columns": df2.columns.tolist(),
                "columns_match": df1.columns.tolist() == df2.columns.tolist(),
                "shapes_match": df1.shape == df2.shape
            }

            return {
                "success": True,
                "comparison_info": comparison_info
            }

        except Exception as e:
            return {"error": f"CSV comparison failed: {str(e)}"}

    def _single_llm_process(self, task: str, task_type: str = "prompt", max_complete_restarts: int = 1,
                           max_error_attempts: int = 2, filters=None, progress_callback=None) -> Dict[str, Any]:
        """Process with single LLM (faster for testing)"""
        print(f"🔄 Single LLM mode for {task_type}: {task}")

        if progress_callback:
            progress_callback(f"data: {json.dumps({'type': 'single_mode', 'message': 'Running single LLM mode...'})}\n\n")

        # Configure output files
        if task_type == "condition":
            output_file = "condition_output.csv"
            filename_prefix = "condition_generated_code"
        else:
            output_file = "output.csv"
            filename_prefix = "generated_code"

        # Load data for analytics (skip for condition)
        data_stats = {}
        if task_type == "prompt":
            data = self.db_manager.get_all_data()
            data_stats = {
                "total_records": len(data),
                "unique_tickers": len(set(d["Ticker"] for d in data)),
                "date_range": {"start": min(d["Date"] for d in data), "end": max(d["Date"] for d in data)}
            }

        # Process single LLM with retries
        for complete_restart in range(max_complete_restarts + 1):
            previous_error = None
            previous_code = None

            for error_attempt in range(1, max_error_attempts + 2):
                try:
                    total_attempt = complete_restart * (max_error_attempts + 1) + error_attempt
                    print(f"Attempt {total_attempt}/{(max_complete_restarts + 1) * (max_error_attempts + 1)}")

                    # Generate code
                    if task_type == "condition":
                        from ..utils.condition_code_generator import generate_condition_code
                        if error_attempt == 1:
                            generation = generate_condition_code(task, self, output_file)
                        else:
                            retry_prompt = f"{task}\n\nPREVIOUS ERROR: {previous_error}\n\nFix and retry."
                            generation = generate_condition_code(retry_prompt, self, output_file)
                    else:
                        ticker_filters = None
                        date_filters = None
                        if filters:
                            if filters.get('selected_tickers') or filters.get('ticker_count'):
                                ticker_filters = {'selected_tickers': filters.get('selected_tickers', []), 'ticker_count': filters.get('ticker_count')}
                            if filters.get('start_date') or filters.get('end_date'):
                                date_filters = {'start_date': filters.get('start_date'), 'end_date': filters.get('end_date')}

                        if error_attempt == 1:
                            generation = self.gemini_client.generate_code(task, output_file=output_file, ticker_filters=ticker_filters, date_filters=date_filters)
                        else:
                            generation = self.gemini_client.generate_code(task, error_context=previous_error, failed_code=previous_code, output_file=output_file, ticker_filters=ticker_filters, date_filters=date_filters)

                    # Install requirements
                    if generation.requirements and not self.code_executor.install_requirements(generation.requirements):
                        continue

                    # Execute code
                    test_result = self.code_executor.execute_code(generation.code, f"{filename_prefix}.py")

                    if test_result.success:
                        print(f"✅ Success on attempt {total_attempt}!")
                        analytics = Analytics.analyze_results([generation], [test_result], data_stats)
                        analytics["generation_info"] = {
                            "retry_attempts": total_attempt,
                            "max_retries": (max_complete_restarts + 1) * (max_error_attempts + 1),
                            "parallel_mode": False,
                            "task_type": task_type
                        }
                        return {"generation": generation, "test_result": test_result, "analytics": analytics, "task_type": task_type}
                    else:
                        previous_error = test_result.error
                        previous_code = generation.code

                except Exception as e:
                    print(f"❌ Exception: {str(e)}")
                    previous_error = str(e)

        return {"error": "All attempts failed in single LLM mode", "task_type": task_type}

    def run_interactive_mode(self):
        """Run interactive mode for testing tasks"""
        print("=== Prompt-to-Code Testing System ===")
        print("Enter tasks to generate and test code against stock market data")
        print("Type 'quit' to exit")

        while True:
            task = input("\nEnter your task: ").strip()
            if task.lower() in ['quit', 'exit', 'q']:
                break

            if not task:
                continue

            result = self.process_task(task)

            if "error" in result:
                print(f"Error: {result['error']}")
                continue

            print("\n" + "="*50)
            print("RESULTS:")
            print("="*50)

            analytics = result["analytics"]

            print(f"Success Rate: {analytics['summary']['success_rate']:.1%}")
            print(f"Execution Time: {analytics['performance']['avg_execution_time']:.4f}s")

            if result["test_result"].success:
                print(f"Result: {result['test_result'].result}")
            else:
                print(f"Error: {result['test_result'].error}")
