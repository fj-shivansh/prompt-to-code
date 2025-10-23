"""
Strategy Optimizer - Automated trading strategy generation and testing
"""
import os
import time
import pandas as pd
import uuid
from typing import Dict, Optional, List
from ..database.strategy_db import StrategyDatabase
from ..utils.strategy_generation_prompt import build_combined_strategy_prompt, build_improved_combined_prompt
from ..utils.condition_generation_prompt import build_auto_condition_prompt
from ..services.gemini_client import GeminiClient
from ..services.code_executor import CodeExecutor
from ..services.nav_calculator import NAVCalculator


class StrategyOptimizer:
    """Orchestrates automated strategy optimization loop"""

    def __init__(self, gemini_client: GeminiClient, code_executor: CodeExecutor,
                 prompt_refiner=None, system=None):
        self.gemini_client = gemini_client
        self.code_executor = code_executor
        self.prompt_refiner = prompt_refiner
        self.system = system
        self.nav_calculator = NAVCalculator()
        self.strategy_db = StrategyDatabase()
        self.should_stop = {}  # Dictionary to track stop signals per run_id

    def stop_optimization(self, run_id: str):
        """Signal to stop the optimization run"""
        self.should_stop[run_id] = True

    def run_optimization(self, iterations: int, ticker_filters: Optional[Dict] = None,
                        date_filters: Optional[Dict] = None, nav_settings: Optional[Dict] = None,
                        run_id: Optional[str] = None) -> Dict:
        """
        Run the full optimization loop

        Args:
            iterations: Number of strategy iterations to run
            ticker_filters: Ticker selection filters
            date_filters: Date range filters
            nav_settings: NAV calculation settings
            run_id: Optional run ID (will generate if not provided)

        Returns:
            Dictionary with run_id and summary stats
        """
        if run_id is None:
            run_id = str(uuid.uuid4())
        self.should_stop[run_id] = False

        # Default NAV settings
        if nav_settings is None:
            nav_settings = {
                'initial_amount': 100000,
                'amount_to_invest': 0.7,
                'max_position_each_ticker': 0.2,
                'trader_cost': 0
            }

        print(f"\n{'='*80}")
        print(f"Starting Strategy Optimization Run: {run_id}")
        print(f"Iterations: {iterations}")
        print(f"Ticker filters: {ticker_filters}")
        print(f"Date filters: {date_filters}")
        print(f"{'='*80}\n")

        results = []

        for iteration in range(1, iterations + 1):
            # Check if user requested stop
            if self.should_stop.get(run_id, False):
                print(f"Optimization stopped by user at iteration {iteration}")
                break

            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}/{iterations}")
            print(f"{'='*60}")

            try:
                result = self._run_single_iteration(
                    run_id=run_id,
                    iteration=iteration,
                    ticker_filters=ticker_filters,
                    date_filters=date_filters,
                    nav_settings=nav_settings
                )
                results.append(result)

                # Log progress
                if result['status'] == 'success':
                    metrics = result.get('nav_metrics', {})
                    print(f"\n✅ Iteration {iteration} SUCCESS")
                    print(f"   Ratio: {metrics.get('ratio', 'N/A')}")
                    print(f"   Annual Return: {metrics.get('annual_return', 'N/A')}%")
                    print(f"   Max Drawdown: {metrics.get('max_drawdown', 'N/A')}%")
                else:
                    print(f"\n❌ Iteration {iteration} FAILED: {result.get('error_message', 'Unknown error')}")

            except Exception as e:
                print(f"\n❌ Iteration {iteration} CRITICAL ERROR: {str(e)}")
                self.strategy_db.save_strategy_run(
                    run_id=run_id,
                    iteration=iteration,
                    main_prompt="",
                    condition_prompt="",
                    nav_metrics=None,
                    status="error",
                    error_message=str(e),
                    ticker_filters=ticker_filters,
                    date_filters=date_filters
                )

        # Cleanup stop signal
        if run_id in self.should_stop:
            del self.should_stop[run_id]

        # Get final results summary
        all_results = self.strategy_db.get_run_results(run_id)
        successful = [r for r in all_results if r['status'] == 'success']

        best_strategy = None
        if successful:
            best_strategy = max(successful, key=lambda x: x.get('nav_metrics', {}).get('ratio', 0) if x.get('nav_metrics') else 0)

        print(f"\n{'='*80}")
        print(f"Optimization Run Complete: {run_id}")
        print(f"Total iterations: {len(all_results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(all_results) - len(successful)}")
        if best_strategy and best_strategy.get('nav_metrics'):
            print(f"Best Ratio: {best_strategy['nav_metrics'].get('ratio', 'N/A')}")
        print(f"{'='*80}\n")

        return {
            'run_id': run_id,
            'total_iterations': len(all_results),
            'successful_iterations': len(successful),
            'best_strategy': best_strategy
        }

    def _run_single_iteration(self, run_id: str, iteration: int,
                             ticker_filters: Optional[Dict],
                             date_filters: Optional[Dict],
                             nav_settings: Dict) -> Dict:
        """Run a single optimization iteration with retry logic"""

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Step 1: Generate main strategy prompt
                print(f"\n[{iteration}] Step 1: Generating strategy prompt (attempt {attempt + 1}/{max_retries})...")
                prompts = self._generate_main_prompt(run_id, iteration)
                print(prompts)
                main_prompt = prompts["main_prompt"]
                print(f"Generated prompt: {main_prompt[:200]}...")

                # Step 2: Generate and execute main code
                print(f"[{iteration}] Step 2: Generating and executing main code...")
                main_result = self._execute_main_pipeline(main_prompt, ticker_filters, date_filters, iteration)

                if not main_result['success']:
                    if attempt < max_retries - 1:
                        print(f"Main pipeline failed, retrying with error context...")
                        continue
                    else:
                        raise Exception(f"Main pipeline failed: {main_result['error']}")

                # Step 3: Generate condition prompt
                print(f"[{iteration}] Step 3: Generating condition prompt...")
                condition_prompt = prompts["condition_prompt"]
                print(f"Condition prompt: {condition_prompt[:150]}...")

                # Step 4: Generate and execute condition code
                print(f"[{iteration}] Step 4: Generating and executing condition code...")
                condition_result = self._execute_condition_pipeline(condition_prompt, iteration)

                if not condition_result['success']:
                    if attempt < max_retries - 1:
                        print(f"Condition pipeline failed, retrying...")
                        continue
                    else:
                        raise Exception(f"Condition pipeline failed: {condition_result['error']}")

                # Step 5: Calculate NAV
                print(f"[{iteration}] Step 5: Calculating NAV...")
                nav_metrics = self._calculate_nav(iteration, nav_settings)

                # Step 6: Save to database
                self.strategy_db.save_strategy_run(
                    run_id=run_id,
                    iteration=iteration,
                    main_prompt=main_prompt,
                    condition_prompt=condition_prompt,
                    nav_metrics=nav_metrics,
                    status="success",
                    ticker_filters=ticker_filters,
                    date_filters=date_filters
                )

                return {
                    'iteration': iteration,
                    'status': 'success',
                    'main_prompt': main_prompt,
                    'condition_prompt': condition_prompt,
                    'nav_metrics': nav_metrics
                }

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Attempt {attempt + 1} failed: {error_msg}")

                if attempt == max_retries - 1:
                    # Final attempt failed, save as failed
                    self.strategy_db.save_strategy_run(
                        run_id=run_id,
                        iteration=iteration,
                        main_prompt=main_prompt if 'main_prompt' in locals() else "",
                        condition_prompt="",
                        nav_metrics=None,
                        status="failed",
                        error_message=error_msg,
                        ticker_filters=ticker_filters,
                        date_filters=date_filters
                    )

                    return {
                        'iteration': iteration,
                        'status': 'failed',
                        'error_message': error_msg
                    }

        # Should not reach here
        return {'iteration': iteration, 'status': 'failed', 'error_message': 'Unknown error'}


    def _generate_main_prompt(self, run_id: str, iteration: int) -> dict:
        """Generate strategy prompt using Gemini and return structured prompts"""
        import json
        import re
        if iteration == 1:
            meta_prompt = build_combined_strategy_prompt()
        else:
            history = self.strategy_db.get_run_history_for_prompt(run_id, max_items=5)
            meta_prompt = build_improved_combined_prompt(history)

        # Call Gemini (old SDK, cannot use config)
        response = self.gemini_client.model.generate_content(meta_prompt)
        text = response.text.strip()

        # Remove "json" prefix if present
        if text.lower().startswith("json"):
            text = text[len("json"):].strip()

        # Remove markdown code blocks ``` or ```json
        text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()

        # Try parsing JSON
        try:
            prompts = json.loads(text)
        except json.JSONDecodeError:
            # Optional: auto-fix common issues (e.g., single quotes)
            text_fixed = text.replace("'", '"')
            prompts = json.loads(text_fixed)

        # Ensure keys exist
        main_prompt = prompts.get("main_prompt", "")
        condition_prompt = prompts.get("condition_prompt", "")
        return {
            "main_prompt": main_prompt,
            "condition_prompt": condition_prompt
        }

    def _generate_condition_prompt(self, iteration: int) -> str:
        """Generate condition prompt based on output.csv columns"""
        # Use absolute path relative to project root
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_csv = os.path.join(project_root, f"output_opt_iter{iteration}.csv")

        if not os.path.exists(output_csv):
            output_csv = os.path.join(project_root, "output.csv")  # Fallback

        try:
            df = pd.read_csv(output_csv)
            available_columns = df.columns.tolist()
        except Exception as e:
            print(f"Warning: Could not read output CSV: {e}")
            available_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']

        meta_prompt = build_auto_condition_prompt(available_columns)

        # Call Gemini to generate condition
        response = self.gemini_client.model.generate_content(meta_prompt)
        condition_prompt = response.text.strip()

        # Clean up formatting
        if condition_prompt.startswith('```'):
            lines = condition_prompt.split('\n')
            condition_prompt = '\n'.join(lines[1:-1])

        return condition_prompt

    def _execute_main_pipeline(self, prompt: str, ticker_filters: Optional[Dict],
                               date_filters: Optional[Dict], iteration: int) -> Dict:
        """Execute main code generation and execution pipeline using normal flow"""
        import os
        import shutil

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_file = os.path.join(project_root, f"output_opt_iter{iteration}.csv")

        try:
            # Step 2: Build filters dictionary for parallel processing
            filters = {}
            if ticker_filters:
                filters.update(ticker_filters)
            if date_filters:
                filters.update(date_filters)

            # Step 3: Use SystemOrchestrator's parallel processing (2 LLMs)
            if self.system:
                print(f"   → Running parallel processing with 2 LLMs...")
                result = self.system.parallel_process_generic(
                    prompt,
                    task_type="prompt",
                    max_complete_restarts=1,
                    max_error_attempts=2,
                    filters=filters,
                    progress_callback=None  # No streaming for optimization
                )

                if "error" in result:
                    return {'success': False, 'error': result['error']}

                if not result.get('test_result') or not result['test_result'].success:
                    error_msg = result.get('test_result', {}).error if result.get('test_result') else 'Unknown error'
                    return {'success': False, 'error': error_msg}

                # Step 4: Copy output.csv to iteration-specific filename
                standard_output = os.path.join(project_root, "output.csv")
                if os.path.exists(standard_output):
                    shutil.copy2(standard_output, output_file)
                    print(f"   → Copied output.csv to output_opt_iter{iteration}.csv")

                return {'success': True, 'output_file': output_file}
            else:
                # Fallback to old method if system not available
                print(f"   ⚠️  SystemOrchestrator not available, using direct generation")
                generation = self.gemini_client.generate_code(
                    task=prompt,
                    output_file=f"output_opt_iter{iteration}.csv",
                    ticker_filters=ticker_filters,
                    date_filters=date_filters
                )

                if generation.requirements:
                    self.code_executor.install_requirements(generation.requirements)

                code_filename = os.path.join(project_root, f"generated_code_opt_iter{iteration}.py")
                test_result = self.code_executor.execute_code(generation.code, code_filename)

                if test_result.success:
                    return {'success': True, 'output_file': output_file}
                else:
                    return {'success': False, 'error': test_result.error}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_condition_pipeline(self, condition_prompt: str, iteration: int) -> Dict:
        """Execute condition code generation and execution using normal flow"""
        import os
        import shutil

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_file = os.path.join(project_root, f"condition_output_opt_iter{iteration}.csv")

        try:
            # Use SystemOrchestrator's condition parallel processing
            if self.system:
                print(f"   → Running condition processing with 2 LLMs...")
                result = self.system.process_condition_parallel(
                    condition_prompt,
                    max_complete_restarts=1,
                    max_error_attempts=2,
                    progress_callback=None  # No streaming for optimization
                )

                if "error" in result:
                    return {'success': False, 'error': result['error']}

                if not result.get('test_result') or not result['test_result'].success:
                    error_msg = result.get('test_result', {}).error if result.get('test_result') else 'Unknown error'
                    return {'success': False, 'error': error_msg}

                # Copy condition_output.csv to iteration-specific filename
                standard_output = os.path.join(project_root, "condition_output.csv")
                if os.path.exists(standard_output):
                    shutil.copy2(standard_output, output_file)
                    print(f"   → Copied condition_output.csv to condition_output_opt_iter{iteration}.csv")

                return {'success': True, 'output_file': output_file}
            else:
                # Fallback to old method if system not available
                print(f"   ⚠️  SystemOrchestrator not available, using direct generation")
                from ..utils.condition_code_generator import generate_condition_code

                class MockSystem:
                    def __init__(self, client):
                        self.gemini_client = client

                mock_system = MockSystem(self.gemini_client)

                generation = generate_condition_code(
                    condition_prompt,
                    mock_system,
                    f"condition_output_opt_iter{iteration}.csv"
                )

                if generation.requirements:
                    self.code_executor.install_requirements(generation.requirements)

                code_filename = os.path.join(project_root, f"condition_generated_code_opt_iter{iteration}.py")
                test_result = self.code_executor.execute_code(generation.code, code_filename)

                if test_result.success:
                    return {'success': True, 'output_file': output_file}
                else:
                    return {'success': False, 'error': test_result.error}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _calculate_nav(self, iteration: int, nav_settings: Dict) -> Dict:
        """Calculate NAV from condition output"""
        # Use absolute path relative to project root
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        condition_csv = os.path.join(project_root, f"condition_output_opt_iter{iteration}.csv")

        if not os.path.exists(condition_csv):
            raise Exception(f"Condition output file not found: {condition_csv}")

        df = pd.read_csv(condition_csv)

        # Validate required columns
        required_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct', 'Signal']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise Exception(f"Missing required columns in condition output: {missing_columns}")

        # Calculate NAV
        nav_df, annual_return, max_drawdown, ratio = self.nav_calculator.calculate_nav_long_only(
            df,
            initial_amount=nav_settings['initial_amount'],
            amount_to_invest=nav_settings['amount_to_invest'],
            max_position_each_ticker=nav_settings['max_position_each_ticker'],
            trader_cost=nav_settings['trader_cost']
        )

        if nav_df.empty:
            raise Exception("No signals found in condition output")

        # Calculate final NAV and return
        initial_nav = nav_df['NAV'].iloc[0]
        final_nav = nav_df['NAV'].iloc[-1]
        total_return = ((final_nav - initial_nav) / initial_nav) * 100

        return {
            'initial_amount': nav_settings['initial_amount'],
            'final_nav': float(final_nav),
            'total_return_pct': float(total_return),
            'annual_return': float(annual_return),
            'max_drawdown': float(max_drawdown),
            'ratio': float(ratio),
            'total_signals': len(nav_df),
            'trader_cost': nav_settings['trader_cost']
        }

    def get_run_status(self, run_id: str) -> Dict:
        """Get current status of an optimization run"""
        # Check if run is actively running (even if no DB results yet)
        is_running = run_id in self.should_stop

        results = self.strategy_db.get_run_results(run_id)

        # If no results yet but it's running, return running status
        if not results and is_running:
            return {
                'status': 'running',
                'total_iterations': 0,
                'successful_iterations': 0,
                'best_ratio': 0,
                'is_stopped': self.should_stop.get(run_id, False)
            }

        # If no results and not running, it's not found
        if not results:
            return {'status': 'not_found'}

        successful = [r for r in results if r['status'] == 'success']
        best_ratio = 0
        if successful:
            best = max(successful, key=lambda x: x.get('nav_metrics', {}).get('ratio', 0) if x.get('nav_metrics') else 0)
            if best.get('nav_metrics'):
                best_ratio = best['nav_metrics'].get('ratio', 0)

        return {
            'status': 'running' if is_running else 'completed',
            'total_iterations': len(results),
            'successful_iterations': len(successful),
            'best_ratio': best_ratio,
            'is_stopped': self.should_stop.get(run_id, False)
        }
