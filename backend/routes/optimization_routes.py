"""
Routes for strategy optimization
"""
from flask import request, jsonify
from . import optimization_bp
import threading

# Global state for tracking running optimizations
running_optimizations = {}


def init_optimization_routes(strategy_optimizer):
    """Initialize optimization routes with dependencies"""

    @optimization_bp.route('/start_optimization', methods=['POST'])
    def start_optimization():
        """Start a new optimization run"""
        try:
            data = request.get_json()
            iterations = data.get('iterations', 10)
            ticker_filters = data.get('ticker_filters')
            date_filters = data.get('date_filters')
            nav_settings = data.get('nav_settings', {
                'initial_amount': 100000,
                'amount_to_invest': 0.7,
                'max_position_each_ticker': 0.2,
                'trader_cost': 0
            })

            # Debug logging
            print(f"🔍 Optimization Request:")
            print(f"   Iterations: {iterations}")
            print(f"   Ticker Filters: {ticker_filters}")
            print(f"   Date Filters: {date_filters}")
            print(f"   NAV Settings: {nav_settings}")

            # Validate iterations
            if iterations < 1 or iterations > 50:
                return jsonify({'error': 'Iterations must be between 1 and 50'}), 400

            # Handle ticker selection if ticker_count is specified without selected_tickers
            # IMPORTANT: Select tickers ONCE here so all iterations use the same tickers for fair comparison
            if ticker_filters and ticker_filters.get('ticker_count') and not ticker_filters.get('selected_tickers'):
                ticker_count = ticker_filters.get('ticker_count')
                if ticker_count != 'all':
                    try:
                        # Get random tickers from database (seed for reproducibility within this run)
                        from ..database.manager import DatabaseManager
                        from ..config.settings import Settings
                        import random

                        db_manager = DatabaseManager(Settings.DB_PATH)
                        all_tickers = db_manager.get_unique_tickers()

                        # Select random tickers - these will be used for ALL iterations
                        count = int(ticker_count) if isinstance(ticker_count, str) else ticker_count
                        selected = random.sample(all_tickers, min(count, len(all_tickers)))
                        ticker_filters['selected_tickers'] = selected
                        print(f"   📊 Selected {len(selected)} tickers for ALL iterations: {selected}")
                        print(f"   ⚠️  All {iterations} iterations will use the SAME tickers for fair comparison")
                    except Exception as e:
                        print(f"   ⚠️  Warning: Could not select tickers: {e}")

            # Convert camelCase nav_settings to snake_case
            if nav_settings:
                nav_settings = {
                    'initial_amount': nav_settings.get('initialAmount', 100000),
                    'amount_to_invest': nav_settings.get('amountToInvest', 0.7),
                    'max_position_each_ticker': nav_settings.get('maxPositionEachTicker', 0.2),
                    'trader_cost': nav_settings.get('traderCost', 0)
                }

            # Generate run_id upfront
            import uuid
            run_id = str(uuid.uuid4())

            # Store thread info
            running_optimizations[run_id] = {
                'thread': None,
                'completed': False,
                'result': None,
                'run_id': run_id
            }

            # Run optimization in background thread
            def run_in_background():
                result = strategy_optimizer.run_optimization(
                    iterations=iterations,
                    ticker_filters=ticker_filters,
                    date_filters=date_filters,
                    nav_settings=nav_settings,
                    run_id=run_id  # Pass the run_id
                )
                # Store result
                if run_id in running_optimizations:
                    running_optimizations[run_id]['completed'] = True
                    running_optimizations[run_id]['result'] = result

            # Start thread
            thread = threading.Thread(target=run_in_background, daemon=True)
            running_optimizations[run_id]['thread'] = thread
            thread.start()

            return jsonify({
                'success': True,
                'message': 'Optimization started',
                'run_id': run_id
            })

        except Exception as e:
            return jsonify({'error': f'Failed to start optimization: {str(e)}'}), 500

    @optimization_bp.route('/optimization_status/<run_id>', methods=['GET'])
    def get_optimization_status(run_id):
        """Get status of a running optimization"""
        try:
            status = strategy_optimizer.get_run_status(run_id)
            return jsonify(status)

        except Exception as e:
            return jsonify({'error': f'Failed to get status: {str(e)}'}), 500

    @optimization_bp.route('/stop_optimization/<run_id>', methods=['POST'])
    def stop_optimization(run_id):
        """Stop a running optimization"""
        try:
            strategy_optimizer.stop_optimization(run_id)
            return jsonify({'success': True, 'message': 'Stop signal sent'})

        except Exception as e:
            return jsonify({'error': f'Failed to stop optimization: {str(e)}'}), 500

    @optimization_bp.route('/optimization_results/<run_id>', methods=['GET'])
    def get_optimization_results(run_id):
        """Get all results for an optimization run"""
        try:
            results = strategy_optimizer.strategy_db.get_run_results(run_id)

            # Format results for frontend
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'iteration': result['iteration_number'],
                    'main_prompt': result['main_prompt'],
                    'condition_prompt': result['condition_prompt'],
                    'nav_metrics': result['nav_metrics'],
                    'status': result['status'],
                    'error_message': result['error_message'],
                    'timestamp': result['timestamp']
                })

            return jsonify({
                'success': True,
                'run_id': run_id,
                'results': formatted_results
            })

        except Exception as e:
            return jsonify({'error': f'Failed to get results: {str(e)}'}), 500

    @optimization_bp.route('/best_strategies', methods=['GET'])
    def get_best_strategies():
        """Get top performing strategies across all runs"""
        try:
            top_n = request.args.get('top_n', 10, type=int)
            strategies = strategy_optimizer.strategy_db.get_best_strategies(top_n)

            # Format for frontend
            formatted_strategies = []
            for strategy in strategies:
                formatted_strategies.append({
                    'run_id': strategy['run_id'],
                    'iteration': strategy['iteration_number'],
                    'main_prompt': strategy['main_prompt'],
                    'condition_prompt': strategy['condition_prompt'],
                    'nav_metrics': strategy['nav_metrics'],
                    'timestamp': strategy['timestamp']
                })

            return jsonify({
                'success': True,
                'strategies': formatted_strategies
            })

        except Exception as e:
            return jsonify({'error': f'Failed to get best strategies: {str(e)}'}), 500

    @optimization_bp.route('/all_runs', methods=['GET'])
    def get_all_runs():
        """Get summary of all optimization runs"""
        try:
            runs = strategy_optimizer.strategy_db.get_all_runs()
            return jsonify({
                'success': True,
                'runs': runs
            })

        except Exception as e:
            return jsonify({'error': f'Failed to get runs: {str(e)}'}), 500

    @optimization_bp.route('/delete_run/<run_id>', methods=['DELETE'])
    def delete_run(run_id):
        """Delete an optimization run"""
        try:
            strategy_optimizer.strategy_db.delete_run(run_id)
            return jsonify({'success': True, 'message': 'Run deleted'})

        except Exception as e:
            return jsonify({'error': f'Failed to delete run: {str(e)}'}), 500
