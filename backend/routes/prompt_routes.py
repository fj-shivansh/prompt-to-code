"""
Routes for prompt processing
"""
from flask import request, jsonify, Response
from . import prompt_bp
import json
import time


def init_prompt_routes(system, prompt_refiner, process_manager):
    """Initialize prompt routes with dependencies"""

    @prompt_bp.route('/process_prompt_stream', methods=['POST'])
    def process_prompt_stream():
        """Server-Sent Events endpoint for real-time status updates"""
        from ..database.manager import DatabaseManager
        from ..config.settings import Settings

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
                db_manager = DatabaseManager(Settings.DB_PATH)
                count = int(ticker_count)
                selected_tickers = db_manager.get_random_tickers(
                    count=count,
                    start_date=start_date,
                    end_date=end_date
                )
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
            from ..services.prompt_processing import generate_status_updates
            for update in generate_status_updates(
                original_prompt,
                filters,
                system,
                prompt_refiner,
                process_manager
            ):
                yield update

        return Response(event_stream(), mimetype='text/event-stream',
                       headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})

    @prompt_bp.route('/refine_prompt', methods=['POST'])
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

    @prompt_bp.route('/stop_processing', methods=['POST'])
    def stop_processing():
        """Endpoint to stop current processing"""
        result = process_manager.stop_processing()

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500

    @prompt_bp.route('/health')
    def health_check():
        return jsonify({'status': 'healthy', 'system_ready': system is not None})
