"""
Routes for condition processing
"""
from flask import request, jsonify, send_file
from . import condition_bp
import os
import pandas as pd
import numpy as np


def init_condition_routes(system, db_manager):
    """Initialize condition routes with dependencies"""

    @condition_bp.route('/process_condition', methods=['POST'])
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
                count = int(ticker_count)
                selected_tickers = db_manager.get_random_tickers(
                    count=count,
                    start_date=start_date,
                    end_date=end_date
                )
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

    @condition_bp.route('/condition_csv_data')
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

    @condition_bp.route('/download_condition_csv')
    def download_condition_csv():
        """Download condition output CSV file"""
        csv_path = os.path.abspath('../condition_output.csv')

        if not os.path.exists(csv_path):
            return jsonify({'error': 'No condition output file found.'}), 404

        return send_file(csv_path, as_attachment=True, download_name='condition_output.csv')
