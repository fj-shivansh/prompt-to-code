"""
Routes for database operations
"""
from flask import request, jsonify, send_file
from . import database_bp
import os
import pandas as pd
import numpy as np


def init_database_routes(db_manager):
    """Initialize database routes with dependencies"""

    @database_bp.route('/database_data')
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
            import sqlite3
            from ..config.settings import Settings

            conn = sqlite3.connect(Settings.DB_PATH)
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

    @database_bp.route('/tickers')
    def get_unique_tickers():
        try:
            tickers = db_manager.get_unique_tickers()
            return jsonify({'tickers': tickers})
        except Exception as e:
            return jsonify({'error': f'Database error: {str(e)}'}), 500

    @database_bp.route('/csv_data')
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
        sort_by = request.args.get('sort_by', 'Date')
        sort_order = request.args.get('sort_order', 'DESC')

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

    @database_bp.route('/download_csv')
    def download_csv():
        csv_path = os.path.abspath('../output.csv')

        if not os.path.exists(csv_path):
            return jsonify({'error': 'No CSV output file found. Run a prompt first.'}), 404

        return send_file(csv_path, as_attachment=True, download_name='output.csv')

    @database_bp.route('/date_range')
    def get_available_date_range():
        """Get min/max dates available in database"""
        try:
            date_range = db_manager.get_date_range()
            return jsonify(date_range)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @database_bp.route('/database_stats')
    def get_database_stats():
        """Get database statistics for 500-ticker dataset"""
        try:
            stats = db_manager.get_database_stats()
            return jsonify(stats)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @database_bp.route('/random_tickers', methods=['POST'])
    def get_random_tickers():
        """Optimized for 500-ticker database"""
        data = request.get_json()
        ticker_count = data.get('ticker_count', '10')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        try:
            import sqlite3
            from ..config.settings import Settings

            conn = sqlite3.connect(Settings.DB_PATH)
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

    @database_bp.route('/compare_csv', methods=['POST'])
    def compare_csv():
        """Compare the two generated CSV files and get similarity analysis"""
        from ..services.system_orchestrator import PromptToCodeSystem

        try:
            system = PromptToCodeSystem()
        except:
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
