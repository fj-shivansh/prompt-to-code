"""
Routes for NAV calculation
"""
from flask import request, jsonify
from . import nav_bp
import os
import pandas as pd


def init_nav_routes(nav_calculator):
    """Initialize NAV routes with dependencies"""

    @nav_bp.route('/calculate_nav', methods=['POST'])
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
            trader_cost = data.get('trader_cost', 0)  # Default to $0 per trade

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
            nav_df, annual_return, max_drawdown, ratio = nav_calculator.calculate_nav_long_only(
                df, initial_amount, amount_to_invest, max_position_each_ticker, trader_cost
            )

            if nav_df.empty:
                return jsonify({
                    'error': 'No signals found in the data. Make sure your condition generates signals (Signal=1).'
                }), 400

            # Generate graph
            graph_base64 = nav_calculator.generate_nav_graph(nav_df)

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
                    'max_position_each_ticker': max_position_each_ticker,
                    'trader_cost': trader_cost
                }
            })

        except Exception as e:
            return jsonify({'error': f'NAV calculation failed: {str(e)}'}), 500
