"""
NAV (Net Asset Value) calculation service
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import base64
from io import BytesIO

matplotlib.use('Agg')  # Use non-GUI backend


class NAVCalculator:
    """Calculates NAV for trading strategies"""

    @staticmethod
    def calculate_nav_long_only(
        df: pd.DataFrame,
        initial_amount: float = 100000,
        amount_to_invest: float = 1,
        max_position_each_ticker: float = 1,
        trader_cost: float = 0
    ):
        """Calculate NAV for long-only strategy with transaction costs

        Args:
            df: DataFrame with stock data and signals
            initial_amount: Starting portfolio value
            amount_to_invest: Fraction of capital to invest (0-1)
            max_position_each_ticker: Max position size per ticker (0-1)
            trader_cost: Flat fee per trade in dollars (e.g., $1, $5)
        """
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        nav_list = [initial_amount]
        date_list = []
        total_trades = 0
        total_cost = 0

        # Group by Date (automatically gives unique dates in ascending order)
        for date, daily_df in df.groupby('Date', sort=True):
            daily_df = daily_df[daily_df["Signal"] == 1]
            count = len(daily_df)
            if count > 0:
                total_forward_gain = daily_df['Forward_Gain_Pct'].sum()

                percentage_of_each_ticker = 1 / count
                percentage_of_each_ticker = min(percentage_of_each_ticker, max_position_each_ticker)
                avg_forward_gain = total_forward_gain * percentage_of_each_ticker

                # Convert percentage to decimal (e.g., 5.0 -> 0.05)
                avg_forward_gain = avg_forward_gain / 100

                # Calculate trading costs
                # For each signal: 2 trades (buy + sell), but we deduct cost only once per position
                # Since we're trading count tickers, we have count round-trip trades
                trades_today = count * 2  # Buy + Sell for each ticker
                cost_today = trades_today * trader_cost
                total_trades += trades_today
                total_cost += cost_today

                # Apply gain and subtract trading costs
                current_nav = nav_list[-1]
                gain_amount = current_nav * amount_to_invest * avg_forward_gain
                new_nav = current_nav + gain_amount - cost_today
            else:
                avg_forward_gain = 0
                new_nav = nav_list[-1]

            date_list.append(date)
            nav_list.append(new_nav)

        nav_list = nav_list[:-1]

        nav_df = pd.DataFrame({"Date": date_list, "NAV": nav_list})
        start_val = nav_df["NAV"].iloc[0]
        end_val = nav_df["NAV"].iloc[-1]

        annual_return = ((end_val / start_val) ** (250 / nav_df.shape[0]) - 1)
        annual_return = float(round(annual_return * 100, 2))

        rolling_max = nav_df["NAV"].cummax()
        drawdown = (nav_df["NAV"] - rolling_max) / rolling_max
        max_drawdown = float(round(-drawdown.min() * 100, 2))
        ratio = float(round((annual_return / max_drawdown) * 100, 2)) if max_drawdown != 0 else 0

        return nav_df, annual_return, max_drawdown, ratio

    @staticmethod
    def generate_nav_graph(nav_df: pd.DataFrame) -> str:
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
