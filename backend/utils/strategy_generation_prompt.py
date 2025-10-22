"""
Strategy generation prompts for optimization
"""


def build_initial_strategy_prompt() -> str:
    """Build prompt for Gemini to generate the first trading strategy"""
    return """You are an expert quantitative trading strategy designer. Generate a novel, creative trading strategy.

DATABASE AVAILABLE:
- File: historical_data_500_tickers_with_gains.db
- Table: stock_data
- Columns: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct

YOUR TASK:
Create a trading strategy prompt that will be used to generate Python code for backtesting.

STRATEGY IDEAS (you can use ANY of these or create your own):
- Moving average crossovers (short MA crosses above/below long MA)
- Momentum strategies (buy stocks with highest recent gains)
- Mean reversion (buy stocks that dropped, sell when recovered)
- Volatility-based (trade based on price stability/instability)
- Multi-factor combinations (combine multiple indicators)
- Pattern recognition (detect specific price patterns)
- Statistical arbitrage (z-score based trading)

REQUIREMENTS:
1. Use technical indicators that can be calculated from: Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct
2. Strategy should be clear and implementable in Python
3. Focus on strategies that maximize risk-adjusted returns (high ratio = annual_return / max_drawdown)
4. Be creative but realistic

OUTPUT FORMAT:
Return ONLY the trading strategy prompt as plain text (2-4 sentences). Do NOT include any introductory phrases like "Here's a strategy:" or "Strategy Prompt:". Just return the raw prompt directly.

EXAMPLE OUTPUT:
Calculate 10-day and 50-day moving averages of Adj_Close for each ticker. Create columns MA_10_Day and MA_50_Day. Also calculate the RSI (14-day) and add a column RSI_14. Sort by Date descending.

NOW GENERATE A NEW, CREATIVE TRADING STRATEGY PROMPT (raw text only, no introduction):"""


def build_improvement_prompt(history: list) -> str:
    """Build prompt for Gemini to generate improved strategy based on history

    Args:
        history: List of dicts with keys: iteration, main_prompt, ratio, annual_return, max_drawdown
    """

    if not history:
        return build_initial_strategy_prompt()

    # Find best performing strategy
    best = max(history, key=lambda x: x.get('ratio', 0))

    # Build history summary
    history_text = "PREVIOUS ATTEMPTS AND RESULTS:\n\n"
    for item in history:
        history_text += f"""Iteration {item['iteration']}:
Strategy: "{item['main_prompt']}"
Results:
  - Ratio (risk-adjusted return): {item['ratio']}
  - Annual Return: {item['annual_return']}%
  - Max Drawdown: {item['max_drawdown']}%
  - Final NAV: ${item.get('final_nav', 'N/A')}

"""

    history_text += f"\nBEST STRATEGY SO FAR:\n"
    history_text += f'"{best["main_prompt"]}"\n'
    history_text += f"Ratio: {best['ratio']}, Return: {best['annual_return']}%, Drawdown: {best['max_drawdown']}%\n"

    prompt = f"""{history_text}

DATABASE AVAILABLE:
- File: historical_data_500_tickers_with_gains.db
- Table: stock_data
- Columns: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct

YOUR TASK:
Generate a NEW trading strategy that improves upon the previous attempts.

ANALYSIS GUIDANCE:
- If strategies with lower drawdown performed better → prioritize risk management
- If momentum strategies worked well → explore variations
- If mean reversion failed → try different approaches
- Look for patterns in what worked vs what didn't

GOAL: Maximize ratio (annual_return / max_drawdown) - this is the risk-adjusted return metric

REQUIREMENTS:
1. Must be DIFFERENT from previous strategies (don't just tweak numbers)
2. Can combine successful elements from multiple past strategies
3. Should be creative but implementable
4. Use technical indicators calculable from available columns

OUTPUT FORMAT:
Return ONLY the trading strategy prompt as plain text (2-4 sentences). Do NOT include any introductory phrases like "Here's an improved strategy:" or "New Strategy:". Just return the raw prompt directly.

GENERATE AN IMPROVED TRADING STRATEGY PROMPT (raw text only, no introduction):"""

    return prompt
