"""
Auto-generate condition prompts based on output CSV columns
"""


def build_auto_condition_prompt(available_columns: list) -> str:
    """Generate a condition prompt based on available columns from output.csv

    Args:
        available_columns: List of column names from the generated output.csv
            Should include base columns (Date, Ticker, Adj_Close, etc.) plus
            any calculated columns (MA_10_Day, RSI, etc.)

    Returns:
        A condition prompt that Gemini will use to generate buy signals
    """

    # Filter out base columns to focus on calculated indicators
    base_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']
    indicator_columns = [col for col in available_columns if col not in base_columns]

    if not indicator_columns:
        # Fallback if no indicators were calculated
        return "Create a Signal column where Signal=1 when Daily_Gain_Pct > 0, otherwise Signal=0"

    # Build a smart condition based on available indicators
    indicators_text = ", ".join(indicator_columns)

    prompt = f"""You are an expert in generating trading signals. Based on the following calculated indicators, create a SMART buy signal condition.

AVAILABLE COLUMNS IN THE DATA:
Base columns: {', '.join(base_columns)}
Calculated indicators: {indicators_text}

YOUR TASK:
Generate a condition that determines when to BUY stocks (Signal=1) or stay out (Signal=0).

GUIDELINES:
1. Use the calculated indicators ({indicators_text}) to create meaningful signals
2. If there are moving averages → check for crossovers or price above/below MA
3. If there is RSI → look for oversold (< 30) or overbought (> 70) conditions
4. If there is momentum/volatility → use thresholds
5. Can combine multiple conditions with AND/OR logic
6. Condition should be conservative (don't signal too often)

GOAL: Generate signals that lead to profitable trades with good risk-adjusted returns

OUTPUT FORMAT:
Describe the condition in plain English (1-2 sentences). Example:
"Signal when the 10-day MA is above the 50-day MA AND RSI is below 70 (not overbought)"

GENERATE A SMART BUY SIGNAL CONDITION:"""

    return prompt
