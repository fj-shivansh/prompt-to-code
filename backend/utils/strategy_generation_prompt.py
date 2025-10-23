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


def build_improved_combined_prompt(history: list) -> str:
    """Build prompt for Gemini to generate an improved main and condition prompt based on history

    Args:
        history: List of dicts with keys: iteration, main_prompt, condition_prompt, ratio, annual_return, max_drawdown
    """

    if not history:
        return build_combined_strategy_prompt()

    # Find best performing strategy
    best = max(history, key=lambda x: x.get('ratio', 0))

    # Build history summary
    history_text = "PREVIOUS ATTEMPTS AND RESULTS:\n\n"
    for item in history:
        history_text += f"""Iteration {item['iteration']}:
Main Strategy: "{item['main_prompt']}"
Condition: "{item['condition_prompt']}"
Results:
  - Ratio (risk-adjusted return): {item['ratio']}
  - Annual Return: {item['annual_return']}%
  - Max Drawdown: {item['max_drawdown']}%
  - Final NAV: ${item.get('final_nav', 'N/A')}

"""

    history_text += f"\nBEST STRATEGY SO FAR:\n"
    history_text += f'Main Strategy: "{best["main_prompt"]}"\n'
    history_text += f'Condition: "{best["condition_prompt"]}"\n'
    history_text += f"Ratio: {best['ratio']}, Return: {best['annual_return']}%, Drawdown: {best['max_drawdown']}%\n"

    prompt = f"""{history_text}

DATABASE AVAILABLE:
- File: historical_data_500_tickers_with_gains.db
- Table: stock_data
- Columns: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct

YOUR TASK:
Generate a NEW trading strategy that improves upon the previous attempts.

REFINEMENT RULES (APPLY INTERNALLY):
- Transform the new strategy into a concise, step-by-step main_prompt with clearly defined calculations and output columns.
- The first 5 columns in CSV must always be:
  Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct.
- Any additional calculated columns come AFTER these five.
- Also generate a clear condition_prompt using both base and newly calculated columns.
- Column names must be consistent and fully defined.

ANALYSIS GUIDANCE:
- Prioritize elements that improved risk-adjusted returns (ratio = annual_return / max_drawdown)
- Avoid strategies that repeatedly failed in past iterations
- Be creative but realistic; do not just tweak numbers

OUTPUT FORMAT (STRICT JSON):
Return a JSON object with exactly two fields:

{{
  "main_prompt": "<refined main strategy prompt text (with Calculations Required and Output Columns sections)>",
  "condition_prompt": "<refined condition prompt text>"
}}

Do NOT include any explanations, markdown, or extra text outside this JSON object.
Only return the pure JSON object.

GENERATE AN IMPROVED STRATEGY PROMPT BASED ON HISTORY:"""

    return prompt

def build_combined_strategy_prompt() -> str:
    """Build a combined prompt for Gemini to generate both main strategy and condition in JSON format with internal refinement and explicit structure"""

    return """You are an expert quantitative trading strategy designer and prompt engineer.

DATABASE AVAILABLE:
- File: historical_data_500_tickers_with_gains.db
- Table: stock_data
- Columns: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct

YOUR TASK:
You must internally refine the generated trading strategy before producing the final output.

REFINEMENT RULES (APPLY INTERNALLY):
Transform the main strategy idea into a **concise, step-by-step instruction** specifying:

1. **Calculations Required**  
   Describe exactly what to calculate and from which base fields.  
   Use clear, short bullet points (e.g., “Calculate 10-day SMA of Adj_Close”).  
   Only use these base DB columns for calculations: Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct.  

2. **Output Columns and Content**  
   Define what each column in the final CSV will represent.  
   The first 5 columns must always be:
   - Date  
   - Ticker  
   - Adj_Close  
   - Daily_Gain_Pct  
   - Forward_Gain_Pct  
   Any newly calculated columns come **after** these five, with clear and consistent names.

CRITICAL REQUIREMENT:
Do NOT rename or remove the first 5 base columns.  
Do NOT include unrelated instructions (like data cleaning or formatting).  
Focus only on calculations and column definitions.

REFINEMENT EXAMPLE (STRICT FORMAT):
1. **Calculations Required**:
    * Calculate the 10-day Simple Moving Average (SMA) of the 'Adj_Close' price.
    * Calculate the 5-day Simple Moving Average (SMA) of the 'Adj_Close' price.

2. **Output Columns and Content**:
    * Date: The trading date.
    * Ticker: The stock ticker symbol.
    * Adj_Close: The adjusted closing price for the day.
    * Daily_Gain_Pct: The daily percentage change in adjusted closing price.
    * Forward_Gain_Pct: The percentage change in adjusted closing price from the current day to the next trading day.
    * SMA_10_Day: The 10-day Simple Moving Average of 'Adj_Close'.
    * SMA_5_Day: The 5-day Simple Moving Average of 'Adj_Close'.

FINAL TASK:
After applying this refinement logic, generate two outputs:

1. **main_prompt** → The fully refined strategy in the same structured format as shown above (with “Calculations Required” and “Output Columns and Content” sections).  
   It must clearly define all calculations and column meanings, formatted as a readable list.

2. **condition_prompt** → A direct and specific rule describing when Signal=1 or Signal=0 using any of the columns (both base and calculated).  
   Example:
   "Signal=1 when SMA_5_Day crosses above SMA_10_Day AND Daily_Gain_Pct > 0, else Signal=0."  
   Keep it short (1–2 sentences).

OUTPUT FORMAT (MANDATORY):
Return your final response strictly in this JSON structure:

{
  "main_prompt": "<refined main strategy prompt text (with the 2 sections above)>",
  "condition_prompt": "<refined condition prompt text>"
}

No markdown, no code fences, no explanations, and no text outside this JSON.
Only return the pure JSON object.

GENERATE NOW:"""
