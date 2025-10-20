"""
Prompt building utility for code generation
"""


def build_code_generation_prompt(task, error_context=None, failed_code=None, output_file="output.csv",
                                 ticker_filters=None, date_filters=None) -> str:
    """Build the complete prompt for code generation"""

    # Add error context if this is a retry attempt
    error_section = ""
    if error_context and failed_code:
        error_section = f"""
ERROR CONTEXT - Fix the following code:
ERROR: {error_context}

FAILED CODE:
{failed_code}

Common issues: indentation, pandas groupby column access, data types, missing imports, syntax errors.
Complete the ORIGINAL TASK while fixing the error.

"""

    # Build filter instruction
    filter_instructions = ""

    if ticker_filters or date_filters:
        filter_instructions = "\nDATA FILTERS:\n"

        selected_tickers = None
        if ticker_filters and ticker_filters.get('selected_tickers'):
            selected_tickers = ticker_filters['selected_tickers']
            filter_instructions += f"- Use ONLY these {len(selected_tickers)} tickers: {', '.join(selected_tickers)}\n"
        elif ticker_filters and ticker_filters.get('ticker_count') == 'all':
            filter_instructions += f"- Use ALL available tickers\n"

        if date_filters:
            start_date = date_filters.get('start_date')
            end_date = date_filters.get('end_date')
            if start_date and end_date:
                filter_instructions += f"- Date range: {start_date} to {end_date}\n"

        filter_instructions += "\nDatabase query template:\n```python\n"

        if selected_tickers:
            filter_instructions += f"selected_tickers = {selected_tickers}\n"
            filter_instructions += "placeholders = ','.join(['?' for _ in selected_tickers])\n"
            filter_instructions += 'query = f"SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL AND Ticker IN ({placeholders})"\n'
            filter_instructions += "params = selected_tickers\n"
        else:
            filter_instructions += 'query = "SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL"\n'
            filter_instructions += "params = []\n"

        if date_filters:
            start_date = date_filters.get('start_date')
            end_date = date_filters.get('end_date')
            if start_date:
                filter_instructions += f'query += " AND Date >= ?"\nparams.append("{start_date}")\n'
            if end_date:
                filter_instructions += f'query += " AND Date <= ?"\nparams.append("{end_date}")\n'

        filter_instructions += "```\n"

    prompt = f"""{error_section}{filter_instructions}Generate Python code for this task: {task}

DATABASE INFO:
- File: historical_data_500_tickers_with_gains.db
- Table: stock_data
- Columns: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct

MANDATORY: CONNECT TO THE ACTUAL DATABASE - DO NOT CREATE DUMMY DATA
You MUST use sqlite3 to connect to the database file and query real data.

Example database connection code:
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('historical_data_500_tickers_with_gains.db')
cursor = conn.cursor()
cursor.execute("SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL")
rows = cursor.fetchall()
conn.close()

df = pd.DataFrame(rows, columns=['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct'])
```

CRITICAL REQUIREMENTS:
1. MUST connect to database file and query real data (NO dummy/simulated data!)
2. Output CSV MUST have these 5 columns FIRST: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct
3. Save result as: {output_file}
4. Sort by Ticker,Date before calculations: df.sort_values(['Ticker','Date'], ascending=[True,True])
5. Sort by Date DESC before saving: df.sort_values('Date', ascending=False)
6. Format dates: df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
7. DO NOT include built-in modules in requirements (sqlite3, os, sys, time, json)

CODE REQUIREMENTS:
- Complete standalone Python code
- MUST use sqlite3.connect() to load real database data
- 4-space indentation, no tabs
- Filter NULL values: WHERE Adj_Close IS NOT NULL
- Configure pandas: pd.set_option('display.max_rows', None)
- No type hints in function signatures

Return JSON format:
{{
    "code": "complete Python code as string",
    "explanation": "brief explanation",
    "requirements": ["external packages only, e.g. pandas"]
}}
"""

    return prompt
