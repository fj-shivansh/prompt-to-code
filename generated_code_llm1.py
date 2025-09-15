#!/usr/bin/env python3
"""
Generated Code Execution Module
This file contains AI-generated code for execution.
"""

#!/usr/bin/env python3
import sqlite3
import pandas as pd

# Set pandas display options
pd.set_option('display.max_rows', None)

# Connect to database and load ALL columns with EXACT names
conn = sqlite3.connect('historical_data_with_gains.db')
cursor = conn.cursor()
cursor.execute('SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL')
rows = cursor.fetchall()
conn.close()

# Convert to DataFrame with EXACT column names (NOT Close, NOT Symbol)
df = pd.DataFrame(rows, columns=['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct'])

# MANDATORY: Sort by Ticker and Date BEFORE any operations
df = df.sort_values(['Ticker', 'Date'], ascending=[True, True])

# Calculate 5-day moving average
df['MA5'] = df.groupby('Ticker')['Adj_Close'].rolling(window=5, min_periods=1).mean().values

# CRITICAL: Final DataFrame MUST have EXACT column names in EXACT order
db_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']
calculated_columns = ['MA5']
final_df = df[db_columns + calculated_columns]

# Sort by Date DESC (latest first) and save
final_df = final_df.sort_values('Date', ascending=False)
final_df.to_csv('output_llm1.csv', index=False)
print('Results saved to output_llm1.csv with EXACT database column names')
