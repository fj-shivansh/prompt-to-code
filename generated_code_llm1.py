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

# Connect to database
conn = sqlite3.connect('historical_data_500_tickers_with_gains.db')
cursor = conn.cursor()

# Get data for specific tickers using the provided template
selected_tickers = ['AES', 'DELL', 'EQR', 'CBOE', 'IBM', 'O', 'ENPH', 'LOW', 'C', 'PWR']
placeholders = ','.join(['?' for _ in selected_tickers])
query = f"SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL AND Ticker IN ({placeholders})"
params = selected_tickers

query += " ORDER BY Ticker, Date"
cursor.execute(query, params)

# Convert to DataFrame and ensure proper sorting
rows = cursor.fetchall()
conn.close()
df = pd.DataFrame(rows, columns=['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct'])

# MANDATORY: Sort by Ticker and Date for processing
df = df.sort_values(['Ticker', 'Date'], ascending=[True, True])

# Calculate 10-day Simple Moving Average (SMA_10_Day)
# min_periods=1 ensures calculation even if fewer than 10 data points are available
df['SMA_10_Day'] = df.groupby('Ticker')['Adj_Close'].rolling(window=10, min_periods=1).mean().values

# Calculate 5-day Simple Moving Average (SMA_5_Day)
# min_periods=1 ensures calculation even if fewer than 5 data points are available
df['SMA_5_Day'] = df.groupby('Ticker')['Adj_Close'].rolling(window=5, min_periods=1).mean().values

# MANDATORY TEMPLATE: Construct final_df with database columns first, then calculated columns
db_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']
calculated_columns = ['SMA_10_Day', 'SMA_5_Day']

final_df = df[db_columns + calculated_columns].copy()

# MANDATORY: Final sort by Date DESC before saving
final_df = final_df.sort_values('Date', ascending=False)

# Save to CSV
final_df.to_csv('output_llm1.csv', index=False)

