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

# Get data for specific tickers using the mandatory template
selected_tickers = ['SWK', 'OKE', 'TFC', 'EXR', 'CAT', 'JNJ', 'SNPS', 'CTSH', 'RF', 'BALL']
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

# Calculate SMA_10_Day as the 10-period Simple Moving Average of Adj_Close
# Using min_periods=1 to calculate with fewer than 10 periods at the start of the series
df['SMA_10_Day'] = df.groupby('Ticker')['Adj_Close'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)

# Calculate SMA_5_Day as the 5-period Simple Moving Average of Adj_Close
# Using min_periods=1 to calculate with fewer than 5 periods at the start of the series
df['SMA_5_Day'] = df.groupby('Ticker')['Adj_Close'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

# MANDATORY TEMPLATE FOR FINAL DATAFRAME
# Step 1: Get all database columns first
db_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']

# Step 2: Get your calculated columns
calculated_columns = ['SMA_10_Day', 'SMA_5_Day']

# Step 3: MANDATORY - Combine db_columns FIRST, then calculated columns
final_df = df[db_columns + calculated_columns].copy()

# MANDATORY: Final sort by Date DESC before saving
final_df = final_df.sort_values('Date', ascending=False)
final_df.to_csv('output_llm2.csv', index=False)

print("Calculations complete and results saved to 'output_llm2.csv' with mandatory column order and sorting.")
