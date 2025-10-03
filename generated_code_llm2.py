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

# Connect to the database
conn = sqlite3.connect('historical_data_500_tickers_with_gains.db')
cursor = conn.cursor()

# Get data for specific tickers using the pre-selected list
selected_tickers = ['AES', 'DELL', 'EQR', 'CBOE', 'IBM', 'O', 'ENPH', 'LOW', 'C', 'PWR']
placeholders = ','.join(['?' for _ in selected_tickers])
query = f"SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL AND Ticker IN ({placeholders})"
params = selected_tickers

query += " ORDER BY Ticker, Date"
cursor.execute(query, params)

# Convert to DataFrame with exact column names
rows = cursor.fetchall()
conn.close()
df = pd.DataFrame(rows, columns=['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct'])

# MANDATORY: Sort by Ticker and Date for processing BEFORE any operations
df = df.sort_values(['Ticker', 'Date'], ascending=[True, True])

# Calculate 10-day Simple Moving Average (SMA_10_Day)
# Group by Ticker to ensure the moving average is calculated independently for each stock
# min_periods=1 ensures that SMA is calculated even if fewer than 10 data points are available
df['SMA_10_Day'] = df.groupby('Ticker')['Adj_Close'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)

# Calculate 5-day Simple Moving Average (SMA_5_Day)
# Group by Ticker to ensure the moving average is calculated independently for each stock
# min_periods=1 ensures that SMA is calculated even if fewer than 5 data points are available
df['SMA_5_Day'] = df.groupby('Ticker')['Adj_Close'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

# MANDATORY TEMPLATE FOR FINAL DATAFRAME CONSTRUCTION
# Step 1: Get all database columns first in their required order
db_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']

# Step 2: Get your calculated columns
calculated_columns = ['SMA_10_Day', 'SMA_5_Day']

# Step 3: MANDATORY - Combine db_columns FIRST, then calculated columns
# Using .copy() to ensure the final_df is a distinct DataFrame
final_df = df[db_columns + calculated_columns].copy()

# Step 4: MANDATORY - Sort by Date DESC (latest dates first) before saving
final_df = final_df.sort_values('Date', ascending=False)

# Step 5: Save the result to a CSV file
output_file = 'output_llm2.csv'
final_df.to_csv(output_file, index=False)

