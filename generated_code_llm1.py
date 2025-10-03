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

# Define the exact pre-selected tickers as per requirements
selected_tickers = ['SWK', 'OKE', 'TFC', 'EXR', 'CAT', 'JNJ', 'SNPS', 'CTSH', 'RF', 'BALL']

# Create placeholders for the IN clause in the SQL query
placeholders = ','.join(['?' for _ in selected_tickers])

# MANDATORY DATABASE QUERY TEMPLATE WITH FILTERS
# Query to get data for specific tickers, ensuring non-null values for critical columns
query = f"SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL AND Ticker IN ({placeholders})"
params = selected_tickers

# MANDATORY: Add ORDER BY Ticker, Date to the query
query += " ORDER BY Ticker, Date"
cursor.execute(query, params)

# Fetch all rows and close the connection
rows = cursor.fetchall()
conn.close()

# Convert to DataFrame with exact required column names
df = pd.DataFrame(rows, columns=['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct'])

# MANDATORY: Sort by Ticker and Date for processing IMMEDIATELY AFTER LOADING DATA
df = df.sort_values(['Ticker', 'Date'], ascending=[True, True])

# --- Calculations ---
# Calculate SMA_10_Day: 10-period Simple Moving Average of Adj_Close
# Using min_periods=1 as per requirement 14
df['SMA_10_Day'] = df.groupby('Ticker')['Adj_Close'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)

# Calculate SMA_5_Day: 5-period Simple Moving Average of Adj_Close
# Using min_periods=1 as per requirement 14
df['SMA_5_Day'] = df.groupby('Ticker')['Adj_Close'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

# --- Final DataFrame Preparation ---
# Define the mandatory database columns in their required order
db_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']

# Define the newly calculated columns
calculated_columns = ['SMA_10_Day', 'SMA_5_Day']

# MANDATORY: Combine db_columns FIRST, then calculated columns for the final DataFrame
final_df = df[db_columns + calculated_columns].copy()

# MANDATORY: Final sort by Date DESC before saving
final_df = final_df.sort_values('Date', ascending=False)

# Save the final DataFrame to output_llm1.csv
output_file = 'output_llm1.csv'
final_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file} with required columns and sorting.")

