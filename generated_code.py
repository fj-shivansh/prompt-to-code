#!/usr/bin/env python3
"""
Generated Code Execution Module
This file contains AI-generated code for execution.
"""

#!/usr/bin/env python3
import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('historical_data.db')

# Read data from the database
try:
    df = pd.read_sql_query("SELECT Date, Ticker, Adj_Close FROM stock_data WHERE Adj_Close IS NOT NULL", conn)
except pd.io.sql.DatabaseError as e:
    print(f"Error reading data from database: {e}")
    exit(1)

conn.close()

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as index
df = df.set_index('Date')

# Pivot the DataFrame to have Tickers as columns
df = df.pivot(columns='Ticker', values='Adj_Close')

# Calculate daily returns
daily_returns = df.pct_change()
daily_returns = daily_returns.dropna()

# Calculate rolling 5-day correlation
rolling_corr = daily_returns.rolling(5).corr()

# Prepare the output
output_df = pd.DataFrame()
output_df['Date'] = rolling_corr.index

tickers = daily_returns.columns

for i, ticker1 in enumerate(tickers):
    for ticker2 in tickers[i+1:]:
        corr_col_name = f"{ticker1}_{ticker2}_corr"
        output_df[corr_col_name] = rolling_corr[ticker1][ticker2]

#Handle cases with fewer than 5 days of data
output_df = output_df.fillna(method='bfill')

# Set display options to show all rows
pd.set_option('display.max_rows', None)

# Print the result
print(output_df)

