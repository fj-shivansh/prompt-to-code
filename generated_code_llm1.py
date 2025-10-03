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

# Connect to the SQLite database
conn = sqlite3.connect('historical_data_500_tickers_with_gains.db')
cursor = conn.cursor()

# Get data for specific tickers using the pre-selected list
selected_tickers = ['ERIE', 'BEN', 'TEL', 'CEG', 'BXP', 'PAYX', 'ARE', 'ANET', 'TPL', 'PLD']
placeholders = ','.join(['?' for _ in selected_tickers])

# Query to select all 5 mandatory columns, filtering for non-NULL values and specific tickers
query = f"""SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct 
            FROM stock_data 
            WHERE Adj_Close IS NOT NULL 
            AND Daily_Gain_Pct IS NOT NULL 
            AND Forward_Gain_Pct IS NOT NULL 
            AND Ticker IN ({placeholders})"""
params = selected_tickers

query += " ORDER BY Ticker, Date"

try:
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Convert to DataFrame with EXACT column names
    df = pd.DataFrame(rows, columns=['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct'])

except sqlite3.Error as e:
    print(f"Database error: {e}")
    df = pd.DataFrame()
finally:
    conn.close()

# Check if DataFrame is empty after loading
if df.empty:
    print("No data loaded or an error occurred. Creating an empty CSV.")
    # Create an empty DataFrame with the required column structure for the CSV
    final_df = pd.DataFrame(columns=['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct', 'SMA_10_Day', 'SMA_5_Day'])
else:
    # MANDATORY: Sort by Ticker and Date BEFORE any operations
    df = df.sort_values(['Ticker', 'Date'], ascending=[True, True])

    # Calculate 10-day Simple Moving Average (SMA) of Adj_Close, grouped by Ticker
    # min_periods=1 ensures that SMA is calculated even if fewer than 10 data points are available
    df['SMA_10_Day'] = df.groupby('Ticker')['Adj_Close'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)

    # Calculate 5-day Simple Moving Average (SMA) of Adj_Close, grouped by Ticker
    # min_periods=1 ensures that SMA is calculated even if fewer than 5 data points are available
    df['SMA_5_Day'] = df.groupby('Ticker')['Adj_Close'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

    # MANDATORY TEMPLATE: Ensure the final DataFrame has the 5 database columns first
    # followed by any calculated columns in the specified order.
    db_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']
    calculated_columns = ['SMA_10_Day', 'SMA_5_Day']

    # Combine db_columns FIRST, then calculated columns
    final_df = df[db_columns + calculated_columns].copy()

    # MANDATORY: Final sort by Date DESC (latest first) before saving
    final_df = final_df.sort_values('Date', ascending=False)

# Save the final DataFrame to a CSV file
output_file_name = 'output_llm1.csv'
final_df.to_csv(output_file_name, index=False)

print(f"Results saved to {output_file_name} with exact database column names and Date DESC sorting.")
