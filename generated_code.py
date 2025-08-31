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
conn = sqlite3.connect('historical_data.db')

# Query the database
query = '''
SELECT Date, Ticker, Adj_Close
FROM stock_data
WHERE Adj_Close IS NOT NULL
ORDER BY Date ASC; 
'''
df = pd.read_sql_query(query, conn)
conn.close()

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by=['Date'])

# Group data by date and ticker
groups = df.groupby(['Date', 'Ticker'])

# Calculate daily percentage change
df['Open'] = groups['Adj_Close'].transform('first')
df['Daily_Percent_Change'] = ((df['Adj_Close'] - df['Open']) / df['Open']) * 100

# Find the minimum percentage change for each day
min_change_df = df.loc[df.groupby('Date')['Daily_Percent_Change'].idxmin()]

# Create a copy to avoid SettingWithCopyWarning
next_day_df = df.copy()

# Shift data to get the next day's data
next_day_df['Next_Day_Adj_Close'] = next_day_df.groupby('Ticker')['Adj_Close'].shift(-1)
next_day_df['Next_Day_Date'] = next_day_df.groupby('Ticker')['Date'].shift(-1)

# Merge to get the next day's data for the minimum change ticker
min_change_df = min_change_df.merge(next_day_df[['Date', 'Ticker', 'Next_Day_Adj_Close', 'Next_Day_Date']], on=['Date', 'Ticker'], how='left', suffixes=('', '_next'))

# Calculate the subsequent day's percentage change
min_change_df['Subsequent_Day_Percent_Change'] = ((min_change_df['Next_Day_Adj_Close'] - min_change_df['Adj_Close']) / min_change_df['Adj_Close']) * 100

#Select required columns and rename for clarity
result_df = min_change_df[['Date', 'Ticker', 'Daily_Percent_Change', 'Subsequent_Day_Percent_Change']]
result_df = result_df.rename(columns={'Daily_Percent_Change': 'Min_Daily_Percent_Change'})

#Handle potential NaN values. Fill with 0 for simplicity.  More sophisticated handling might be appropriate in a production environment.
result_df = result_df.fillna(0)

# Sort by date in descending order
result_df = result_df.sort_values('Date', ascending=False)

# Save to CSV
result_df.to_csv('output.csv', index=False)

print('Results saved to output.csv')
