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

# Fetch data from the database. Handle potential errors.
try:
    df = pd.read_sql_query("SELECT Date, Ticker, Adj_Close FROM stock_data WHERE Adj_Close IS NOT NULL", conn)
except sqlite3.Error as e:
    print(f"Error fetching data from database: {e}")
    exit(1)
finally:
    conn.close()

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort the DataFrame by Date and Ticker
df = df.sort_values(['Date', 'Ticker'])

# Create a pivot table to get opening and closing prices for each ticker on each day
pivot_df = df.pivot_table(index='Date', columns='Ticker', values='Adj_Close')

#Calculate daily percent change
daily_pct_change = (pivot_df.diff().div(pivot_df.shift()) * 100).stack().reset_index(name='Daily Percent Change')

# Find minimum daily percent change for each day
min_daily_pct_change = daily_pct_change.loc[daily_pct_change.groupby('Date')['Daily Percent Change'].idxmin()]

#Merge to get ticker
min_daily_pct_change = min_daily_pct_change.merge(df, on=['Date', 'Ticker'], how='left')

#Calculate subsequent day percent change
min_daily_pct_change['next_day'] = min_daily_pct_change['Date'] + pd.Timedelta(days=1)
min_daily_pct_change = min_daily_pct_change.merge(df, left_on=['next_day','Ticker'], right_on=['Date', 'Ticker'], how='left', suffixes=('_current','_next'))
min_daily_pct_change['Subsequent Day Percent Change'] = ((min_daily_pct_change['Adj_Close_next'] - min_daily_pct_change['Adj_Close_current']) / min_daily_pct_change['Adj_Close_current']) * 100
min_daily_pct_change = min_daily_pct_change[['Date_current','Ticker','Daily Percent Change','Subsequent Day Percent Change']]
min_daily_pct_change = min_daily_pct_change.rename(columns={'Date_current':'Date'})
min_daily_pct_change['Subsequent Day Percent Change'] = min_daily_pct_change['Subsequent Day Percent Change'].fillna(0)

#Sort by date in descending order
min_daily_pct_change = min_daily_pct_change.sort_values('Date', ascending=False)

#Save to CSV
min_daily_pct_change.to_csv('output.csv', index=False)

print('Results saved to output.csv')
