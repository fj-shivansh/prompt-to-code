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
ORDER BY Date, Ticker
'''
df = pd.read_sql_query(query, conn)
conn.close()

# Group data by date
grouped = df.groupby('Date')

# Calculate daily percentage change
df['Daily_Pct_Change'] = grouped['Adj_Close'].pct_change()

# Find the ticker with the lowest percentage change for each day
result = []
for date, group in grouped:
    min_change_row = group.loc[group['Daily_Pct_Change'].idxmin()]
    next_day_change = 0.0
    next_day_date_str = min_change_row['Date']
    try:
        next_day = df[(df['Date'] > next_day_date_str) & (df['Ticker'] == min_change_row['Ticker'])]
        next_day_change = next_day['Daily_Pct_Change'].iloc[0]
    except IndexError:
        pass  #Handle cases where there is no next day data
    result.append([
        min_change_row['Date'],
        min_change_row['Ticker'],
        min_change_row['Daily_Pct_Change'],
        next_day_change
    ])

# Create DataFrame and sort
result_df = pd.DataFrame(result, columns=['Date', 'Ticker', 'Daily_Pct_Change', 'Next_Day_Pct_Change'])
result_df = result_df.sort_values('Date', ascending=False)

# Save to CSV
result_df.to_csv('output.csv', index=False)
print('Results saved to output.csv')
