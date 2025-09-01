#!/usr/bin/env python3
"""
Generated Code Execution Module
This file contains AI-generated code for execution.
"""

#!/usr/bin/env python3
import sqlite3
import pandas as pd

pd.set_option('display.max_rows', None)

conn = sqlite3.connect('historical_data.db')
query = '''
SELECT Date, Ticker, Adj_Close
FROM stock_data
WHERE Adj_Close IS NOT NULL
ORDER BY Date, Ticker;
'''
df = pd.read_sql_query(query, conn)
conn.close()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Date', 'Ticker'])

df = df.set_index(['Date', 'Ticker'])

open_prices = df['Adj_Close'].groupby(level='Date').first()
close_prices = df['Adj_Close'].groupby(level='Date').last()

df['Open'] = df.groupby(level='Date')['Adj_Close'].transform('first')
df['Close'] = df.groupby(level='Date')['Adj_Close'].transform('last')

def percent_change(row):
    return ((row['Close'] - row['Open']) / row['Open']) * 100

df['pct_change'] = df.apply(percent_change, axis=1)

df_daily_min = df.groupby(level='Date')['pct_change'].agg(['idxmin', 'min'])
df_daily_min = df_daily_min.rename(columns={'idxmin': 'Ticker', 'min': 'Current Day % Change'})
df_daily_min['Ticker'] = df_daily_min['Ticker'].apply(lambda x: x[1])

next_day_changes = []
for date, row in df_daily_min.iterrows():
    next_date = date + pd.Timedelta(days=1)
    try:
        next_day_ticker = row['Ticker']
        next_day_change = df.loc[(next_date, next_day_ticker), 'pct_change']
        next_day_changes.append(next_day_change)
    except KeyError:
        next_day_changes.append(None)

df_daily_min['Next Day % Change'] = next_day_changes
df_daily_min = df_daily_min.fillna(0)

df_daily_min = df_daily_min.sort_values('Date', ascending=False)
df_daily_min.to_csv('output.csv', index=True)
print('Results saved to output.csv')
