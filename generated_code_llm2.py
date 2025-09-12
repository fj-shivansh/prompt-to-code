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
conn = sqlite3.connect('historical_data_with_gains.db')

# Query the database
query = '''
SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct
FROM stock_data
WHERE Adj_Close IS NOT NULL
ORDER BY Ticker, Date
'''
df = pd.read_sql_query(query, conn)
conn.close()

# Calculate moving averages
df['5_Day_MA'] = df.groupby('Ticker')['Adj_Close'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
df['20_Day_MA'] = df.groupby('Ticker')['Adj_Close'].rolling(window=20, min_periods=1).mean().reset_index(0, drop=True)

# Calculate positive daily gain count
df['Positive_Day_Count'] = df.groupby('Ticker')['Daily_Gain_Pct'].rolling(window=5, min_periods=1).apply(lambda x: (x > 0).sum(), raw=False).reset_index(0, drop=True)

# Identify momentum stocks
df['Momentum_Condition'] = (df['5_Day_MA'] > df['20_Day_MA']) & (df['Positive_Day_Count'] >= 3)

#Calculate average daily gain over last 5 days
df['Avg_Daily_Gain_5d'] = df.groupby('Ticker')['Daily_Gain_Pct'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)

#Calculate Momentum Score
df['Momentum_Score'] = ((df['5_Day_MA'] / df['20_Day_MA']) * df['Avg_Daily_Gain_5d']) * df['Momentum_Condition']

# Get the top 25 momentum stocks
top_25 = df[df['Momentum_Condition']].sort_values('Momentum_Score', ascending=False).head(25)

# Prepare final dataframe
db_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']
calculated_columns = ['5_Day_MA', '20_Day_MA', 'Positive_Day_Count', 'Momentum_Condition', 'Avg_Daily_Gain_5d', 'Momentum_Score']
final_df = top_25[db_columns + calculated_columns]
final_df = final_df.sort_values('Date', ascending=False)

# Save to CSV
final_df.to_csv('output_llm2.csv', index=False)
print('Results saved to output_llm2.csv')
