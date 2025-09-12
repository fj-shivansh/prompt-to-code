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
query = """
SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct
FROM stock_data
WHERE Adj_Close IS NOT NULL
ORDER BY Ticker, Date
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Calculate moving averages
df['5_Day_MA'] = df.groupby('Ticker')['Adj_Close'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
df['20_Day_MA'] = df.groupby('Ticker')['Adj_Close'].rolling(window=20, min_periods=1).mean().reset_index(0, drop=True)

# Calculate positive daily gain count
df['Positive_Day_Count'] = df.groupby('Ticker')['Daily_Gain_Pct'].rolling(window=5, min_periods=1).apply(lambda x: sum(x > 0), raw=False).reset_index(0, drop=True)

#Identify momentum stocks
df['Momentum_Stock'] = ((df['5_Day_MA'] > df['20_Day_MA']) & (df['Positive_Day_Count'] >= 3)).astype(int)

# Calculate average daily gain over last 5 days
df['Avg_Daily_Gain_5'] = df.groupby('Ticker')['Daily_Gain_Pct'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)

# Calculate momentum score
df['Momentum_Score'] = ((df['5_Day_MA'] / df['20_Day_MA']) * df['Avg_Daily_Gain_5']) * df['Momentum_Stock']

# Get top 25 momentum stocks
top_25 = df.sort_values(by=['Momentum_Score'], ascending=False).groupby('Ticker').head(1).nlargest(25, 'Momentum_Score')

# Ensure mandatory columns are first and in the correct order
db_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']
calculated_columns = ['5_Day_MA', '20_Day_MA', 'Positive_Day_Count', 'Momentum_Stock', 'Avg_Daily_Gain_5', 'Momentum_Score']
final_df = top_25[db_columns + calculated_columns]

#Sort by Date Descending
final_df = final_df.sort_values('Date', ascending=False)

# Save to CSV
final_df.to_csv('output_llm1.csv', index=False)

print('Results saved to output_llm1.csv')
