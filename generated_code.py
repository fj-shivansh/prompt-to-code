#!/usr/bin/env python3
"""
Generated Code Execution Module
This file contains AI-generated code for execution.
"""

#!/usr/bin/env python3
import pandas as pd

# Set pandas display options
pd.set_option('display.max_rows', None)

# Load the CSV file
try:
    df = pd.read_csv('output.csv')
except FileNotFoundError:
    print("Error: output.csv not found. Please ensure the file exists in the same directory.")
    exit(1)

# Check for required columns
required_cols = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct', 'Abs_Daily_Gain_Pct', 'MA20_Abs_Daily_Gain_Pct']
if not all(col in df.columns for col in required_cols):
    print("Error: output.csv is missing required columns.")
    exit(1)

# Group by date and find the minimum 20-day average absolute daily gain
df['Date'] = pd.to_datetime(df['Date'])
df['Signal'] = 0
for date in df['Date'].unique():
    daily_data = df[df['Date'] == date]
    min_ma20 = daily_data['MA20_Abs_Daily_Gain_Pct'].min()
    min_tickers = daily_data[daily_data['MA20_Abs_Daily_Gain_Pct'] == min_ma20]['Ticker']
    for ticker in min_tickers:
      df.loc[(df['Date'] == date) & (df['Ticker'] == ticker), 'Signal'] = 1

#Reorder columns for final output
final_cols = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct', 'Abs_Daily_Gain_Pct', 'MA20_Abs_Daily_Gain_Pct', 'Signal']
final_df = df[final_cols]

# Sort by Date DESC before saving
final_df = final_df.sort_values('Date', ascending=False)

# Save the result to condition_output.csv
final_df.to_csv('condition_output.csv', index=False)

print('Results saved to condition_output.csv')
