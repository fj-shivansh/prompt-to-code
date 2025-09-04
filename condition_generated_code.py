#!/usr/bin/env python3
"""
Generated Condition Code Execution Module
"""

#!/usr/bin/env python3
import pandas as pd

# Set pandas display options
pd.set_option('display.max_rows', None)

# Load the CSV file
try:
    df = pd.read_csv('output.csv')
except FileNotFoundError:
    print("Error: output.csv not found. Please ensure the file exists.")
    exit(1)

# Map natural language condition to column names
condition_column_10 = 'MA10'
condition_column_5 = 'MA5'

# Check if the necessary columns exist
if condition_column_10 not in df.columns or condition_column_5 not in df.columns:
    print(f"Error: Columns '{condition_column_10}' or '{condition_column_5}' not found in output.csv.")
    exit(1)

# Apply the condition and create the 'Signal' column. Handle NaN values
df['Signal'] = ((df[condition_column_10] > df[condition_column_5]) | (df[condition_column_10].isna() | df[condition_column_5].isna())).astype(int)

# Ensure all original columns are present and in the correct order
required_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct', 'Signal']
if not all(col in df.columns for col in required_columns):
    print("Error: Missing required columns in the DataFrame. ")
    exit(1)

final_df = df[required_columns]

# Sort by Date DESC (latest first) and save to CSV
final_df = final_df.sort_values('Date', ascending=False)
final_df.to_csv('condition_output.csv', index=False)

print('Results saved to condition_output.csv')
