#!/usr/bin/env python3
"""
Generated Code Execution Module
This file contains AI-generated code for execution.
"""

import pandas as pd

pd.set_option('display.max_rows', None)

# Read the existing output.csv file
df = pd.read_csv('output.csv')

# Interpret the condition: 'when 10 day moving avg is less than that of 5 day'
# Create the 'Signal' column (1 for True, 0 for False)
df['Signal'] = (df['SMA_10_Day'] < df['SMA_5_Day']).astype(int)

# Define the exact final column order
final_columns_order = [
    'Date',
    'Ticker',
    'Adj_Close',
    'Daily_Gain_Pct',
    'Forward_Gain_Pct',
    'SMA_10_Day',
    'SMA_5_Day',
    'Signal'
]

# Select and reorder columns according to the requirement
df_final = df[final_columns_order]

# Save the result to 'condition_output_llm1.csv'
df_final.to_csv('condition_output_llm1.csv', index=False)
