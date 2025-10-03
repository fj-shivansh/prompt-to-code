#!/usr/bin/env python3
"""
Generated Code Execution Module
This file contains AI-generated code for execution.
"""

import pandas as pd

# Set pandas display options (mandatory)
pd.set_option('display.max_rows', None)

# Read the output.csv file
df = pd.read_csv('output.csv')

# MANDATORY: Sort data by Ticker and Date BEFORE any operations
df = df.sort_values(['Ticker', 'Date'], ascending=[True, True])

# Interpret the condition: 'when 10 day moving avg is less than that of 5 day'
# Create a 'Signal' column (1 for True, 0 for False)
df['Signal'] = (df['SMA_10_Day'] < df['SMA_5_Day']).astype(int)

# Define the exact order of columns for the output CSV
# Step 1: Get all database columns first (from the prompt's base requirements)
db_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']

# Step 2: Get your existing calculated columns (from output.csv) and the new 'Signal' column
existing_calculated_columns = ['SMA_10_Day', 'SMA_5_Day']
new_calculated_column = ['Signal']

# Step 3: MANDATORY - Combine db_columns FIRST, then other calculated columns
final_column_order = db_columns + existing_calculated_columns + new_calculated_column

# Ensure the DataFrame has the columns in the specified order
final_df = df[final_column_order].copy()

# MANDATORY: Sort by Date DESC (latest dates first) before saving
final_df = final_df.sort_values('Date', ascending=False)

# Save the result to 'condition_output_llm2.csv'
final_df.to_csv('condition_output_llm2.csv', index=False)
