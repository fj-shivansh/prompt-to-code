#!/usr/bin/env python3
"""
Generated Condition Code Execution Module
"""

#!/usr/bin/env python3
import pandas as pd

# Set pandas display options
pd.set_option('display.max_rows', None)

try:
    # Read the CSV file
    df = pd.read_csv('output.csv')

    # Map natural language condition to column names
    # We'll assume 'ClosingPrice' is 'Adj_Close', '10DayMA' is '10_Day_MA', and '5DayMA' is '5_Day_MA'
    # If these column names are different in your actual CSV, adjust accordingly.
    
    #Handle potential missing columns gracefully
    if 'Adj_Close' not in df.columns or '10DayMA' not in df.columns or '5DayMA' not in df.columns:
        raise ValueError("Missing required columns in 'output.csv'. Check column names for 'Adj_Close', '10DayMA', and '5DayMA'.")
    
    # Apply the condition and create the 'Signal' column
    df['Signal'] = (df['10DayMA'] < df['5DayMA']).astype(int)

    # Ensure all original columns are present and in the correct order
    final_df = df[['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct', 'Signal']]

    # Sort by Date in descending order
    final_df = final_df.sort_values('Date', ascending=False)

    # Save the result to a new CSV file
    final_df.to_csv('condition_output.csv', index=False)
    print('Results saved to condition_output.csv')

except FileNotFoundError:
    print("Error: 'output.csv' not found. Please ensure the file exists.")
except pd.errors.EmptyDataError:
    print("Error: 'output.csv' is empty.")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

