#!/usr/bin/env python3
"""
Generated Code Execution Module
This file contains AI-generated code for execution.
"""

import pandas as pd

df = pd.read_csv('output.csv')
df['Signal'] = (df['MA10'] < df['MA5']).astype(int)
df = df[['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct', 'MA10', 'MA5', 'Signal']]
df.to_csv('condition_output_llm1.csv', index=False)
