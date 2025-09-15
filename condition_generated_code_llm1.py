#!/usr/bin/env python3
"""
Generated Code Execution Module
This file contains AI-generated code for execution.
"""

import pandas as pd

df = pd.read_csv('output.csv')
df['Signal'] = (df['MA5'] > 300).astype(int)
df.to_csv('condition_output_llm1.csv', index=False)
