#!/usr/bin/env python3
"""Test generated code in a separate file"""

import sqlite3
from typing import List, Dict

# Load data from database
def load_test_data():
    conn = sqlite3.connect('historical_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT Date, Ticker, Adj_Close FROM stock_data LIMIT 1000")
    rows = cursor.fetchall()
    conn.close()
    return [{"Date": row[0], "Ticker": row[1], "Adj_Close": row[2]} for row in rows]

# Generated code from Gemini
from typing import List, Dict

def average_adj_close(data: List[Dict[str, object]], ticker: str) -> float:
    prices = [row['Adj_Close'] for row in data if row['Ticker'] == ticker]
    if not prices:
        return 0.0
    return sum(prices) / len(prices)

# Test the function
if __name__ == "__main__":
    try:
        print("Loading test data...")
        data = load_test_data()
        print(f"Loaded {len(data)} records")
        
        print("\nTesting generated function...")
        result = average_adj_close(data, "AAPL")
        print(f"✅ Success! Average AAPL price: {result}")
        
        # Test with another ticker
        result2 = average_adj_close(data, "MSFT")
        print(f"✅ Success! Average MSFT price: {result2}")
        
        # Test with non-existent ticker
        result3 = average_adj_close(data, "NONEXISTENT")
        print(f"✅ Success! Non-existent ticker returns: {result3}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()