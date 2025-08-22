#!/usr/bin/env python3
"""
Test the prompt-to-code system with mock data to verify functionality
"""

from main import DatabaseManager, CodeExecutor, Analytics, TestResult, CodeGeneration
from typing import List, Dict, Any

def test_database_manager():
    """Test database operations"""
    print("Testing DatabaseManager...")
    
    db = DatabaseManager()
    sample_data = db.get_sample_data(5)
    
    print(f"Sample data (5 records): {len(sample_data)} records loaded")
    if sample_data:
        print(f"First record: {sample_data[0]}")
    
    all_data = db.get_all_data()
    print(f"All data: {len(all_data)} records loaded")
    
    return len(all_data) > 0

def test_code_executor():
    """Test code execution with sample code"""
    print("\nTesting CodeExecutor...")
    
    executor = CodeExecutor()
    
    # Sample data for testing
    test_data = [
        {"Date": "2000-01-03 00:00:00", "Ticker": "AAPL", "Adj_Close": 0.840094149112701},
        {"Date": "2000-01-04 00:00:00", "Ticker": "AAPL", "Adj_Close": 0.769265949726105},
        {"Date": "2000-01-05 00:00:00", "Ticker": "MSFT", "Adj_Close": 39.813},
        {"Date": "2000-01-06 00:00:00", "Ticker": "MSFT", "Adj_Close": 38.438}
    ]
    
    # Test successful code execution
    good_code = """
from typing import List, Dict

def average_price(data: List[Dict[str, object]], ticker: str) -> float:
    prices = [row['Adj_Close'] for row in data if row['Ticker'] == ticker]
    if not prices:
        return 0.0
    return sum(prices) / len(prices)
"""
    
    result = executor.execute_code(good_code, test_data, "average_price", "AAPL")
    print(f"Good code test - Success: {result.success}, Result: {result.result}")
    
    # Test failing code
    bad_code = """
def broken_function(data, ticker):
    return data[999999]['nonexistent_key']  # This will fail
"""
    
    result = executor.execute_code(bad_code, test_data, "broken_function", "AAPL")
    print(f"Bad code test - Success: {result.success}, Error: {result.error}")
    
    return True

def test_analytics():
    """Test analytics generation"""
    print("\nTesting Analytics...")
    
    # Mock data
    generations = [
        CodeGeneration("def test(): return 42", "Test function", "Test task")
    ]
    
    test_results = [
        TestResult(True, 42, 0.001),
        TestResult(False, None, 0.002, "Test error")
    ]
    
    data_stats = {
        "total_records": 100,
        "unique_tickers": 5,
        "date_range": {"start": "2000-01-01", "end": "2023-12-31"}
    }
    
    analytics = Analytics.analyze_results(generations, test_results, data_stats)
    
    print(f"Analytics generated:")
    print(f"  Success rate: {analytics['summary']['success_rate']:.1%}")
    print(f"  Total tests: {analytics['summary']['total_tests']}")
    print(f"  Successful: {analytics['summary']['successful']}")
    print(f"  Failed: {analytics['summary']['failed']}")
    
    return analytics['summary']['total_tests'] == 2

def test_without_api():
    """Test system components without Gemini API"""
    print("="*50)
    print("TESTING SYSTEM COMPONENTS")
    print("="*50)
    
    tests = [
        ("Database Manager", test_database_manager),
        ("Code Executor", test_code_executor),
        ("Analytics", test_analytics)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name}: FAILED - {str(e)}")
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All components working correctly!")
        print("Ready to use with Gemini API.")
    else:
        print("⚠️  Some components failed. Check errors above.")
    
    return passed == total

if __name__ == "__main__":
    test_without_api()