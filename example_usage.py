#!/usr/bin/env python3
"""
Example usage of the Prompt-to-Code Testing System
"""

from main import PromptToCodeSystem
import os

def run_examples():
    """Run example tasks to demonstrate the system"""
    
    # Set your API key here or in environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
    
    system = PromptToCodeSystem(api_key)
    
    # Example tasks
    tasks = [
        "Write a function that calculates the average adjusted close price for a given ticker.",
        "Write a function that finds the date with the highest adjusted close price for a given ticker.",
        "Write a function that counts how many records exist for a given ticker.",
        "Write a function that calculates the price change percentage between two dates for a ticker."
    ]
    
    print("Running example tasks...")
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i}: {task}")
        print('='*60)
        
        # Run the task with AAPL as test ticker
        result = system.process_task(task, test_kwargs={"ticker": "AAPL"})
        
        if "error" in result:
            print(f"❌ Failed: {result['error']}")
            continue
        
        # Display results
        generation = result["generation"]
        test_result = result["test_result"]
        analytics = result["analytics"]
        
        print(f"\n📝 Generated Code:")
        print("-" * 40)
        print(generation.code)
        
        print(f"\n💭 Explanation:")
        print("-" * 40)
        print(generation.explanation)
        
        print(f"\n🧪 Test Results:")
        print("-" * 40)
        if test_result.success:
            print(f"✅ Success: {test_result.result}")
            print(f"⏱️  Execution time: {test_result.execution_time:.4f}s")
        else:
            print(f"❌ Failed: {test_result.error}")
        
        print(f"\n📊 Analytics:")
        print("-" * 40)
        print(f"Data records processed: {analytics['data_stats']['total_records']}")
        print(f"Unique tickers: {analytics['data_stats']['unique_tickers']}")
        print(f"Success rate: {analytics['summary']['success_rate']:.1%}")


if __name__ == "__main__":
    run_examples()