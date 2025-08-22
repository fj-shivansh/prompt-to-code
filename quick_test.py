#!/usr/bin/env python3
"""Quick test of the system"""

import os
from main import PromptToCodeSystem

def test_with_api():
    api_key = "AIzaSyC8jMKReXdf8WnHNOo1_m8HAmIjvEs14ks"
    
    try:
        system = PromptToCodeSystem(api_key)
        print("✅ System initialized successfully")
        
        task = "Write a function that calculates the average adjusted close price for a given ticker."
        print(f"Testing task: {task}")
        
        result = system.process_task(task, test_kwargs={'ticker': 'AAPL'})
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        test_result = result["test_result"]
        generation = result["generation"]
        
        print(f"\n📝 Generated Code:")
        print(generation.code)
        
        print(f"\n💭 Explanation:")
        print(generation.explanation)
        
        if test_result.success:
            print(f"\n✅ Test Result: {test_result.result}")
            print(f"⏱️  Execution Time: {test_result.execution_time:.4f}s")
        else:
            print(f"\n❌ Test Failed: {test_result.error}")
        
        # Show analytics
        analytics = result["analytics"]
        print(f"\n📊 Analytics:")
        print(f"Data records: {analytics['data_stats']['total_records']}")
        print(f"Success rate: {analytics['summary']['success_rate']:.1%}")
        
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_api()