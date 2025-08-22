#!/usr/bin/env python3
"""
Simplified Prompt-to-Code Testing System
Generates code and tests it by writing to separate files
"""

import sqlite3
import json
import subprocess
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import google.generativeai as genai


@dataclass
class TestResult:
    success: bool
    output: str
    execution_time: float
    error: Optional[str] = None


@dataclass 
class CodeGeneration:
    code: str
    explanation: str
    task: str


class SimplePromptToCodeSystem:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def get_sample_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get sample data from database"""
        conn = sqlite3.connect("historical_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT Date, Ticker, Adj_Close FROM stock_data LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [{"Date": row[0], "Ticker": row[1], "Adj_Close": row[2]} for row in rows]
    
    def generate_code(self, task: str) -> CodeGeneration:
        """Generate code using Gemini API"""
        prompt = f"""You are given stock market data in Python. Each record is a dictionary with these keys:
- Date: string, format "YYYY-MM-DD HH:MM:SS"  
- Ticker: string, stock symbol like "AAPL"
- Adj_Close: float, adjusted closing price

Example input:
data = [
    {{"Date": "2000-01-03 00:00:00", "Ticker": "AAPL", "Adj_Close": 0.840094149112701}},
    {{"Date": "2000-01-04 00:00:00", "Ticker": "AAPL", "Adj_Close": 0.769265949726105}},
    {{"Date": "2000-01-05 00:00:00", "Ticker": "AAPL", "Adj_Close": 0.780522704124451}},
    {{"Date": "2000-01-06 00:00:00", "Ticker": "AAPL", "Adj_Close": 0.712977468967438}},
    {{"Date": "2000-01-07 00:00:00", "Ticker": "AAPL", "Adj_Close": 0.746750473976135}},
    {{"Date": "2000-01-03 00:00:00", "Ticker": "MSFT", "Adj_Close": 39.813}},
    {{"Date": "2000-01-04 00:00:00", "Ticker": "MSFT", "Adj_Close": 38.438}}
]

TASK: {task}

Expected response format (JSON):
{{
    "code": "your Python function here as a string",
    "explanation": "your step-by-step explanation here as a string"
}}"""
        
        response = self.model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean JSON response
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        try:
            parsed = json.loads(response_text)
            code = parsed['code']
            
            # Clean code from markdown blocks
            if code.startswith('```python'):
                code = code[9:].rstrip('```').strip()
            elif code.startswith('```'):
                code = code[3:].rstrip('```').strip()
            
            return CodeGeneration(
                code=code,
                explanation=parsed['explanation'], 
                task=task
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to parse Gemini response: {e}")
    
    def test_code(self, generation: CodeGeneration, test_ticker: str = "AAPL") -> TestResult:
        """Test generated code by writing to file and executing"""
        
        # Create test file
        test_code = f'''#!/usr/bin/env python3
import sqlite3
from typing import List, Dict

def load_test_data():
    conn = sqlite3.connect('historical_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT Date, Ticker, Adj_Close FROM stock_data")
    rows = cursor.fetchall()
    conn.close()
    return [{{"Date": row[0], "Ticker": row[1], "Adj_Close": row[2]}} for row in rows]

# Generated code
{generation.code}

# Test the function
if __name__ == "__main__":
    try:
        data = load_test_data()
        print(f"Loaded {{len(data)}} records")
        
        # Find function name
        import inspect
        functions = [name for name, obj in globals().items() 
                    if inspect.isfunction(obj) and name != 'load_test_data']
        
        if not functions:
            print("ERROR: No function found in generated code")
            exit(1)
        
        func_name = functions[0]
        func = globals()[func_name]
        
        # Test the function
        result = func(data, "{test_ticker}")
        print(f"SUCCESS: {{result}}")
        
    except Exception as e:
        print(f"ERROR: {{str(e)}}")
        import traceback
        traceback.print_exc()
        exit(1)
'''
        
        # Write to temporary file
        test_filename = "temp_test_code.py"
        with open(test_filename, 'w') as f:
            f.write(test_code)
        
        # Execute and capture results
        start_time = time.time()
        try:
            result = subprocess.run(
                ["python", test_filename], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return TestResult(
                    success=True,
                    output=result.stdout.strip(),
                    execution_time=execution_time
                )
            else:
                return TestResult(
                    success=False,
                    output=result.stdout.strip(),
                    execution_time=execution_time,
                    error=result.stderr.strip()
                )
        except subprocess.TimeoutExpired:
            return TestResult(
                success=False,
                output="",
                execution_time=30.0,
                error="Execution timeout"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                success=False,
                output="",
                execution_time=execution_time,
                error=str(e)
            )
        finally:
            # Clean up
            if os.path.exists(test_filename):
                os.remove(test_filename)
    
    def process_task(self, task: str, test_ticker: str = "AAPL") -> Dict[str, Any]:
        """Process complete task: generate and test code"""
        
        print(f"🎯 Task: {task}")
        print("-" * 60)
        
        # Generate code
        try:
            print("🤖 Generating code with Gemini API...")
            generation = self.generate_code(task)
            print("✅ Code generated successfully!")
        except Exception as e:
            return {"error": f"Code generation failed: {str(e)}"}
        
        # Test code
        print("🧪 Testing generated code...")
        test_result = self.test_code(generation, test_ticker)
        
        # Get data stats
        sample_data = self.get_sample_data(10)
        unique_tickers = len(set(d["Ticker"] for d in sample_data))
        
        return {
            "generation": generation,
            "test_result": test_result,
            "data_info": {
                "sample_size": len(sample_data),
                "unique_tickers": unique_tickers
            }
        }


def run_examples():
    """Run example tasks"""
    
    api_key = "AIzaSyC8jMKReXdf8WnHNOo1_m8HAmIjvEs14ks"
    system = SimplePromptToCodeSystem(api_key)
    
    tasks = [
        "Write a function that calculates the average adjusted close price for a given ticker.",
        "Write a function that finds the highest adjusted close price for a given ticker.",
        "Write a function that counts how many records exist for a given ticker.",
        "Write a function that calculates the price difference between first and last record for a ticker."
    ]
    
    print("="*70)
    print("🚀 PROMPT-TO-CODE TESTING SYSTEM")  
    print("="*70)
    
    for i, task in enumerate(tasks, 1):
        print(f"\n📋 EXAMPLE {i}")
        result = system.process_task(task, "AAPL")
        
        if "error" in result:
            print(f"❌ Failed: {result['error']}")
            continue
            
        generation = result["generation"]
        test_result = result["test_result"]
        
        print(f"\n📝 Generated Code:")
        print(generation.code)
        
        print(f"\n💭 Explanation:")
        print(generation.explanation)
        
        print(f"\n🧪 Test Results:")
        if test_result.success:
            print(f"✅ SUCCESS: {test_result.output}")
            print(f"⏱️  Execution time: {test_result.execution_time:.4f}s")
        else:
            print(f"❌ FAILED: {test_result.error}")
            if test_result.output:
                print(f"Output: {test_result.output}")
        
        print("-" * 60)


if __name__ == "__main__":
    run_examples()