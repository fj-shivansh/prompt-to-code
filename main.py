#!/usr/bin/env python3
"""
Prompt-to-Code Testing System
Main orchestrator for generating and testing code using Gemini API
"""

import sqlite3
import json
import traceback
import time
import subprocess
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import google.generativeai as genai
import os


@dataclass
class TestResult:
    success: bool
    result: Any
    execution_time: float
    error: Optional[str] = None


@dataclass
class CodeGeneration:
    code: str
    explanation: str
    task: str
    requirements: List[str] = None


class DatabaseManager:
    def __init__(self, db_path: str = "historical_data.db"):
        self.db_path = db_path
    
    def get_sample_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get sample data from database for testing, filtering out NULL Adj_Close values"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT Date, Ticker, Adj_Close FROM stock_data WHERE Adj_Close IS NOT NULL LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"Date": row[0], "Ticker": row[1], "Adj_Close": row[2]}
            for row in rows
        ]
    
    def get_all_data(self) -> List[Dict[str, Any]]:
        """Get all data from database, filtering out NULL Adj_Close values"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT Date, Ticker, Adj_Close FROM stock_data WHERE Adj_Close IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"Date": row[0], "Ticker": row[1], "Adj_Close": row[2]}
            for row in rows
        ]


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get from environment
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key not provided")
            genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_code(self, task: str) -> CodeGeneration:
        """Generate code for a given task using Gemini API"""
        prompt = f"""You have access to a SQLite database file called "historical_data.db" with stock market data. The database has a table called "stock_data" with columns:
Date: string, format "YYYY-MM-DD HH:MM:SS"
Ticker: string, stock symbol like "AAPL"
Adj_Close: float, adjusted closing price

IMPORTANT: 
1. Generate COMPLETE, STANDALONE Python code that can be executed directly
2. Your code must load data from the SQLite database "historical_data.db"
3. Include all necessary imports (sqlite3, etc.)
4. Do not use any type hints in function signatures
5. If your solution requires external libraries (like pandas, numpy, matplotlib, etc.), please list all required packages in a "requirements" field. DO NOT include built-in modules like sqlite3, os, sys, time, json, etc.
6. Write production-ready code that handles edge cases and includes proper error handling
7. The code should print the final result at the end and only once.
8. If using pandas DataFrames for output, YOU MUST configure pandas display options to show ALL rows by adding this line BEFORE printing: pd.set_option('display.max_rows', None)
9. Filter out NULL Adj_Close values when querying the database
10. When calculating rolling metrics (moving averages, drawdowns, correlations, etc.), if the task specifies x days, 
interpret it as "today plus the previous (x-1) datapoints." If fewer than (x-1) datapoints exist 
(e.g., at the start of the series), then calculate using all available rows instead of skipping.
11. By default, perform the calculation for all tickers. Only restrict to a specific ticker if it is explicitly mentioned in the task.
12. pd.set_option('display.max_rows', None) IS MANDATORY TO BE USED IF USING PANDAS
EXAMPLE Task: Write code that calculates the highest adjusted close price for AAPL ticker.

EXAMPLE Expected response (JSON):
{{
    "code": "#!/usr/bin/env python3\\nimport sqlite3\\n\\n# Connect to database and load data\\nconn = sqlite3.connect('historical_data.db')\\ncursor = conn.cursor()\\ncursor.execute('SELECT Date, Ticker, Adj_Close FROM stock_data WHERE Adj_Close IS NOT NULL')\\nrows = cursor.fetchall()\\nconn.close()\\n\\n# Convert to list of dictionaries\\ndata = [\\n    {{'Date': row[0], 'Ticker': row[1], 'Adj_Close': row[2]}}\\n    for row in rows\\n]\\n\\ndef highest_adj_close(data, ticker):\\n    prices = [row['Adj_Close'] for row in data if row['Ticker'] == ticker]\\n    if not prices:\\n        return 0.0\\n    return max(prices)\\n\\n# Execute the task\\nresult = highest_adj_close(data, 'AAPL')\\nprint(f'Highest adjusted close price for AAPL: {{result}}')\\nprint(result)",
    "explanation": "The code connects to the SQLite database, loads all stock data, converts it to a list of dictionaries, defines a function to find the highest adjusted close price for a given ticker, executes it for AAPL, and prints the result.",
    "requirements": []
}}

TASK: {task}

Expected response format (JSON):
{{
    "code": "your complete standalone Python code here as a string",
    "explanation": "your step-by-step explanation here as a string",
    "requirements": ["list", "of", "required", "packages", "if", "any"]
}}"""
        response = self.model.generate_content(prompt)
        
        try:
            # Extract JSON from response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            parsed = json.loads(response_text)
            return CodeGeneration(
                code=parsed['code'],
                explanation=parsed['explanation'],
                task=task,
                requirements=parsed.get('requirements', [])
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to parse Gemini response: {e}")


class CodeExecutor:
    def __init__(self):
        self.namespace = {
            "List": list, "Dict": dict, "Any": Any,
            "list": list, "dict": dict, "str": str, "float": float, "int": int,
            "typing": __import__("typing")
        }
        self.generated_file_path = "generated_code.py"
    
    def install_requirements(self, requirements: List[str]) -> bool:
        """Install required packages using pip"""
        if not requirements:
            return True
        
        print(f"Installing requirements: {', '.join(requirements)}")
        try:
            for package in requirements:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install requirements: {e}")
            return False
    
    def write_code_to_file(self, code: str) -> str:
        """Write generated code to separate file"""
        full_code = f'''#!/usr/bin/env python3
"""
Generated Code Execution Module
This file contains AI-generated code for execution.
"""

{code}
'''
        
        with open(self.generated_file_path, "w") as f:
            f.write(full_code)
        
        return self.generated_file_path
    
    def execute_code(self, code: str) -> TestResult:
        """Execute generated code directly as a standalone script"""
        start_time = time.time()
        
        try:
            # Write code to separate file
            code_file = self.write_code_to_file(code)
            
            # Execute the code as a subprocess to capture output
            result = subprocess.run(
                [sys.executable, code_file], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                # Debug logging
                print(f"=== DEBUG: Code Execution Success ===")
                print(f"STDOUT length: {len(result.stdout)}")
                print(f"STDOUT content (first 500 chars): {result.stdout[:500]}")
                print(f"STDOUT content (last 500 chars): {result.stdout[-500:]}")
                
                # Return the full output instead of just the last line
                full_output = result.stdout.strip()
                final_result = full_output if full_output else ""
                
                output_lines = full_output.split('\n') if full_output else []
                print(f"Output lines count: {len(output_lines)}")
                print(f"Final result: {repr(final_result)}")
                print("=====================================")
                
                return TestResult(
                    success=True,
                    result=final_result,
                    execution_time=execution_time
                )
            else:
                return TestResult(
                    success=False,
                    result=None,
                    execution_time=execution_time,
                    error=f"Execution failed with return code {result.returncode}. STDERR: {result.stderr}. STDOUT: {result.stdout}"
                )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return TestResult(
                success=False,
                result=None,
                execution_time=execution_time,
                error="Code execution timed out after 30 seconds"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                success=False,
                result=None,
                execution_time=execution_time,
                error=str(e)
            )


class Analytics:
    @staticmethod
    def analyze_results(generations: List[CodeGeneration], 
                       test_results: List[TestResult],
                       data_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analytics from test results"""
        successful_tests = [r for r in test_results if r.success]
        failed_tests = [r for r in test_results if not r.success]
        
        analytics = {
            "summary": {
                "total_tests": len(test_results),
                "successful": len(successful_tests),
                "failed": len(failed_tests),
                "success_rate": len(successful_tests) / len(test_results) if test_results else 0
            },
            "performance": {
                "avg_execution_time": sum(r.execution_time for r in successful_tests) / len(successful_tests) if successful_tests else 0,
                "fastest_execution": min(r.execution_time for r in successful_tests) if successful_tests else 0,
                "slowest_execution": max(r.execution_time for r in successful_tests) if successful_tests else 0
            },
            "errors": [r.error for r in failed_tests],
            "data_stats": data_stats,
            "results": [r.result for r in successful_tests]
        }
        
        return analytics


class PromptToCodeSystem:
    def __init__(self, api_key: Optional[str] = None):
        self.db_manager = DatabaseManager()
        self.gemini_client = GeminiClient(api_key)
        self.code_executor = CodeExecutor()
        self.generations = []
        self.test_results = []
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """Process a complete task: generate code, test it, and return analytics"""
        
        print(f"Processing task: {task}")
        
        # Generate code
        print("Generating code with Gemini API...")
        try:
            generation = self.gemini_client.generate_code(task)
            self.generations.append(generation)
            print(f"Generated code:\n{generation.code}")
            print(f"Explanation: {generation.explanation}")
            if generation.requirements:
                print(f"Requirements: {', '.join(generation.requirements)}")
                # Install requirements
                if not self.code_executor.install_requirements(generation.requirements):
                    return {"error": "Failed to install required packages"}
        except Exception as e:
            return {"error": f"Code generation failed: {str(e)}"}
        
        # Load data
        print("Loading test data...")
        data = self.db_manager.get_all_data()
        data_stats = {
            "total_records": len(data),
            "unique_tickers": len(set(d["Ticker"] for d in data)),
            "date_range": {
                "start": min(d["Date"] for d in data),
                "end": max(d["Date"] for d in data)
            }
        }
        
        # Execute and test code
        print("Executing generated code...")
        test_result = self.code_executor.execute_code(generation.code)
        self.test_results.append(test_result)
        
        if test_result.success:
            print(f"Execution successful! Result: {test_result.result}")
            print(f"Execution time: {test_result.execution_time:.4f} seconds")
        else:
            print(f"Execution failed: {test_result.error}")
        
        # Generate analytics
        analytics = Analytics.analyze_results([generation], [test_result], data_stats)
        
        # Add additional analytics for requirements and execution environment
        analytics["generation_info"] = {
            "code_length": len(generation.code),
            "has_requirements": bool(generation.requirements),
            "requirements_count": len(generation.requirements) if generation.requirements else 0,
            "requirements_list": generation.requirements if generation.requirements else [],
            "executed_in_separate_file": True
        }
        
        return {
            "generation": generation,
            "test_result": test_result,
            "analytics": analytics
        }
    
    def run_interactive_mode(self):
        """Run interactive mode for testing tasks"""
        print("=== Prompt-to-Code Testing System ===")
        print("Enter tasks to generate and test code against stock market data")
        print("Type 'quit' to exit")
        
        while True:
            task = input("\nEnter your task: ").strip()
            if task.lower() in ['quit', 'exit', 'q']:
                break
            
            if not task:
                continue
            
            result = self.process_task(task)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                continue
            
            print("\n" + "="*50)
            print("RESULTS:")
            print("="*50)
            
            analytics = result["analytics"]
            gen_info = analytics["generation_info"]
            
            print(f"Success Rate: {analytics['summary']['success_rate']:.1%}")
            print(f"Execution Time: {analytics['performance']['avg_execution_time']:.4f}s")
            print(f"Code Length: {gen_info['code_length']} characters")
            
            if gen_info['has_requirements']:
                print(f"Requirements: {', '.join(gen_info['requirements_list'])}")
            
            print(f"Executed in separate file: {gen_info['executed_in_separate_file']}")
            
            if result["test_result"].success:
                print(f"Result: {result['test_result'].result}")
            else:
                print(f"Error: {result['test_result'].error}")


if __name__ == "__main__":
    # Example usage
    try:
        system = PromptToCodeSystem()
        system.run_interactive_mode()
    except ValueError as e:
        print(f"Setup error: {e}")
        print("Please set GEMINI_API_KEY environment variable or provide API key")