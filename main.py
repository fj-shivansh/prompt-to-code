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
        self.api_keys = []
        self.current_key_index = 0
        
        if api_key:
            self.api_keys = [api_key]
        else:
            # Try to get from environment - support both single and multiple keys
            single_key = os.getenv('GEMINI_API_KEY')
            multiple_keys = os.getenv('GEMINI_API_KEYS')
            
            if multiple_keys:
                # Parse comma-separated keys, strip whitespace
                self.api_keys = [key.strip() for key in multiple_keys.split(',') if key.strip()]
            elif single_key:
                self.api_keys = [single_key]
            else:
                raise ValueError("No Gemini API keys provided. Set GEMINI_API_KEY or GEMINI_API_KEYS environment variable")
        
        if not self.api_keys:
            raise ValueError("No valid API keys found")
        
        # Configure with the first API key
        genai.configure(api_key=self.api_keys[0])
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        print(f"Initialized GeminiClient with {len(self.api_keys)} API key(s)")
    
    def _rotate_api_key(self):
        """Rotate to the next API key in round-robin fashion"""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            current_key = self.api_keys[self.current_key_index]
            genai.configure(api_key=current_key)
            print(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def _get_current_api_key_info(self) -> str:
        """Get info about current API key for debugging"""
        return f"Using API key {self.current_key_index + 1}/{len(self.api_keys)}"
    
    def generate_code(self, task: str, error_context: Optional[str] = None, failed_code: Optional[str] = None) -> CodeGeneration:
        """Generate code for a given task using Gemini API with round-robin key rotation"""
        
        print(f"Generating code - {self._get_current_api_key_info()}")
        
        # Add error context if this is a retry attempt
        error_section = ""
        if error_context and failed_code:
            error_section = f"""
ERROR CONTEXT FOR CODE FIXING:
ORIGINAL TASK: {task}

The following code failed with this error:
ERROR: {error_context}

FAILED CODE:
{failed_code}

Please analyze the error and fix the code. Common issues to check:
1. **INDENTATION ERRORS**: Check for extra spaces, mixed tabs/spaces, inconsistent indentation
2. **PANDAS GROUPBY ISSUES**: When using groupby.apply(), columns created in one apply() may not be available in subsequent apply() calls on the same grouped object
3. Variable naming conflicts or undefined variables
4. Data type mismatches or conversion errors
5. Missing imports or incorrect library usage
6. Database connection or query issues
7. File path or permission problems
8. Pandas operations on empty DataFrames
9. Syntax errors like missing colons, parentheses, or quotes
10. **DATAFRAME COLUMN ACCESS**: Make sure columns exist before accessing them, especially after groupby operations

IMPORTANT: You must complete the ORIGINAL TASK above. Focus on the task requirements while fixing the technical error.

"""

        prompt = f"""{error_section}You have access to a SQLite database file called "historical_data.db" with stock market data. The database has a table called "stock_data" with columns:
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
7. **CSV OUTPUT REQUIREMENT**: The final result MUST be saved as a CSV file named 'output.csv'. Use pandas to_csv() method or manual CSV writing. DO NOT print tabulate output.
8. MANDATORY: For DataFrame results, save to CSV like this: df.to_csv('output.csv', index=False). For non-DataFrame results, convert to DataFrame first then save as CSV.
9. If using pandas DataFrames for processing, YOU MUST configure pandas display options: pd.set_option('display.max_rows', None)
10. Filter out NULL Adj_Close values when querying the database
11. When calculating rolling metrics (moving averages, drawdowns, correlations, etc.), if the task specifies x days, 
interpret it as "today plus the previous (x-1) datapoints." If fewer than (x-1) datapoints exist 
(e.g., at the start of the series), then calculate using all available rows instead of skipping.
12. By default, perform the calculation for all tickers. Only restrict to a specific ticker if it is explicitly mentioned in the task.
13. pd.set_option('display.max_rows', None) IS MANDATORY TO BE USED IF USING PANDAS
14. **VARIABLE DEFINITIONS**: Always define all variables before using them. If a function needs parameters like 'window', make sure to define them or pass them as arguments. Avoid using undefined variables.
15. **INDENTATION**: Use consistent 4-space indentation throughout. NO TABS. ALL lines at the same level must have identical indentation. Check for extra spaces before code lines.
16. **DATE SORTING - MANDATORY**: When working with date-based data, you MUST ALWAYS sort the FINAL OUTPUT by Date in descending order (latest date first) unless explicitly specified otherwise. This applies to:
   - SQL queries: Use ORDER BY Date DESC 
   - Pandas DataFrames: Use df.sort_values('Date', ascending=False)
   - Final CSV output: ALWAYS sort by Date DESC before saving to CSV
   - Even if the task asks for "top N" or "highest/lowest" values, the final result should still be sorted by Date DESC

EXAMPLE Expected response (JSON):
{{
    "code": "#!/usr/bin/env python3\\nimport sqlite3\\nimport pandas as pd\\n\\n# Set pandas display options\\npd.set_option('display.max_rows', None)\\n\\n# Connect to database and load data (sorted by date descending - latest first)\\nconn = sqlite3.connect('historical_data.db')\\ncursor = conn.cursor()\\ncursor.execute('SELECT Date, Ticker, Adj_Close FROM stock_data WHERE Adj_Close IS NOT NULL ORDER BY Date DESC')\\nrows = cursor.fetchall()\\nconn.close()\\n\\n# Convert to DataFrame\\ndf = pd.DataFrame(rows, columns=['Date', 'Ticker', 'Adj_Close'])\\n\\n# Filter for AAPL and find highest adjusted close price\\naapl_data = df[df['Ticker'] == 'AAPL']\\nhighest_price = aapl_data['Adj_Close'].max() if not aapl_data.empty else 0.0\\ntotal_records = len(aapl_data)\\n\\n# Create result DataFrame\\nresult_data = [\\n    ['Ticker', 'AAPL'],\\n    ['Highest Adjusted Close Price', f'${{highest_price:.2f}}'],\\n    ['Total Records Analyzed', total_records]\\n]\\n\\nresult_df = pd.DataFrame(result_data, columns=['Metric', 'Value'])\\nresult_df.to_csv('output.csv', index=False)\\nprint('Results saved to output.csv')",
    "explanation": "The code connects to the SQLite database, loads all stock data sorted by date (latest first), calculates the highest adjusted close price for AAPL, and saves the result as a CSV file with proper date sorting.",
    "requirements": ["pandas"]
}}


TASK: {task}

Expected response format (JSON):
{{
    "code": "your complete standalone Python code here as a string",
    "explanation": "your step-by-step explanation here as a string",
    "requirements": ["list", "of", "required", "packages", "if", "any"]
}}"""
        # Try generating with current API key, with retry logic for rate limiting
        max_key_attempts = len(self.api_keys)
        last_exception = None
        
        for attempt in range(max_key_attempts):
            try:
                response = self.model.generate_content(prompt)
                
                # Extract JSON from response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3].strip()
                
                parsed = json.loads(response_text)
                
                # Successful generation - rotate to next key for next request
                self._rotate_api_key()
                
                return CodeGeneration(
                    code=parsed['code'],
                    explanation=parsed['explanation'],
                    task=task,
                    requirements=parsed.get('requirements', [])
                )
                
            except Exception as e:
                last_exception = e
                error_message = str(e).lower()
                
                # Check if it's a rate limit or quota error
                if any(keyword in error_message for keyword in ['quota', 'rate limit', 'resource exhausted', '429']):
                    print(f"Rate limit hit on API key {self.current_key_index + 1}, trying next key...")
                    self._rotate_api_key()
                    continue
                
                # For JSON parsing errors, don't rotate key - it's not a quota issue
                if isinstance(e, (json.JSONDecodeError, KeyError)):
                    raise ValueError(f"Failed to parse Gemini response: {e}")
                
                # For other errors, try next key
                print(f"Error with API key {self.current_key_index + 1}: {e}")
                self._rotate_api_key()
                continue
        
        # All keys exhausted
        raise ValueError(f"All API keys exhausted. Last error: {last_exception}")


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
                
                # Check if output.csv file was created
                csv_file_path = "output.csv"
                csv_exists = os.path.exists(csv_file_path)
                print(f"CSV file exists: {csv_exists}")
                
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
    
    def process_task(self, task: str, max_retries: int = 5) -> Dict[str, Any]:
        """Process a complete task with retry mechanism for failed code execution"""
        
        print(f"Processing task: {task}")
        
        # Load data for analytics
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
        
        generation = None
        test_result = None
        attempt = 0
        error_context = None
        failed_code = None
        
        while attempt < max_retries:
            attempt += 1
            print(f"\n=== Attempt {attempt}/{max_retries} ===")
            
            # Generate code (with error context if retry)
            print("Generating code with Gemini API...")
            try:
                if attempt == 1:
                    generation = self.gemini_client.generate_code(task)
                else:
                    print(f"Retrying with error context: {error_context[:200]}...")
                    generation = self.gemini_client.generate_code(task, error_context, failed_code)
                
                self.generations.append(generation)
                print(f"Generated code:\n{generation.code}")
                print(f"Explanation: {generation.explanation}")
                
                if generation.requirements:
                    print(f"Requirements: {', '.join(generation.requirements)}")
                    # Install requirements
                    if not self.code_executor.install_requirements(generation.requirements):
                        error_context = "Failed to install required packages"
                        failed_code = generation.code
                        continue
                        
            except Exception as e:
                error_context = f"Code generation failed: {str(e)}"
                if attempt == max_retries:
                    return {"error": error_context}
                continue
            
            # Execute and test code
            print("Executing generated code...")
            test_result = self.code_executor.execute_code(generation.code)
            self.test_results.append(test_result)
            
            if test_result.success:
                print(f"Execution successful! Result: {test_result.result}")
                print(f"Execution time: {test_result.execution_time:.4f} seconds")
                break  # Success - exit retry loop
            else:
                print(f"Execution failed: {test_result.error}")
                error_context = test_result.error
                failed_code = generation.code
                
                if attempt == max_retries:
                    print(f"Max retries ({max_retries}) reached. Final error: {test_result.error}")
                    break
                else:
                    print(f"Retrying... ({max_retries - attempt} attempts remaining)")
        
        # Generate analytics
        analytics = Analytics.analyze_results([generation], [test_result], data_stats)
        
        # Add additional analytics for requirements and execution environment
        analytics["generation_info"] = {
            "code_length": len(generation.code),
            "has_requirements": bool(generation.requirements),
            "requirements_count": len(generation.requirements) if generation.requirements else 0,
            "requirements_list": generation.requirements if generation.requirements else [],
            "executed_in_separate_file": True,
            "retry_attempts": attempt,
            "max_retries": max_retries
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
        print("Please set GEMINI_API_KEY (single key) or GEMINI_API_KEYS (comma-separated multiple keys) environment variable")