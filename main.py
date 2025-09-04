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
    def __init__(self, db_path: str = "historical_data_with_gains.db"):
        self.db_path = db_path
    
    def get_sample_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get sample data from database for testing, filtering out NULL Adj_Close values"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"Date": row[0], "Ticker": row[1], "Adj_Close": row[2], "Daily_Gain_Pct": row[3], "Forward_Gain_Pct": row[4]}
            for row in rows
        ]
    
    def get_all_data(self) -> List[Dict[str, Any]]:
        """Get all data from database, filtering out NULL Adj_Close values"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"Date": row[0], "Ticker": row[1], "Adj_Close": row[2], "Daily_Gain_Pct": row[3], "Forward_Gain_Pct": row[4]}
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

        prompt = f"""{error_section}🚨 CRITICAL REQUIREMENT: ALL 5 DATABASE COLUMNS MUST BE IN OUTPUT.CSV 🚨
🚨 COLUMN NAMES MUST BE EXACTLY: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct 🚨

⚠️ DO NOT USE: Close, Symbol, daily_gain_pct, or any other variations
✅ MUST USE: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct

You have access to a SQLite database file called "historical_data_with_gains.db" with stock market data. The database has a table called "stock_data" with ONLY these 5 columns:

**AVAILABLE COLUMNS (ONLY THESE 5):**
- Date: string, format "YYYY-MM-DD HH:MM:SS"
- Ticker: string, stock symbol like "AAPL"  
- Adj_Close: float, adjusted closing price
- Daily_Gain_Pct: float, daily percentage gain/loss (percentage change from previous day)
- Forward_Gain_Pct: float, forward percentage gain/loss (percentage change to next day)

**CRITICAL**: The database contains these 5 columns: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct. You can ONLY use these 5 columns. Using any other column will result in a "no such column" error.

IMPORTANT: 
1. **DATABASE COLUMNS RESTRICTION**: You can ONLY use these 5 columns: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct. DO NOT reference any other columns (Open, High, Low, Close, Volume, etc.) as they do not exist and will cause "no such column" errors.
2. Generate COMPLETE, STANDALONE Python code that can be executed directly
3. Your code must load data from the SQLite database "historical_data_with_gains.db"
4. Include all necessary imports (sqlite3, etc.)
5. Do not use any type hints in function signatures
6. If your solution requires external libraries (like pandas, numpy, matplotlib, etc.), please list all required packages in a "requirements" field. DO NOT include built-in modules like sqlite3, os, sys, time, json, etc.
7. Write production-ready code that handles edge cases and includes proper error handling
8. 🚨 **NON-NEGOTIABLE CSV COLUMN REQUIREMENT**: The final output.csv file MUST ALWAYS contain these exact 5 columns from the database: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct. You may add additional calculated columns, but these 5 MUST ALWAYS be present. Failure to include all 5 database columns will result in task failure.
8a. 🚨 **EXACT COLUMN NAME ENFORCEMENT**: Use EXACTLY these column names (case-sensitive): Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct. DO NOT use Close, Symbol, daily_gain_pct, or any other variations.
9. **CSV OUTPUT REQUIREMENT**: The final result MUST be saved as a CSV file named 'output.csv'. Use pandas to_csv() method or manual CSV writing. DO NOT print tabulate output.
10. MANDATORY: For DataFrame results, save to CSV like this: df.to_csv('output.csv', index=False). For non-DataFrame results, convert to DataFrame first then save as CSV.
11. If using pandas DataFrames for processing, YOU MUST configure pandas display options: pd.set_option('display.max_rows', None)
12. Filter out NULL Adj_Close values when querying the database
13. **COLUMN PRESERVATION REQUIREMENT**: When saving to output.csv, you MUST preserve all original database columns (Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct) in the final result alongside any new calculated columns. Never drop or exclude the original database columns.
14. When calculating rolling metrics (moving averages, drawdowns, correlations, etc.), if the task specifies x days, 
interpret it as "today plus the previous (x-1) datapoints." If fewer than (x-1) datapoints exist 
(e.g., at the start of the series), then calculate using all available rows instead of skipping.
15. By default, perform the calculation for all tickers. Only restrict to a specific ticker if it is explicitly mentioned in the task.
16. pd.set_option('display.max_rows', None) IS MANDATORY TO BE USED IF USING PANDAS
17. **VARIABLE DEFINITIONS**: Always define all variables before using them. If a function needs parameters like 'window', make sure to define them or pass them as arguments. Avoid using undefined variables.
18. **INDENTATION**: Use consistent 4-space indentation throughout. NO TABS. ALL lines at the same level must have identical indentation. Check for extra spaces before code lines.
19. **DATE SORTING - MANDATORY**: When working with date-based data, you MUST ALWAYS sort the FINAL OUTPUT by Date in descending order (latest date first) unless explicitly specified otherwise. This applies to:
   - SQL queries: Use ORDER BY Date DESC 
   - Pandas DataFrames: Use df.sort_values('Date', ascending=False)
   - Final CSV output: ALWAYS sort by Date DESC before saving to CSV
   - Even if the task asks for "top N" or "highest/lowest" values, the final result should still be sorted by Date DESC

🚨 MANDATORY TEMPLATE - Your final DataFrame BEFORE saving to CSV MUST follow this EXACT pattern:
```
# Final DataFrame MUST ALWAYS have these EXACT 5 column names first, in this EXACT order:
final_df = result_df[['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct'] + [list_of_your_calculated_columns]]
final_df = final_df.sort_values('Date', ascending=False)  # Date DESC
final_df.to_csv('output.csv', index=False)
```

🚨 CRITICAL: The first 5 columns in output.csv MUST be EXACTLY: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct (in this exact order and spelling)

⚠️ WRONG COLUMN NAMES THAT WILL CAUSE FAILURE:
❌ "Close" (should be "Adj_Close")  
❌ "Symbol" (should be "Ticker")
❌ "daily_gain_pct" (should be "Daily_Gain_Pct") 
❌ Any other variation of these names

✅ CORRECT COLUMN NAMES (copy these exactly):
Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct

EXAMPLE Expected response (JSON):
{{
    "code": "#!/usr/bin/env python3\\nimport sqlite3\\nimport pandas as pd\\n\\n# Set pandas display options\\npd.set_option('display.max_rows', None)\\n\\n# Connect to database and load ALL columns with EXACT names\\nconn = sqlite3.connect('historical_data_with_gains.db')\\ncursor = conn.cursor()\\ncursor.execute('SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL ORDER BY Date DESC')\\nrows = cursor.fetchall()\\nconn.close()\\n\\n# Convert to DataFrame with EXACT column names (NOT Close, NOT Symbol)\\ndf = pd.DataFrame(rows, columns=['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct'])\\n\\n# Calculate 10-day and 5-day moving averages for all tickers\\ndf = df.sort_values(['Ticker', 'Date'])\\ndf['10_Day_MA'] = df.groupby('Ticker')['Adj_Close'].rolling(window=10, min_periods=1).mean().values\\ndf['5_Day_MA'] = df.groupby('Ticker')['Adj_Close'].rolling(window=5, min_periods=1).mean().values\\n\\n# CRITICAL: Final DataFrame MUST have EXACT column names in EXACT order\\nfinal_df = df[['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct', '10_Day_MA', '5_Day_MA']].copy()\\n\\n# Sort by Date DESC (latest first) and save\\nfinal_df = final_df.sort_values('Date', ascending=False)\\nfinal_df.to_csv('output.csv', index=False)\\nprint('Results saved to output.csv with EXACT database column names')",
    "explanation": "The code connects to the SQLite database and loads ALL 5 database columns with EXACT names (Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct), calculates moving averages, and saves the complete result to output.csv. CRITICAL: Uses exact database column names - 'Adj_Close' NOT 'Close', 'Ticker' NOT 'Symbol'. Final CSV has exact column order required.",
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
                
                # Capture token usage if available
                tokens = None
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    tokens = {
                        'input_tokens': response.usage_metadata.prompt_token_count,
                        'output_tokens': response.usage_metadata.candidates_token_count,
                        'total_tokens': response.usage_metadata.total_token_count
                    }
                
                # Extract JSON from response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3].strip()
                
                parsed = json.loads(response_text)
                
                # Successful generation - rotate to next key for next request
                self._rotate_api_key()
                
                generation = CodeGeneration(
                    code=parsed['code'],
                    explanation=parsed['explanation'],
                    task=task,
                    requirements=parsed.get('requirements', [])
                )
                # Add token info to the generation object
                generation.tokens = tokens
                return generation
                
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
        self.current_process = None
        self.process_callback = None
    
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
            self.current_process = subprocess.Popen(
                [sys.executable, code_file], 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Call the process callback if set (for tracking in backend)
            if self.process_callback:
                self.process_callback(self.current_process)
            
            try:
                stdout, stderr = self.current_process.communicate(timeout=120)
                result_returncode = self.current_process.returncode
            except subprocess.TimeoutExpired:
                self.current_process.kill()
                stdout, stderr = self.current_process.communicate()
                raise
            
            execution_time = time.time() - start_time
            
            if result_returncode == 0:
                # Debug logging
                print(f"=== DEBUG: Code Execution Success ===")
                print(f"STDOUT length: {len(stdout)}")
                print(f"STDOUT content (first 500 chars): {stdout[:500]}")
                print(f"STDOUT content (last 500 chars): {stdout[-500:]}")
                
                # Check if output.csv file was created
                csv_file_path = "output.csv"
                csv_exists = os.path.exists(csv_file_path)
                print(f"CSV file exists: {csv_exists}")
                
                # Return the full output instead of just the last line
                full_output = stdout.strip()
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
                    error=f"Execution failed with return code {result_returncode}. STDERR: {stderr}. STDOUT: {stdout}"
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
    
    def process_task_streaming(self, task: str, max_retries: int = 3, progress_callback=None):
        """Process task with streaming progress updates"""
        
        if progress_callback:
            progress_callback(f"data: {json.dumps({'type': 'task_start', 'message': 'Starting task processing...'})}\n\n")
        
        print(f"Processing task: {task}")
        
        # Load data for analytics
        if progress_callback:
            progress_callback(f"data: {json.dumps({'type': 'loading_data', 'message': 'Loading database...'})}\n\n")
        
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
            
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'retry', 'message': f'Execution Retry {attempt}', 'retry': attempt, 'max_retries': max_retries})}\n\n")
            
            print(f"\n=== Attempt {attempt}/{max_retries} ===")
            
            # Generate code (with error context if retry)
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'generating_code', 'message': f'Generating code (retry {attempt})...'})}\n\n")
            
            print("Generating code with Gemini API...")
            try:
                if attempt == 1:
                    generation = self.gemini_client.generate_code(task)
                else:
                    print(f"Retrying with error context: {error_context[:200]}...")
                    generation = self.gemini_client.generate_code(task, error_context, failed_code)
                
                self.generations.append(generation)
                
                if generation.requirements:
                    print(f"Requirements: {', '.join(generation.requirements)}")
                    if progress_callback:
                        progress_callback(f"data: {json.dumps({'type': 'installing_deps', 'message': 'Installing dependencies...'})}\n\n")
                    
                    # Install requirements
                    if not self.code_executor.install_requirements(generation.requirements):
                        error_context = "Failed to install required packages"
                        failed_code = generation.code
                        if progress_callback:
                            progress_callback(f"data: {json.dumps({'type': 'retry_failed', 'message': f'Retry {attempt} failed: dependency installation', 'retry': attempt})}\n\n")
                        continue
                        
            except Exception as e:
                error_context = f"Code generation failed: {str(e)}"
                if progress_callback:
                    progress_callback(f"data: {json.dumps({'type': 'retry_failed', 'message': f'Retry {attempt} failed: code generation error', 'retry': attempt})}\n\n")
                if attempt == max_retries:
                    return {"error": error_context}
                continue
            
            # Execute and test code
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'executing', 'message': f'Executing code (retry {attempt})...'})}\n\n")
            
            print("Executing generated code...")
            test_result = self.code_executor.execute_code(generation.code)
            self.test_results.append(test_result)
            
            if test_result.success:
                if progress_callback:
                    progress_callback(f"data: {json.dumps({'type': 'retry_success', 'message': f'Retry {attempt} succeeded!', 'retry': attempt})}\n\n")
                
                print(f"Execution successful! Result: {test_result.result}")
                print(f"Execution time: {test_result.execution_time:.4f} seconds")
                break  # Success - exit retry loop
            else:
                if progress_callback:
                    progress_callback(f"data: {json.dumps({'type': 'retry_failed', 'message': f'Retry {attempt} failed: execution error', 'retry': attempt})}\n\n")
                
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
        
        # Add generation info to analytics
        analytics["generation_info"] = {
            "retry_attempts": attempt,
            "max_retries": max_retries,
            "code_length": len(generation.code) if generation else 0,
            "requirements_count": len(generation.requirements) if generation and generation.requirements else 0,
            "tokens": generation.tokens if generation and hasattr(generation, 'tokens') and generation.tokens else None
        }
        
        return {
            "generation": generation,
            "test_result": test_result,
            "analytics": analytics
        }

    def process_task(self, task: str, max_retries: int = 3) -> Dict[str, Any]:
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
            "max_retries": max_retries,
            "tokens": generation.tokens if hasattr(generation, 'tokens') and generation.tokens else None
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