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
    def __init__(self, db_path: str = "historical_data_500_tickers_with_gains.db"):
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
    
    def get_filtered_data(self, ticker_list=None, start_date=None, end_date=None) -> List[Dict[str, Any]]:
        """Get filtered data based on tickers and date range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct 
            FROM stock_data 
            WHERE Adj_Close IS NOT NULL 
            AND Daily_Gain_Pct IS NOT NULL 
            AND Forward_Gain_Pct IS NOT NULL
        """
        params = []
        
        if ticker_list:
            placeholders = ','.join(['?' for _ in ticker_list])
            query += f" AND Ticker IN ({placeholders})"
            params.extend(ticker_list)
        
        if start_date:
            query += " AND Date >= ?"
            params.append(start_date)
            
        if end_date:
            query += " AND Date <= ?"
            params.append(end_date)
        
        query += " ORDER BY Ticker, Date"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"Date": row[0], "Ticker": row[1], "Adj_Close": row[2], 
             "Daily_Gain_Pct": row[3], "Forward_Gain_Pct": row[4]}
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
        self.model = genai.GenerativeModel('gemini-2.5-flash')
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
    
    def generate_code(self, task: str, error_context: Optional[str] = None, failed_code: Optional[str] = None, output_file: str = "output.csv", ticker_filters=None, date_filters=None) -> CodeGeneration:
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

        # Build filter instruction
        filter_instructions = ""
        
        if ticker_filters or date_filters:
            filter_instructions = "\n🎯 DATA FILTERING REQUIREMENTS:\n"
            
            # Handle pre-selected tickers
            selected_tickers = None
            if ticker_filters and ticker_filters.get('selected_tickers'):
                selected_tickers = ticker_filters['selected_tickers']
                filter_instructions += f"📊 TICKER FILTER: Use ONLY these {len(selected_tickers)} pre-selected tickers: {', '.join(selected_tickers)}\n"
                filter_instructions += f"🚨 DO NOT use random selection - use these EXACT tickers\n"
            elif ticker_filters and ticker_filters.get('ticker_count') == 'all':
                filter_instructions += f"📊 TICKER FILTER: Use ALL available tickers\n"
            
            if date_filters:
                start_date = date_filters.get('start_date')
                end_date = date_filters.get('end_date')
                if start_date and end_date:
                    filter_instructions += f"📅 DATE FILTER: Use ONLY data from {start_date} to {end_date}\n"
                    filter_instructions += f"🚨 Add WHERE Date >= '{start_date}' AND Date <= '{end_date}' to your database query\n"
            
            filter_instructions += "\n"

            # Add database query template with filters
            query_template = """
MANDATORY DATABASE QUERY TEMPLATE WITH FILTERS:
```python
conn = sqlite3.connect('historical_data_500_tickers_with_gains.db')
cursor = conn.cursor()

# Get data for specific tickers"""
            
            if selected_tickers:
                query_template += f"""
# Use pre-selected tickers
selected_tickers = {selected_tickers}
placeholders = ','.join(['?' for _ in selected_tickers])
query = f"SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL AND Ticker IN ({{placeholders}})"
params = selected_tickers
"""
            else:
                query_template += f"""
# Get all tickers
query = "SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL"
params = []
"""
            
            if date_filters:
                start_date = date_filters.get('start_date')
                end_date = date_filters.get('end_date')
                if start_date:
                    query_template += f"""
# Add date filters
query += " AND Date >= ?"
params.append('{start_date}')
"""
                if end_date:
                    query_template += f"""
query += " AND Date <= ?"
params.append('{end_date}')
"""
            
            query_template += """
query += " ORDER BY Ticker, Date"
cursor.execute(query, params)

# Convert to DataFrame and ensure proper sorting
rows = cursor.fetchall()
conn.close()
df = pd.DataFrame(rows, columns=['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct'])

# MANDATORY: Sort by Ticker and Date for processing
df = df.sort_values(['Ticker', 'Date'], ascending=[True, True])

# ... perform your calculations here ...

# MANDATORY: Final sort by Date DESC before saving
final_df = final_df.sort_values('Date', ascending=False)
final_df.to_csv('{output_file}', index=False)
```
🚨 YOU MUST USE THIS EXACT QUERY PATTERN AND SORTING 🚨
"""
            filter_instructions += query_template

        prompt = f"""{error_section}{filter_instructions}🚨🚨🚨 CRITICAL REQUIREMENT: ALL 5 DATABASE COLUMNS MUST BE IN OUTPUT.CSV 🚨🚨🚨
🚨🚨🚨 COLUMN NAMES MUST BE EXACTLY: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct 🚨🚨🚨
🚨🚨🚨 THESE 5 COLUMNS MUST ALWAYS BE THE FIRST 5 COLUMNS IN YOUR OUTPUT CSV 🚨🚨🚨
🚨🚨🚨 ANY ADDITIONAL COLUMNS MUST COME AFTER THESE 5 MANDATORY COLUMNS 🚨🚨🚨

⚠️ DO NOT USE: Close, Symbol, daily_gain_pct, or any other variations
✅ MUST USE: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct
🚨 FAILURE TO INCLUDE THESE 5 COLUMNS WILL BREAK DOWNSTREAM NAV CALCULATIONS 🚨

🚨🚨🚨 ABSOLUTELY FORBIDDEN: DO NOT RENAME DATABASE COLUMNS 🚨🚨🚨
🚨🚨🚨 DO NOT USE df.rename() TO CHANGE Daily_Gain_Pct OR Forward_Gain_Pct 🚨🚨🚨
🚨🚨🚨 KEEP ORIGINAL COLUMN NAMES: Daily_Gain_Pct, Forward_Gain_Pct 🚨🚨🚨
🚨🚨🚨 IF YOU RENAME COLUMNS, THE SYSTEM WILL BREAK 🚨🚨🚨

You have access to a SQLite database file called "historical_data_500_tickers_with_gains.db" with stock market data. The database has a table called "stock_data" with ONLY these 5 columns:

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
3. Your code must load data from the SQLite database "historical_data_500_tickers_with_gains.db"
4. Include all necessary imports (sqlite3, etc.)
5. Do not use any type hints in function signatures
6. If your solution requires external libraries (like pandas, numpy, matplotlib, etc.), please list all required packages in a "requirements" field. DO NOT include built-in modules like sqlite3, os, sys, time, json, etc.
7. Write production-ready code that handles edge cases and includes proper error handling
8. 🚨🚨🚨 **NON-NEGOTIABLE CSV COLUMN REQUIREMENT**: The final output.csv file MUST ALWAYS contain these exact 5 columns from the database AS THE FIRST 5 COLUMNS: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct. You may add additional calculated columns AFTER these 5, but these 5 MUST ALWAYS be present AS THE FIRST 5 COLUMNS. Failure to include all 5 database columns AS THE FIRST 5 COLUMNS will result in task failure and break NAV calculations.
8a. 🚨🚨🚨 **EXACT COLUMN NAME ENFORCEMENT**: Use EXACTLY these column names (case-sensitive): Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct. DO NOT use Close, Symbol, daily_gain_pct, or any other variations.
8b. 🚨🚨🚨 **COLUMN ORDER ENFORCEMENT**: The CSV MUST start with these 5 columns in this EXACT order: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct. Any additional calculated columns must come AFTER these 5.
9. **CSV OUTPUT REQUIREMENT**: The final result MUST be saved as a CSV file named '{output_file}'. Use pandas to_csv() method or manual CSV writing. DO NOT print tabulate output.
10. MANDATORY: For DataFrame results, save to CSV like this: df.to_csv('{output_file}', index=False). For non-DataFrame results, convert to DataFrame first then save as CSV.
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
19. 🚨🚨🚨 **MANDATORY SORTING REQUIREMENT**: The final output CSV MUST be sorted by Date in DESCENDING order (latest dates first). This is CRITICAL for consistent output across all LLMs.
20. **SORTING ENFORCEMENT**: Before saving the final CSV with df.to_csv(), you MUST include this exact line: df = df.sort_values('Date', ascending=False). Both LLMs must produce identically sorted results.
21. **DATE SORTING DETAILS**: When working with date-based data, the final output must be sorted by Date DESC (latest date first). Use df.sort_values('Date', ascending=False) right before df.to_csv().
   - Even if the task asks for "top N" or "highest/lowest" values, the final result should still be sorted by Date DESC
🚨🚨🚨 MANDATORY DATA SORTING RULE - MUST BE DONE BEFORE ANY OPERATIONS 🚨🚨🚨
🚨🚨🚨 ALWAYS SORT BY TICKER AND DATE BEFORE PERFORMING ANY CALCULATIONS 🚨🚨🚨
```python
# MANDATORY: Sort data by Ticker and Date BEFORE any operations
df = df.sort_values(['Ticker', 'Date'], ascending=[True, True])
```
🚨🚨🚨 THIS SORTING MUST BE APPLIED IMMEDIATELY AFTER LOADING DATA FROM DATABASE 🚨🚨🚨
🚨🚨🚨 ALL ROLLING WINDOWS, GROUPBY OPERATIONS DEPEND ON THIS SORTING 🚨🚨🚨

🚨🚨🚨 MANDATORY TEMPLATE - Your final DataFrame BEFORE saving to CSV MUST follow this EXACT pattern 🚨🚨🚨
🚨🚨🚨 DO NOT DEVIATE FROM THIS TEMPLATE - IT IS REQUIRED FOR NAV CALCULATIONS 🚨🚨🚨
```python
# Step 1: Get all database columns first
db_columns = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']

# Step 2: Get your calculated columns (replace with your actual calculated column names)
calculated_columns = ['Your_Calculated_Column1', 'Your_Calculated_Column2']  # Replace with actual names

# Step 3: MANDATORY - Combine db_columns FIRST, then calculated columns
final_df = result_df[db_columns + calculated_columns]

# Step 4: Sort by Date DESC
final_df = final_df.sort_values('Date', ascending=False)

# Step 5: Save to CSV
final_df.to_csv('{output_file}', index=False)
```

🚨🚨🚨 CRITICAL: The first 5 columns in output.csv MUST be EXACTLY: Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct (in this exact order and spelling) 🚨🚨🚨
🚨🚨🚨 ANY DEVIATION FROM THIS COLUMN ORDER WILL BREAK NAV CALCULATIONS 🚨🚨🚨

⚠️ WRONG COLUMN NAMES THAT WILL CAUSE FAILURE:
❌ "Close" (should be "Adj_Close")  
❌ "Symbol" (should be "Ticker")
❌ "daily_gain_pct" (should be "Daily_Gain_Pct") 
❌ Any other variation of these names

✅ CORRECT COLUMN NAMES (copy these exactly):
Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct

EXAMPLE Expected response (JSON):
{{
    "code": "#!/usr/bin/env python3\\nimport sqlite3\\nimport pandas as pd\\n\\n# Set pandas display options\\npd.set_option('display.max_rows', None)\\n\\n# Connect to database and load ALL columns with EXACT names\\nconn = sqlite3.connect('historical_data_500_tickers_with_gains.db')\\ncursor = conn.cursor()\\ncursor.execute('SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct FROM stock_data WHERE Adj_Close IS NOT NULL AND Daily_Gain_Pct IS NOT NULL AND Forward_Gain_Pct IS NOT NULL')\\nrows = cursor.fetchall()\\nconn.close()\\n\\n# Convert to DataFrame with EXACT column names (NOT Close, NOT Symbol)\\ndf = pd.DataFrame(rows, columns=['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct'])\\n\\n# MANDATORY: Sort by Ticker and Date BEFORE any operations\\ndf = df.sort_values(['Ticker', 'Date'], ascending=[True, True])\\n\\n# Calculate 10-day and 5-day moving averages for all tickers\\ndf['10_Day_MA'] = df.groupby('Ticker')['Adj_Close'].rolling(window=10, min_periods=1).mean().values\\ndf['5_Day_MA'] = df.groupby('Ticker')['Adj_Close'].rolling(window=5, min_periods=1).mean().values\\n\\n# CRITICAL: Final DataFrame MUST have EXACT column names in EXACT order\\nfinal_df = df[['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct', '10_Day_MA', '5_Day_MA']].copy()\\n\\n# MANDATORY: Sort by Date DESC (latest first) before saving\\nfinal_df = final_df.sort_values('Date', ascending=False)\\nfinal_df.to_csv('{output_file}', index=False)\\nprint('Results saved to output.csv with EXACT database column names and Date DESC sorting')",
    "explanation": "The code connects to the SQLite database and loads ALL 5 database columns with EXACT names (Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct), MANDATORILY sorts by Ticker and Date before any calculations, calculates moving averages, and saves the complete result to output.csv. CRITICAL: Uses exact database column names - 'Adj_Close' NOT 'Close', 'Ticker' NOT 'Symbol'. Final CSV has exact column order required.",
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
    
    def write_code_to_file(self, code: str, filename: str = None) -> str:
        """Write generated code to separate file"""
        if filename is None:
            filename = self.generated_file_path
            
        full_code = f'''#!/usr/bin/env python3
"""
Generated Code Execution Module
This file contains AI-generated code for execution.
"""

{code}
'''
        
        with open(filename, "w") as f:
            f.write(full_code)
        
        return filename
    
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
    
    
    def execute_code(self, code: str, filename: str = None) -> TestResult:
        """Execute generated code directly as a standalone script"""
        start_time = time.time()
        
        try:
            # Write code to separate file
            code_file = self.write_code_to_file(code, filename)
            
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
                stdout, stderr = self.current_process.communicate(timeout=500)
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
                error="Code execution timed out after 120 seconds. Consider optimizing the code: reduce data processing complexity, use vectorized operations instead of loops, limit dataset size with date filters, optimize database queries with appropriate WHERE clauses, avoid nested loops or expensive calculations. For rolling window operations, pivot dataframes only once to wide format, use pandas/numpy rolling correlation or efficient slicing instead of full dataframe scans, ensure algorithms are O(n × tickers²) or better, and store only necessary results for each window to minimize memory usage."
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
    
    def parallel_process_generic(self, task: str, task_type: str = "prompt", max_complete_restarts: int = 1, max_error_attempts: int = 2, filters=None, progress_callback=None) -> Dict[str, Any]:
        """Generic parallel processing for both prompts and conditions"""
        print(f"Starting parallel processing for {task_type}: {task}")

        if progress_callback:
            progress_callback(f"data: {json.dumps({'type': 'parallel_init', 'message': f'Initializing parallel {task_type} processing...'})}\n\n")
        
        # Configure based on task type
        if task_type == "condition":
            output_file1 = "condition_output_llm1.csv"
            output_file2 = "condition_output_llm2.csv"
            final_output = "condition_output.csv"
            filename_prefix = "condition_generated_code"
        else:  # prompt
            output_file1 = "output_llm1.csv"
            output_file2 = "output_llm2.csv" 
            final_output = "output.csv"
            filename_prefix = "generated_code"
        
        # Load data for analytics (skip for condition processing)
        data_stats = {}
        if task_type == "prompt":
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
        
        # Try up to 3 complete attempts to get identical CSVs
        max_identity_attempts = 3
        for identity_attempt in range(max_identity_attempts):
            print(f"\n=== Identity Attempt {identity_attempt + 1}/{max_identity_attempts} ===")
            results = {"llm1": None, "llm2": None}
            
            def process_llm(llm_id: str, output_file: str):
                """Process single LLM generation and execution with complete restart mechanism"""
                print(f"Starting {llm_id} processing...")
                if progress_callback:
                    progress_callback(f"data: {json.dumps({'type': 'llm_processing', 'message': f'{llm_id.upper()} processing started', 'llm': llm_id})}\n\n")

                # Complete restart loop
                for complete_restart in range(max_complete_restarts + 1):  # +1 for initial attempt
                    print(f"{llm_id} - Complete restart {complete_restart + 1}/{max_complete_restarts + 1}")
                    if progress_callback:
                        progress_callback(f"data: {json.dumps({'type': 'llm_restart', 'message': f'{llm_id.upper()} restart {complete_restart + 1}/{max_complete_restarts + 1}', 'llm': llm_id, 'restart': complete_restart + 1})}\n\n")
                    
                    previous_error = None
                    previous_code = None
                    
                    # Error attempt loop within each complete restart
                    for error_attempt in range(1, max_error_attempts + 2):  # +1 for initial attempt within restart
                        try:
                            total_attempt = complete_restart * (max_error_attempts + 1) + error_attempt
                            print(f"{llm_id} - Error attempt {error_attempt}/{max_error_attempts + 1} (total attempt {total_attempt})")
                            if progress_callback:
                                progress_callback(f"data: {json.dumps({'type': 'llm_attempt', 'message': f'{llm_id.upper()} attempt {error_attempt}/{max_error_attempts + 1}', 'llm': llm_id, 'attempt': error_attempt, 'total_attempt': total_attempt})}\n\n")

                            # Generate code based on task type
                            if progress_callback:
                                progress_callback(f"data: {json.dumps({'type': 'llm_generating', 'message': f'{llm_id.upper()} generating code...', 'llm': llm_id})}\n\n")
                            if task_type == "condition":
                                # Use condition generation logic - import here to avoid circular import
                                import sys
                                import os
                                sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
                                try:
                                    from backend.app import generate_condition_code
                                except ImportError:
                                    # Fallback: create a simple condition code generator
                                    def generate_condition_code(prompt, output_file="condition_output.csv"):
                                        # Use the existing gemini client to generate condition code
                                        return self.gemini_client.generate_code(f"Generate Python code to evaluate this condition on existing CSV data: {prompt}. Save to {output_file}")
                                
                                if error_attempt == 1:
                                    generation = generate_condition_code(task, output_file)
                                else:
                                    retry_prompt = f"{task}\n\nPREVIOUS ATTEMPT FAILED WITH ERROR: {previous_error}\n\nPlease fix the above error and try again."
                                    generation = generate_condition_code(retry_prompt, output_file)
                            else:
                                # Use regular prompt generation
                                if error_attempt == 1:
                                    # Build filter parameters
                                    ticker_filters = None
                                    date_filters = None
                                    
                                    if filters:
                                        if filters.get('selected_tickers') or filters.get('ticker_count'):
                                            ticker_filters = {
                                                'selected_tickers': filters.get('selected_tickers', []),
                                                'ticker_count': filters.get('ticker_count')
                                            }
                                        
                                        if filters.get('start_date') or filters.get('end_date'):
                                            date_filters = {
                                                'start_date': filters.get('start_date'),
                                                'end_date': filters.get('end_date')
                                            }
                                    
                                    generation = self.gemini_client.generate_code(
                                        task, 
                                        output_file=output_file,
                                        ticker_filters=ticker_filters,
                                        date_filters=date_filters
                                    )
                                else:
                                    print(f"{llm_id} - Retrying with error context from previous attempt")
                                    # Build filter parameters for retry
                                    ticker_filters = None
                                    date_filters = None
                                    
                                    if filters:
                                        if filters.get('selected_tickers') or filters.get('ticker_count'):
                                            ticker_filters = {
                                                'selected_tickers': filters.get('selected_tickers', []),
                                                'ticker_count': filters.get('ticker_count')
                                            }
                                        
                                        if filters.get('start_date') or filters.get('end_date'):
                                            date_filters = {
                                                'start_date': filters.get('start_date'),
                                                'end_date': filters.get('end_date')
                                            }
                                    
                                    generation = self.gemini_client.generate_code(
                                        task, 
                                        error_context=previous_error, 
                                        failed_code=previous_code, 
                                        output_file=output_file,
                                        ticker_filters=ticker_filters,
                                        date_filters=date_filters
                                    )
                            
                            # Install requirements if needed
                            if generation.requirements:
                                if progress_callback:
                                    progress_callback(f"data: {json.dumps({'type': 'llm_installing', 'message': f'{llm_id.upper()} installing requirements...', 'llm': llm_id})}\n\n")
                                if not self.code_executor.install_requirements(generation.requirements):
                                    print(f"{llm_id} - Failed to install requirements on attempt {total_attempt}")
                                    if progress_callback:
                                        progress_callback(f"data: {json.dumps({'type': 'llm_install_failed', 'message': f'{llm_id.upper()} failed to install requirements', 'llm': llm_id})}\n\n")
                                    continue

                            # Execute code with specific filename
                            if progress_callback:
                                progress_callback(f"data: {json.dumps({'type': 'llm_executing', 'message': f'{llm_id.upper()} executing code...', 'llm': llm_id})}\n\n")

                            filename = f"{filename_prefix}_{llm_id}.py"
                            test_result = self.code_executor.execute_code(generation.code, filename)

                            if test_result.success:
                                if progress_callback:
                                    progress_callback(f"data: {json.dumps({'type': 'llm_success', 'message': f'{llm_id.upper()} succeeded on attempt {total_attempt}!', 'llm': llm_id, 'attempt': total_attempt})}\n\n")
                                print(f"{llm_id} - Success on total attempt {total_attempt}!")
                                results[llm_id] = {
                                    "generation": generation,
                                    "test_result": test_result,
                                    "attempts": total_attempt,
                                    "complete_restarts": complete_restart,
                                    "error_attempts": error_attempt
                                }
                                return
                            else:
                                if progress_callback:
                                    progress_callback(f"data: {json.dumps({'type': 'llm_failed', 'message': f'{llm_id.upper()} failed on attempt {total_attempt}', 'llm': llm_id, 'error': test_result.error[:100]})}\n\n")
                                print(f"{llm_id} - Failed on attempt {total_attempt}: {test_result.error}")
                                # Store error context for next retry
                                previous_error = test_result.error
                                previous_code = generation.code

                        except Exception as e:
                            total_attempt = complete_restart * (max_error_attempts + 1) + error_attempt
                            if progress_callback:
                                progress_callback(f"data: {json.dumps({'type': 'llm_exception', 'message': f'{llm_id.upper()} exception on attempt {total_attempt}', 'llm': llm_id, 'error': str(e)[:100]})}\n\n")
                            print(f"{llm_id} - Exception on attempt {total_attempt}: {str(e)}")
                            # Store exception context for next retry
                            previous_error = str(e)
                            if 'generation' in locals():
                                previous_code = generation.code
                    
                    # All error attempts for this complete restart failed
                    if complete_restart < max_complete_restarts:
                        if progress_callback:
                            progress_callback(f"data: {json.dumps({'type': 'llm_restart_needed', 'message': f'{llm_id.upper()} restart {complete_restart + 1} failed, starting fresh...', 'llm': llm_id})}\n\n")
                        print(f"{llm_id} - Complete restart {complete_restart + 1} failed, starting fresh...")
                    else:
                        if progress_callback:
                            progress_callback(f"data: {json.dumps({'type': 'llm_all_restarts_failed', 'message': f'{llm_id.upper()} all restarts failed', 'llm': llm_id})}\n\n")
                        print(f"{llm_id} - All complete restarts failed")

                if progress_callback:
                    progress_callback(f"data: {json.dumps({'type': 'llm_finished', 'message': f'{llm_id.upper()} finished (all attempts exhausted)', 'llm': llm_id, 'success': False})}\n\n")
                print(f"{llm_id} - All attempts failed")
            
            # Run both LLMs in parallel using threads
            import threading

            print("Running LLM1 and LLM2 in parallel...")
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'llm_parallel_start', 'message': 'Starting LLM1 and LLM2 in parallel...', 'identity_attempt': identity_attempt + 1})}\n\n")

            thread1 = threading.Thread(target=lambda: process_llm("llm1", output_file1))
            thread2 = threading.Thread(target=lambda: process_llm("llm2", output_file2))

            thread1.start()
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'llm_started', 'message': 'LLM1 started', 'llm': 'llm1'})}\n\n")

            thread2.start()
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'llm_started', 'message': 'LLM2 started', 'llm': 'llm2'})}\n\n")

            thread1.join()
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'llm_completed', 'message': 'LLM1 completed', 'llm': 'llm1'})}\n\n")

            thread2.join()
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'llm_completed', 'message': 'LLM2 completed', 'llm': 'llm2'})}\n\n")

            print("Both LLMs completed")
            
            # Check if both succeeded
            successful_results = [r for r in results.values() if r is not None]
            
            if len(successful_results) == 2:
                # Both succeeded, check if CSVs are identical
                print("Both LLMs succeeded. Checking if CSV outputs are identical...")
                if progress_callback:
                    progress_callback(f"data: {json.dumps({'type': 'csv_comparison_start', 'message': 'Both LLMs succeeded. Comparing CSV outputs...'})}\n\n")

                csv1_path = output_file1
                csv2_path = output_file2

                try:
                    import pandas as pd
                    df1 = pd.read_csv(csv1_path)
                    df2 = pd.read_csv(csv2_path)

                    # Check if CSVs are identical
                    are_identical = df1.equals(df2)

                    if are_identical:
                        if progress_callback:
                            progress_callback(f"data: {json.dumps({'type': 'csv_identical', 'message': f'CSVs are identical! (attempt {identity_attempt + 1})'})}\n\n")
                        print(f"SUCCESS! CSVs are identical on identity attempt {identity_attempt + 1}")
                        # Use the first result and mark as identical
                        final_generation = results["llm1"]["generation"]
                        final_test_result = results["llm1"]["test_result"]
                        
                        self.copy_to_output_csv(csv1_path, final_output)
                        
                        analytics = Analytics.analyze_results([final_generation], [final_test_result], data_stats)
                        analytics["generation_info"] = {
                            "retry_attempts": results["llm1"]["attempts"],
                            "max_retries": (max_complete_restarts + 1) * (max_error_attempts + 1),
                            "complete_restarts": results["llm1"].get("complete_restarts", 0),
                            "max_complete_restarts": max_complete_restarts,
                            "error_attempts": results["llm1"].get("error_attempts", 1),
                            "max_error_attempts": max_error_attempts,
                            "parallel_mode": True,
                            "successful_llms": 2,
                            "total_llms": 2,
                            "identity_attempts": identity_attempt + 1,
                            "max_identity_attempts": max_identity_attempts,
                            "csvs_identical": True,
                            "selected_llm": "llm1",
                            "task_type": task_type
                        }
                        
                        return {
                            "generation": final_generation,
                            "test_result": final_test_result,
                            "analytics": analytics,
                            "parallel_results": results,
                            "csvs_identical": True,
                            "identity_attempt": identity_attempt + 1,
                            "task_type": task_type
                        }
                    else:
                        if progress_callback:
                            progress_callback(f"data: {json.dumps({'type': 'csv_not_identical', 'message': f'CSVs are NOT identical (attempt {identity_attempt + 1})'})}\n\n")

                        print(f"CSVs are NOT identical on identity attempt {identity_attempt + 1}")
                        if identity_attempt + 1 < max_identity_attempts:
                            if progress_callback:
                                progress_callback(f"data: {json.dumps({'type': 'retry_identity', 'message': f'Restarting for identity attempt {identity_attempt + 2}...'})}\n\n")
                            print(f"Restarting both LLMs for identity attempt {identity_attempt + 2}...")
                            continue
                        else:
                            if progress_callback:
                                progress_callback(f"data: {json.dumps({'type': 'identity_failed', 'message': 'Max identity attempts reached. Using comparison logic.'})}\n\n")
                            print("Maximum identity attempts reached. Falling back to comparison logic.")
                            break
                            
                except Exception as e:
                    print(f"Error comparing CSVs: {str(e)}. Falling back to comparison logic.")
                    break
            else:
                # Not both succeeded, exit identity attempts and handle normally
                print(f"Only {len(successful_results)} LLM(s) succeeded. Exiting identity attempts.")
                break
        
        # Fallback logic - analyze results using existing comparison logic
        print("\\nFalling back to existing CSV comparison logic...")
        successful_results = [r for r in results.values() if r is not None]
        
        if len(successful_results) == 0:
            return {"error": "Both LLM calls failed", "task_type": task_type}
        elif len(successful_results) == 1:
            # One succeeded, use it
            successful_result = successful_results[0]
            
            # Determine which LLM succeeded and copy its CSV to output.csv
            if results["llm1"] is not None:
                selected_file = output_file1
                selected_llm = "llm1"
            else:
                selected_file = output_file2
                selected_llm = "llm2"
            
            self.copy_to_output_csv(selected_file, final_output)
            
            analytics = Analytics.analyze_results([successful_result["generation"]], [successful_result["test_result"]], data_stats)
            analytics["generation_info"] = {
                "retry_attempts": successful_result["attempts"],
                "max_retries": (max_complete_restarts + 1) * (max_error_attempts + 1),
                "complete_restarts": successful_result.get("complete_restarts", 0),
                "max_complete_restarts": max_complete_restarts,
                "error_attempts": successful_result.get("error_attempts", 1),
                "max_error_attempts": max_error_attempts,
                "parallel_mode": True,
                "successful_llms": 1,
                "total_llms": 2,
                "identity_attempts": max_identity_attempts,
                "max_identity_attempts": max_identity_attempts,
                "csvs_identical": False,
                "selected_llm": selected_llm,
                "task_type": task_type
            }
            
            return {
                "generation": successful_result["generation"],
                "test_result": successful_result["test_result"],
                "analytics": analytics,
                "parallel_results": results,
                "task_type": task_type
            }
        else:
            # Both succeeded, now compare and choose the best one
            print("Both LLMs succeeded. Comparing CSVs and selecting the best result...")
            
            comparison_result = self.compare_and_select_best_csv(task, results["llm1"]["generation"], results["llm2"]["generation"])
            
            if "error" in comparison_result:
                # If comparison fails, use first result as fallback
                print(f"Comparison failed: {comparison_result['error']}. Using first result as fallback.")
                final_generation = successful_results[0]["generation"]
                final_test_result = successful_results[0]["test_result"]
            else:
                final_generation = comparison_result["final_generation"]
                # Find the corresponding test result
                if comparison_result["selected_llm"] == "llm1":
                    final_test_result = results["llm1"]["test_result"]
                else:
                    final_test_result = results["llm2"]["test_result"]
            
            # Copy the final result to output.csv for frontend
            self.copy_to_output_csv(comparison_result.get("selected_file", output_file1), final_output)
            
            analytics = Analytics.analyze_results([final_generation], [final_test_result], data_stats)
            analytics["generation_info"] = {
                "retry_attempts": max(r["attempts"] for r in successful_results),
                "max_retries": (max_complete_restarts + 1) * (max_error_attempts + 1),
                "complete_restarts": max(r.get("complete_restarts", 0) for r in successful_results),
                "max_complete_restarts": max_complete_restarts,
                "error_attempts": max(r.get("error_attempts", 1) for r in successful_results),
                "max_error_attempts": max_error_attempts,
                "parallel_mode": True,
                "successful_llms": 2,
                "total_llms": 2,
                "identity_attempts": max_identity_attempts,
                "max_identity_attempts": max_identity_attempts,
                "csvs_identical": False,
                "comparison_performed": True,
                "similarity_score": comparison_result.get("similarity_score"),
                "selected_llm": comparison_result.get("selected_llm", "llm1"),
                "task_type": task_type
            }
            
            return {
                "generation": final_generation,
                "test_result": final_test_result,
                "analytics": analytics,
                "parallel_results": results,
                "comparison_result": comparison_result,
                "both_succeeded": True,
                "task_type": task_type
            }

    def generate_and_execute_parallel(self, task: str, max_complete_restarts: int = 1, max_error_attempts: int = 2, filters=None, progress_callback=None) -> Dict[str, Any]:
        """Generate code with 2 parallel LLM calls and execute both with restart mechanism and CSV identity check"""
        return self.parallel_process_generic(task, "prompt", max_complete_restarts, max_error_attempts, filters, progress_callback)

    def process_condition_parallel(self, task: str, max_complete_restarts: int = 1, max_error_attempts: int = 2, progress_callback=None) -> Dict[str, Any]:
        """Process condition with 2 parallel LLM calls and execute both with restart mechanism and CSV identity check"""
        return self.parallel_process_generic(task, "condition", max_complete_restarts, max_error_attempts, None, progress_callback)

    def copy_to_output_csv(self, source_file: str, destination_file: str = "output.csv"):
        """Copy the selected CSV to the specified output file"""
        import shutil
        try:
            if os.path.exists(source_file):
                shutil.copy2(source_file, destination_file)
                print(f"Copied {source_file} to {destination_file}")
            else:
                print(f"Warning: {source_file} not found, cannot copy to {destination_file}")
        except Exception as e:
            print(f"Error copying {source_file} to {destination_file}: {str(e)}")

    def compare_csv_files(self) -> Dict[str, Any]:
        """Compare the two generated CSV files and get similarity analysis"""
        import pandas as pd
        
        csv1_path = "output_llm1.csv"
        csv2_path = "output_llm2.csv"
        
        # Check if both files exist
        if not os.path.exists(csv1_path):
            return {"error": f"File {csv1_path} not found"}
        if not os.path.exists(csv2_path):
            return {"error": f"File {csv2_path} not found"}
        
        try:
            # Load both CSV files
            df1 = pd.read_csv(csv1_path)
            df2 = pd.read_csv(csv2_path)
            
            # Basic comparison info
            comparison_info = {
                "file1": csv1_path,
                "file2": csv2_path,
                "file1_shape": df1.shape,
                "file2_shape": df2.shape,
                "file1_columns": df1.columns.tolist(),
                "file2_columns": df2.columns.tolist(),
                "columns_match": df1.columns.tolist() == df2.columns.tolist(),
                "shapes_match": df1.shape == df2.shape
            }
            
            # Get sample data for LLM analysis
            sample_size = min(5, len(df1), len(df2))
            sample1 = df1.head(sample_size).to_dict('records') if len(df1) > 0 else []
            sample2 = df2.head(sample_size).to_dict('records') if len(df2) > 0 else []
            
            # Generate comparison code using LLM
            comparison_prompt = f"""
Generate Python code to compare two CSV files and calculate a similarity percentage (0-100%).

FILE 1: {csv1_path}
Columns: {df1.columns.tolist()}
Shape: {df1.shape}
Sample data: {sample1}

FILE 2: {csv2_path}
Columns: {df2.columns.tolist()}
Shape: {df2.shape}
Sample data: {sample2}

Create code that:
1. Loads both CSV files
2. Compares structure (columns, data types, row counts)
3. Compares data content (values, distributions)
4. Calculates an overall similarity percentage
5. Prints a detailed comparison report
6. Saves the similarity score to 'similarity_result.txt'

The similarity should consider:
- Row count similarity (20% weight)
- Actual data values (80% weight)

Expected response format (JSON):
{{
    "code": "your complete standalone Python code here",
    "explanation": "explanation of the comparison approach",
    "requirements": ["pandas", "numpy"]
}}
"""
            
            comparison_generation = self.gemini_client.generate_code(comparison_prompt)
            
            # Execute comparison code
            comparison_filename = "csv_comparison_code.py"
            comparison_result = self.code_executor.execute_code(comparison_generation.code, comparison_filename)
            
            similarity_score = None
            if comparison_result.success:
                # Try to read similarity score from file
                try:
                    if os.path.exists('similarity_result.txt'):
                        with open('similarity_result.txt', 'r') as f:
                            content = f.read().strip()
                            # Extract number from the content
                            import re
                            numbers = re.findall(r'\d+\.?\d*', content)
                            if numbers:
                                similarity_score = float(numbers[0])
                except:
                    pass
            
            return {
                "success": True,
                "comparison_info": comparison_info,
                "comparison_generation": comparison_generation,
                "comparison_result": comparison_result,
                "similarity_score": similarity_score
            }
            
        except Exception as e:
            return {"error": f"CSV comparison failed: {str(e)}"}
    
    def compare_and_select_best_csv(self, original_task: str, generation1: CodeGeneration, generation2: CodeGeneration) -> Dict[str, Any]:
        """Compare CSVs and if different, use LLM to select the best code"""
        import pandas as pd
        
        csv1_path = "output_llm1.csv"
        csv2_path = "output_llm2.csv"
        
        # Check if both files exist
        if not os.path.exists(csv1_path) or not os.path.exists(csv2_path):
            return {"error": "One or both CSV files not found"}
        
        try:
            # Load and compare CSVs
            df1 = pd.read_csv(csv1_path)
            df2 = pd.read_csv(csv2_path)
            
            # Quick comparison - check if they're identical
            are_identical = False
            try:
                are_identical = df1.equals(df2)
            except:
                are_identical = False
            
            if are_identical:
                print("CSVs are identical. Using first result.")
                return {
                    "similarity_score": 100.0,
                    "selected_llm": "llm1",
                    "selected_file": csv1_path,
                    "final_generation": generation1,
                    "final_test_result": None,  # Will be filled by caller
                    "reason": "CSVs are identical"
                }
            
            print("CSVs are different. Using simple heuristics to select the best result...")
            
            # Simple programmatic selection based on data quality metrics
            score1 = self.calculate_csv_quality_score(df1)
            score2 = self.calculate_csv_quality_score(df2)
            
            # If scores are very close (within 1.0), use LLM to decide
            if abs(score1 - score2) < 1.0:
                print(f"Quality scores are very close ({score1:.2f} vs {score2:.2f}). Using LLM for final decision...")
                return self.llm_compare_solutions(original_task, generation1, generation2, df1, df2, csv1_path, csv2_path)
            
            if score1 >= score2:
                selected_llm = "llm1"
                selected_file = csv1_path
                final_generation = generation1
                reason = f"LLM1 selected: Quality score {score1:.2f} vs {score2:.2f}"
            else:
                selected_llm = "llm2"
                selected_file = csv2_path
                final_generation = generation2
                reason = f"LLM2 selected: Quality score {score2:.2f} vs {score1:.2f}"
            
            print(reason)
            
            return {
                "similarity_score": None,  # Different CSVs
                "selected_llm": selected_llm,
                "selected_file": selected_file,
                "final_generation": final_generation,
                "final_test_result": None,  # Will be filled by caller
                "reason": reason,
                "quality_scores": {"llm1": score1, "llm2": score2}
            }
                
        except Exception as e:
            return {"error": f"Comparison failed: {str(e)}"}
    
    def calculate_csv_quality_score(self, df) -> float:
        """Calculate a quality score for a CSV DataFrame"""
        try:
            score = 0.0
            
            # Row count score (more rows is better, up to a point)
            row_count = len(df)
            if row_count > 0:
                score += min(row_count / 1000, 10.0)  # Max 10 points for rows
            
            # Column completeness score (fewer null values is better)
            if row_count > 0:
                null_ratio = df.isnull().sum().sum() / (len(df.columns) * row_count)
                completeness_score = (1.0 - null_ratio) * 20.0  # Max 20 points
                score += completeness_score
            
            # Data variety score (more unique values in key columns is better)
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols[:3]:  # Check up to 3 numeric columns
                    unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                    score += unique_ratio * 5.0  # Max 15 points total
            
            # Required columns score (having all 5 database columns)
            required_cols = ['Date', 'Ticker', 'Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']
            present_cols = sum(1 for col in required_cols if col in df.columns)
            score += (present_cols / len(required_cols)) * 25.0  # Max 25 points
            
            # Data type appropriateness (numeric columns should be numeric)
            type_score = 0
            for col in ['Adj_Close', 'Daily_Gain_Pct', 'Forward_Gain_Pct']:
                if col in df.columns:
                    if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                        type_score += 10.0  # 10 points per correct numeric column
            score += min(type_score, 30.0)  # Max 30 points
            
            return score
            
        except Exception as e:
            print(f"Error calculating quality score: {e}")
            return 0.0
    
    def llm_compare_solutions(self, original_task: str, generation1: CodeGeneration, generation2: CodeGeneration, 
                            df1, df2, csv1_path: str, csv2_path: str) -> Dict[str, Any]:
        """Use LLM to compare solutions when quality scores are too close"""
        try:
            # Get sample data for LLM analysis
            sample_size = min(5, len(df1), len(df2))
            sample1 = df1.head(sample_size).to_dict('records') if len(df1) > 0 else []
            sample2 = df2.head(sample_size).to_dict('records') if len(df2) > 0 else []
            
            # Create selection prompt
            selection_prompt = f"""
You are an expert code reviewer. Two Python solutions for the same task have very similar quality scores. Analyze them and determine which is better.

ORIGINAL TASK: {original_task}

CODE SOLUTION 1:
{generation1.code[:2000]}...

RESULTING CSV 1:
Columns: {df1.columns.tolist()}
Shape: {df1.shape}
Sample data: {sample1[:3]}

CODE SOLUTION 2:
{generation2.code[:2000]}...

RESULTING CSV 2:
Columns: {df2.columns.tolist()}
Shape: {df2.shape}
Sample data: {sample2[:3]}

Choose the better solution considering correctness, data quality, and code practices.

Respond with just the number: 1 or 2
"""
            
            # Get LLM selection (simple response, not full code generation)
            response = self.gemini_client.model.generate_content(selection_prompt)
            selection_text = response.text.strip()
            
            # Parse simple response
            if "2" in selection_text:
                selected_solution = 2
            else:
                selected_solution = 1  # Default to 1
            
            if selected_solution == 1:
                selected_llm = "llm1"
                selected_file = csv1_path
                final_generation = generation1
            else:
                selected_llm = "llm2"
                selected_file = csv2_path
                final_generation = generation2
            
            reason = f"LLM selected solution {selected_solution} (quality scores were too close)"
            print(reason)
            
            return {
                "similarity_score": None,
                "selected_llm": selected_llm,
                "selected_file": selected_file,
                "final_generation": final_generation,
                "final_test_result": None,
                "reason": reason,
                "llm_decision": selected_solution
            }
            
        except Exception as e:
            print(f"LLM comparison failed: {str(e)}. Using first result as fallback.")
            return {
                "similarity_score": None,
                "selected_llm": "llm1",
                "selected_file": csv1_path,
                "final_generation": generation1,
                "final_test_result": None,
                "reason": f"LLM comparison failed: {str(e)}"
            }
    
    def process_task_streaming(self, task: str, max_complete_restarts: int = 1, max_error_attempts: int = 2, progress_callback=None, filters=None):
        """Process task with streaming progress updates using parallel processing"""

        if progress_callback:
            progress_callback(f"data: {json.dumps({'type': 'task_start', 'message': 'Starting parallel processing...'})}\n\n")

        if progress_callback:
            progress_callback(f"data: {json.dumps({'type': 'parallel_start', 'message': 'Running 2 LLMs in parallel...'})}\n\n")

        # Use the parallel processing method with filters and pass the callback
        result = self.generate_and_execute_parallel(task, max_complete_restarts, max_error_attempts, filters, progress_callback)
        
        if "error" in result:
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'error', 'message': result['error']})}\n\n")
            return result
        
        # Update progress based on results
        if result.get("both_succeeded"):
            if progress_callback:
                progress_callback(f"data: {json.dumps({'type': 'comparison_complete', 'message': 'Both LLMs succeeded. Comparison completed.'})}\n\n")
        else:
            if progress_callback:
                selected_llm = result["analytics"]["generation_info"].get("selected_llm", "unknown")
                progress_callback(f"data: {json.dumps({'type': 'single_success', 'message': f'One LLM succeeded: {selected_llm}'})}\n\n")
        
        return result

    def process_task(self, task: str, max_complete_restarts: int = 1, max_error_attempts: int = 2) -> Dict[str, Any]:
        """Process a complete task with parallel LLM processing"""
        return self.generate_and_execute_parallel(task, max_complete_restarts, max_error_attempts)
    
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