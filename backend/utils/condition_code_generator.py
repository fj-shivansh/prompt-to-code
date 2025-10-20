"""
Condition code generation utility
"""
import os
import pandas as pd


def generate_condition_code(condition_prompt: str, system, output_file: str = "condition_output.csv"):
    """Generate Python code for condition evaluation using existing CSV data"""

    # Get CSV headers information and actual column names
    csv_headers = ""
    actual_columns = []

    # Try multiple possible paths for output.csv
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    possible_paths = [
        os.path.abspath('../output.csv'),
        os.path.abspath('./output.csv'),
        os.path.abspath('output.csv'),
        os.path.join(current_dir, 'output.csv'),
        os.path.join(current_dir, '..', 'output.csv')
    ]

    print(f"Will try these paths for output.csv:")
    for path in possible_paths:
        print(f"  - {path} (exists: {os.path.exists(path)})")

    csv_found = False
    for output_csv_path in possible_paths:
        try:
            print(f"Checking for CSV at: {output_csv_path}")
            if os.path.exists(output_csv_path):
                df = pd.read_csv(output_csv_path)
                actual_columns = df.columns.tolist()
                csv_headers = f"Available columns in output.csv: {', '.join(actual_columns)}"

                # Add sample data for better context
                if len(df) > 0:
                    sample_row = df.iloc[0].to_dict()
                    sample_values = {k: v for k, v in sample_row.items()}
                    csv_headers += f"\n\nSample row for reference: {sample_values}"
                csv_found = True
                print(f"CSV found at: {output_csv_path}")
                print(f"Columns found: {actual_columns}")
                break
        except Exception as e:
            print(f"Error reading CSV at {output_csv_path}: {str(e)}")
            continue

    if not csv_found:
        print("ERROR: Could not find output.csv at any expected location!")
        print("Available files in current directory:")
        try:
            files = os.listdir(current_dir)
            for f in files:
                if f.endswith('.csv'):
                    print(f"  - {f}")
        except Exception as e:
            print(f"  Error listing files: {e}")

        csv_headers = "ERROR: output.csv not found. Cannot determine available columns."
        actual_columns = ["Date", "Ticker", "Adj_Close", "Daily_Gain_Pct", "Forward_Gain_Pct"]
        print(f"Using default columns: {actual_columns}")

    condition_prompt_template = f"""
You are an expert Python developer. Generate CLEAN, EXECUTABLE Python code that:

1. Reads the CSV file 'output.csv'
2. Interprets the condition: "{condition_prompt}"
3. Creates a 'Signal' column (1 for True, 0 for False)
4. Saves result to '{output_file}'

AVAILABLE COLUMNS IN output.csv:
{csv_headers}

🚨 CRITICAL: The available columns are EXACTLY: {actual_columns}
🚨 DO NOT use any columns not in this list: {actual_columns}
🚨 DO NOT assume columns like '10_Day_MA', '5_Day_MA', 'MA10', 'MA5' exist unless they are in the list above

IMPORTANT CODE STYLE REQUIREMENTS:
- Write CLEAN Python code suitable for programmatic execution
- DO NOT include shebang lines (#!/usr/bin/env python3)
- DO NOT include docstring headers or module comments
- DO NOT use print() statements for user output
- DO NOT use exit() or sys.exit()
- DO NOT include try/except blocks for file operations
- Write simple, direct pandas operations
- Only import required libraries (pandas)

Required output format:
- Must include ALL original columns: {actual_columns}
- Plus new 'Signal' column
- Final column order: {actual_columns + ['Signal']}

🚨🚨🚨 MANDATORY DATE FORMAT REQUIREMENT 🚨🚨🚨:
- The Date column MUST be in format "YYYY-MM-DD" (e.g., "2025-09-11")
- DO NOT include timestamps or time portions (no "00:00:00")
- Use this exact code before saving: df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
- This ensures both LLMs produce identical date formats

🚨🚨🚨 MANDATORY SORTING REQUIREMENT 🚨🚨🚨:
- The output CSV MUST be sorted by Date in DESCENDING order (latest first)
- Use this exact code before saving: df = df.sort_values('Date', ascending=False)

Expected response (JSON):
{{
    "code": "your complete Python code as string",
    "explanation": "brief explanation",
    "requirements": ["pandas"]
}}

CONDITION: {condition_prompt}
"""

    try:
        # Log the full prompt being sent to the LLM
        print("=" * 80)
        print("CONDITION PROMPT BEING SENT TO LLM:")
        print("=" * 80)
        print(condition_prompt_template)
        print("=" * 80)

        # Reuse existing GeminiClient from the system
        if system and system.gemini_client:
            generation = system.gemini_client.generate_condition_code(condition_prompt_template)
            print("=" * 80)
            print("LLM GENERATED CODE:")
            print("=" * 80)
            print(generation.code)
            print("=" * 80)
            return generation
        else:
            raise ValueError("System not initialized")
    except Exception as e:
        raise ValueError(f"Failed to generate condition code: {str(e)}")
