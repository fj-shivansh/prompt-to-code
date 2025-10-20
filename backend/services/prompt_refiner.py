"""
Prompt refinement and condition generation service
"""
import os
import json
from typing import Optional, List
import google.generativeai as genai


class PromptRefiner:
    """Refines user prompts and generates condition suggestions"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client for prompt refinement"""
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Use existing API key logic from environment
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                # Try multiple API keys
                api_keys_str = os.getenv("GEMINI_API_KEYS")
                if api_keys_str:
                    api_keys = [key.strip() for key in api_keys_str.split(',')]
                    api_key = api_keys[0]  # Use first key for refinement

            if not api_key:
                raise ValueError("Gemini API key not provided")
            genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def refine_prompt(self, user_request: str) -> str:
        """Refine a vague user request into a detailed, structured prompt"""
        system_prompt = f"""
You are an expert financial analyst and prompt engineer.
Transform the following user request into a **concise, step-by-step instruction** specifying:

1. Calculations required
2. Output columns and what each should contain

🚨 CRITICAL REQUIREMENT: The output CSV MUST ALWAYS contain these 5 database columns AS THE FIRST 5 COLUMNS:
Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct

Any additional calculated columns should come AFTER these 5 mandatory columns.
DO NOT rename or exclude these 5 database columns - they are required for NAV calculations.

Focus ONLY on calculations and column mapping.
Do NOT include extra instructions, handling missing data, or formatting advice.

User Request:
{user_request}
        """

        response = self.model.generate_content(system_prompt)
        return response.text.strip()

    def generate_condition_suggestions(
        self,
        columns: List[str],
        sample_data: Optional[dict] = None
    ) -> List[str]:
        """Generate profitable trading condition suggestions based on available columns"""
        print(f"[generate_condition_suggestions] Called with {len(columns)} columns")
        print(f"[generate_condition_suggestions] Columns: {columns}")
        print(f"[generate_condition_suggestions] Sample data provided: {sample_data is not None}")

        system_prompt = f"""
You are an expert quantitative trading analyst. Based on the following CSV columns from a stock trading dataset, generate 5 PROFITABLE trading condition prompts that could be used to create signals for a trading strategy.

Available Columns: {', '.join(columns)}
Sample Data (first row): {sample_data if sample_data else 'Not available'}

REQUIREMENTS:
1. Each condition should be a clear, actionable trading rule
2. Focus on conditions that have potential for profitability based on technical analysis
3. Each condition should result in a Signal column (1 for True/Buy, 0 for False/No action)
4. Be specific about thresholds and comparisons
5. Return ONLY a JSON array of condition strings, no explanations

IMPORTANT: The conditions should work with the available columns only. Don't reference columns that don't exist.

Example format (return only the JSON array):
[
  "Create Signal=1 when RSI < 30 and Daily_Gain_Pct < -2, otherwise Signal=0",
  "Create Signal=1 when MA_5 > MA_20 and volume is above average, otherwise Signal=0"
]

Generate 5 profitable condition prompts as a JSON array:
"""

        print(f"[generate_condition_suggestions] Sending request to Gemini...")
        try:
            response = self.model.generate_content(system_prompt)
            print(f"[generate_condition_suggestions] Received response from Gemini")
            print(f"[generate_condition_suggestions] Response text: {response.text[:200]}...")

            # Extract JSON from response
            text = response.text.strip()
            print(f"[generate_condition_suggestions] Looking for JSON array in response...")

            # Try to find JSON array in the response
            if '[' in text and ']' in text:
                start = text.index('[')
                end = text.rindex(']') + 1
                json_str = text[start:end]
                print(f"[generate_condition_suggestions] Extracted JSON: {json_str[:200]}...")

                conditions = json.loads(json_str)
                print(f"[generate_condition_suggestions] Successfully parsed {len(conditions)} conditions")
                result = conditions[:5]  # Ensure max 5 conditions
                print(f"[generate_condition_suggestions] Returning {len(result)} conditions")
                return result
            else:
                print(f"[generate_condition_suggestions] WARNING: No JSON array found in response")
                print(f"[generate_condition_suggestions] Full response: {text}")
                return []
        except Exception as e:
            print(f"[generate_condition_suggestions] ERROR: {str(e)}")
            print(f"[generate_condition_suggestions] Exception type: {type(e).__name__}")
            import traceback
            print(f"[generate_condition_suggestions] Traceback: {traceback.format_exc()}")
            return []
