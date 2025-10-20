"""
Gemini API client with multi-key support
"""
import os
import json
from typing import Optional
import google.generativeai as genai
from ..models.base import CodeGeneration


class GeminiClient:
    """Client for Gemini API with multi-key support and round-robin rotation"""

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

        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
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

    def generate_code(self, task: str, error_context: Optional[str] = None, failed_code: Optional[str] = None,
                     output_file: str = "output.csv", ticker_filters=None, date_filters=None) -> CodeGeneration:
        """Generate code for a given task using Gemini API with round-robin key rotation"""

        print(f"Generating code - {self._get_current_api_key_info()}")

        # Build the prompt
        from ..utils.prompt_builder import build_code_generation_prompt

        prompt = build_code_generation_prompt(
            task=task,
            error_context=error_context,
            failed_code=failed_code,
            output_file=output_file,
            ticker_filters=ticker_filters,
            date_filters=date_filters
        )

        return self._send_generation_request(prompt, task)

    def generate_condition_code(self, prompt: str) -> CodeGeneration:
        """Generate condition evaluation code (reads from CSV, not database)"""
        print(f"Generating condition code - {self._get_current_api_key_info()}")
        return self._send_generation_request(prompt, "Condition evaluation")

    def _send_generation_request(self, prompt: str, task: str) -> CodeGeneration:
        """Internal method to send generation request to Gemini API"""
        max_key_attempts = len(self.api_keys)
        last_exception = None

        for attempt in range(max_key_attempts):
            try:
                print(f"🔄 Sending request to Gemini (attempt {attempt + 1}/{max_key_attempts})...")
                import time
                start_time = time.time()

                # Add request configuration with timeout
                generation_config = {
                    'temperature': 0.7,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 8192,
                }

                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={'timeout': 120}  # 2 minute timeout
                )

                elapsed = time.time() - start_time
                print(f"✅ Received response from Gemini in {elapsed:.2f}s")

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
                print(f"📝 Response length: {len(response_text)} characters")

                # Try to extract JSON from markdown code blocks
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3].strip()

                # Try to find JSON object if it's embedded in other text
                if not response_text.startswith('{'):
                    # Look for first { and last }
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        response_text = response_text[start_idx:end_idx + 1]
                        print(f"⚠️  Extracted JSON from position {start_idx} to {end_idx}")

                try:
                    parsed = json.loads(response_text)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON Parse Error at position {e.pos}: {e.msg}")
                    print(f"📄 Response preview: {response_text[:500]}...")
                    print(f"📄 Response end: ...{response_text[-500:]}")
                    raise

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
