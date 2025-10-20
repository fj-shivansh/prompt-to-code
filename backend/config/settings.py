"""
Application configuration settings
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.env')

class Settings:
    """Application settings"""

    # Database
    DB_PATH = os.path.abspath('../historical_data_500_tickers_with_gains.db')

    # API Keys
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_API_KEYS = os.getenv('GEMINI_API_KEYS')

    # Performance Settings
    ENABLE_PARALLEL_LLMS = os.getenv('ENABLE_PARALLEL_LLMS', 'true').lower() == 'true'

    # Flask
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5000

    # File exclusions for reloader
    EXCLUDE_PATTERNS = [
        '**/generated_code.py',
        '**/condition_generated_code_llm1.py',
        '**/condition_generated_code_llm2.py',
        '**/generated_code_llm1.py',
        '**/generated_code_llm2.py',
        '**/csv_comparison_code.py',
        '**/output.csv',
        '**/output_llm1.csv',
        '**/output_llm2.csv',
        '**/condition_output.csv',
        '**/similarity_result.txt'
    ]

    @classmethod
    def get_api_keys(cls):
        """Get list of API keys"""
        if cls.GEMINI_API_KEYS:
            return [key.strip() for key in cls.GEMINI_API_KEYS.split(',') if key.strip()]
        elif cls.GEMINI_API_KEY:
            return [cls.GEMINI_API_KEY]
        return []
