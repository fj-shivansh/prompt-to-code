import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv('.env')

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_keys_str = os.getenv("GEMINI_API_KEYS")
    if api_keys_str:
        api_keys = [key.strip() for key in api_keys_str.split(',')]
        api_key = api_keys[0]

if not api_key:
    print("Error: No API key found")
    exit(1)

genai.configure(api_key=api_key)

print("Available Gemini models:")
print("=" * 60)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"Model: {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Supported Methods: {model.supported_generation_methods}")
        print()
