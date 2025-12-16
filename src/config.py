import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found. Some features will be limited.")
    OPENAI_API_KEY = "sk-placeholder"  # Allow server to start for UI testing

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")