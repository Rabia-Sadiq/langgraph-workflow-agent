import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env file")