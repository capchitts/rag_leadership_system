# config/settings.py

import os

# Active provider: groq | huggingface | google
LLM_PROVIDER = "groq"

# Model for selected provider
LLM_MODEL = "llama-3.3-70b-versatile"

LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1200
LLM_TIMEOUT = 120

LLM_BASE_URL = None
LLM_API_VERSION = None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")