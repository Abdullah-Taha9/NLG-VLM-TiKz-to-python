"""Configuration settings for the TikZ to Matplotlib pipeline."""

import os

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.5-pro" #models/gemini-2.5-pro

# Data settings
DATA_LIMIT = 10  # Number of samples to process (for testing)

# Rate limiting
DELAY_SECONDS = 1  # Delay between API calls

# Output
OUTPUT_DIR = "./output"
