import os
from dotenv import load_dotenv

# 1. Load the raw values from .env
# This looks for a .env file and loads it into os.environ
load_dotenv() 

# 2. Fetch the basics
BUCKET_NAME = os.getenv("BUCKET_NAME")
FMP_API_KEY = os.getenv("FMP_API_KEY")
GCP_KEY_PATH = os.getenv("GCP_KEY_PATH", "./keys/gcs-key.json") # Default value if missing

# 3. Construct the Derived Paths (The "f-string" part)
# This logic belongs here, in Python!
class Paths:
    BRONZE = f"gs://{BUCKET_NAME}/bronze"
    SILVER = f"gs://{BUCKET_NAME}/silver"
    GOLD   = f"gs://{BUCKET_NAME}/gold"
    
    # Specific Tables
    STOCK_PRICES = f"{BRONZE}/stock_prices"
    SP500_HISTORY = f"{SILVER}/sp500_composition_history"

# 4. Verification (Optional but recommended)
if not BUCKET_NAME:
    raise ValueError("❌ BUCKET_NAME is missing from .env file!")