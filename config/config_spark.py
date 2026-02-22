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
    
    # Specific Tables BRONZE
    SP500_LATEST_TICKERS = f"{BRONZE}/sp500_latest_tickers"
    SP500_LIST_TICKERS = f"{BRONZE}/sp500_list_tickers"
    SP500_STOCK_PRICES = f"{BRONZE}/sp500_stock_prices"
    SP500_INCOME_STATEMENT = f"{BRONZE}/sp500_income_statement"
    SP500_BALANCE_SHEET = f"{BRONZE}/sp500_balance_sheet"
    SP500_CASH_FLOW = f"{BRONZE}/sp500_cash_flow"
    SP500_RATING = f"{BRONZE}/sp500_rating"
    SP500_RATIOS = f"{BRONZE}/sp500_ratios"
    SP500_CONSOLIDATED_HISTORY = f"{SILVER}/sp500_consolidated_history"


    # Specific Tables SILVER
    SP500_CONSOLIDATED_HISTORY = f"{SILVER}/sp500_consolidated_history"
    SP500_STOCK_PRICES_SILVER = f"{SILVER}/sp500_stock_prices"
    SP500_INCOME_STATEMENT_SILVER = f"{SILVER}/sp500_income_statement"
    SP500_BALANCE_SHEET_SILVER = f"{SILVER}/sp500_balance_sheet"
    SP500_CASH_FLOW_SILVER = f"{SILVER}/sp500_cash_flow"
    SP500_RATING_SILVER = f"{SILVER}/sp500_rating"
    SP500_RATIOS_SILVER = f"{SILVER}/sp500_ratios"
    

    


# 4. Verification (Optional but recommended)
if not BUCKET_NAME:
    raise ValueError("❌ BUCKET_NAME is missing from .env file!")