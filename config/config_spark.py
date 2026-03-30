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
    
    # Specific Tables SP500 BRONZE
    SP500_LATEST_TICKERS = f"{BRONZE}/sp500_latest_tickers"
    SP500_LIST_TICKERS = f"{BRONZE}/sp500_list_tickers"
    SP500_STOCK_PRICES = f"{BRONZE}/sp500_stock_prices"
    SP500_STOCK_PRICES_WEEKLY = f"{BRONZE}/sp500_stock_prices_weekly"
    SP500_INCOME_STATEMENT = f"{BRONZE}/sp500_income_statement"
    SP500_BALANCE_SHEET = f"{BRONZE}/sp500_balance_sheet"
    SP500_CASH_FLOW = f"{BRONZE}/sp500_cash_flow"
    SP500_RATING = f"{BRONZE}/sp500_rating"
    SP500_RATIOS = f"{BRONZE}/sp500_ratios"
    TREASURY_BOND_BRONZE = f"{BRONZE}/treasury_bond"
    SP500_CONSOLIDATED_HISTORY = f"{BRONZE}/sp500_consolidated_history"
    SP500_KEY_METRICS = f"{BRONZE}/sp500_key_metrics"
    SP500_CRASH_BRONZE = f"{BRONZE}/sp500_crash"
    SP500_INCOME_STATEMENT_GROWTH = f"{BRONZE}/sp500_income_statement_growth"
    SP500_BALANCE_SHEET_GROWTH = f"{BRONZE}/sp500_balance_sheet_growth"
    SP500_CASH_FLOW_GROWTH = f"{BRONZE}/sp500_cash_flow_growth"
    SP500_FINANCIAL_STATEMENT_GROWTH = f"{BRONZE}/sp500_financial_statement_growth"
    SP500_EARNINGS_SURPRISE = f"{BRONZE}/sp500_earnings_surprise"

    COMMODITIES_LIST_TICKERS = f"{BRONZE}/commodities_list_tickers"
    COMMODITIES_STOCK_PRICES = f"{BRONZE}/commodities_stock_prices"
    COMMODITIES_STOCK_PRICES_WEEKLY = f"{BRONZE}/commodities_stock_prices_weekly"
    


    # Specific Tables SILVER
    SP500_CONSOLIDATED_HISTORY = f"{SILVER}/sp500_consolidated_history"
    SP500_STOCK_PRICES_SILVER = f"{SILVER}/sp500_stock_prices"
    SP500_INCOME_STATEMENT_SILVER = f"{SILVER}/sp500_income_statement"
    SP500_BALANCE_SHEET_SILVER = f"{SILVER}/sp500_balance_sheet"
    SP500_CASH_FLOW_SILVER = f"{SILVER}/sp500_cash_flow"
    SP500_RATING_SILVER = f"{SILVER}/sp500_rating"
    SP500_RATIOS_SILVER = f"{SILVER}/sp500_ratios"
    MACRO_PRICES_SILVER = f"{SILVER}/macro_prices"
    SP500_CRASH_SILVER = f"{SILVER}/sp500_crash"
    SP500_INCOME_STATEMENT_GROWTH_SILVER = f"{SILVER}/sp500_income_statement_growth"
    SP500_BALANCE_SHEET_GROWTH_SILVER = f"{SILVER}/sp500_balance_sheet_growth"
    SP500_CASH_FLOW_GROWTH_SILVER = f"{SILVER}/sp500_cash_flow_growth"
    SP500_FINANCIAL_STATEMENT_GROWTH_SILVER = f"{SILVER}/sp500_financial_statement_growth"
    SP500_EARNINGS_SURPRISE_SILVER = f"{SILVER}/sp500_earnings_surprise"
    
    # Specific Tables GOLD
    SP500_STOCK_PRICES_GOLD = f"{GOLD}/sp500_stock_prices"
    TREASURY_BOND_GOLD = f"{GOLD}/treasury_bond"
    SP500_STOCK_PRICES_WEEKLY_GOLD = f"{GOLD}/sp500_stock_prices_weekly"
    SP500_MOMENTUM_WEEKLY_GOLD = f"{GOLD}/sp500_momentum_weekly"
    SP500_MOMENTUM_VALUE_PROFITABLE_WEEKLY_GOLD = f"{GOLD}/sp500_momentum_value_profitable_weekly"
    SP500_MOMENTUM_VALUE_PROFITABLE_TREND_WEEKLY_GOLD = f"{GOLD}/sp500_momentum_value_profitable_trend_weekly"
    SP500_MOMENTUM_CRASH_WEEKLY_GOLD = f"{GOLD}/sp500_momentum_crash_weekly"
    SP500_MOMENTUM_VALUE_PROFITABLE_CRASH_WEEKLY_GOLD = f"{GOLD}/sp500_momentum_value_profitable_crash_weekly"
    SP500_MOMENTUM_VALUE_PROFITABLE_GROWTH_CRASH_WEEKLY_GOLD = f"{GOLD}/sp500_momentum_value_profitable_growth_crash_weekly"
    SP500_MOMENTUM_VALUE_PROFITABLE_GROWTH_SURPRISE_CRASH_WEEKLY_GOLD = f"{GOLD}/sp500_momentum_value_profitable_growth_surprise_crash_weekly"
    


# 4. Verification (Optional but recommended)
if not BUCKET_NAME:
    raise ValueError("❌ BUCKET_NAME is missing from .env file!")