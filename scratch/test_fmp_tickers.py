import os
import requests
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")

def fetch_tickers_test(use_exchange_filter=True):
    """
    Test function to fetch tickers with or without exchange filter.
    """
    TARGET_COUNTRIES = "US,FR,IT,DE,CN,IN,BR,CA,JP,GB,NL,CH,TW,KR,AU,DK"
    
    base_url = "https://financialmodelingprep.com/stable/company-screener?"
    params = {
        "marketCapMoreThan": 2000000000,
        "country": TARGET_COUNTRIES,
        "isEtf": "false",
        "isFund": "false",
        "isActivelyTrading": "true",
        "limit": 10000,
        "apikey": FMP_API_KEY
    }
    
    if use_exchange_filter:
        params["exchange"] = "NYSE,NASDAQ"
        logger.info("🔍 Testing WITH exchange=NYSE,NASDAQ filter...")
    else:
        logger.info("🌍 Testing WITHOUT exchange filter (All exchanges in target countries)...")

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return pd.DataFrame()

def analyze_results(df, label):
    if df.empty:
        print(f"\n--- {label} ---")
        print("No data found.")
        return

    print(f"\n--- {label} ---")
    print(f"Total tickers found: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Country distribution
    if 'country' in df.columns:
        print("\nTop 10 Countries:")
        print(df['country'].value_counts().head(10))
    
    # Exchange distribution
    exch_col = 'exchangeShortName' if 'exchangeShortName' in df.columns else 'exchange'
    if exch_col in df.columns:
        print(f"\nTop 10 Exchanges ({exch_col}):")
        print(df[exch_col].value_counts().head(10))
    
    # Sample of non-US tickers
    if 'country' in df.columns:
        non_us = df[df['country'] != 'US']
        if not non_us.empty:
            print("\nSample of Non-US tickers:")
            cols_to_show = [c for c in ['symbol', 'companyName', 'name', 'country', exch_col] if c in df.columns]
            print(non_us[cols_to_show].head(10))
        else:
            print("\nNo Non-US tickers found.")

def fetch_tickers_final_test(exchange):
    """
    Test function to fetch tickers with the correct exchange code.
    """
    base_url = "https://financialmodelingprep.com/stable/company-screener?"
    params = {
        "marketCapMoreThan": 1000000000, # Lowered to 1B for better chances
        "exchange": exchange,
        "isEtf": "false",
        "isFund": "false",
        "isActivelyTrading": "true",
        "limit": 100,
        "apikey": FMP_API_KEY
    }
    
    logger.info(f"🧪 Testing final attempt with exchange={exchange}...")

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logger.error(f"❌ Error for {exchange}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    if not FMP_API_KEY:
        logger.error("FMP_API_KEY not found in .env file")
    else:
        for exch in ["PAR", "XETRA", "NYSE"]:
            df = fetch_tickers_final_test(exch)
            analyze_results(df, f"EXCHANGE: {exch}")
