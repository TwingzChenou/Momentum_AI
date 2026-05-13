
import os
import requests
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")

def fetch_tickers_test():
    """
    Test function to fetch tickers with country filters.
    """
    TARGET_COUNTRIES = "US,FR,IT,DE,CN,IN,BR,CA,JP,GB,NL,CH,TW,KR,AU,DK"
    
    base_url = "https://financialmodelingprep.com/stable/company-screener?"
    params = {
        "marketCapMoreThan": 2000000000,
        "country": TARGET_COUNTRIES,
        "exchange": "NYSE,NASDAQ,PAR,XETRA,MIL,AMS,LSE,SIX,TSX,JPX,ASX,CPH,HKSE,BSE",
        "isEtf": "false",
        "isFund": "false",
        "isActivelyTrading": "true",
        "limit": 10000,
        "apikey": FMP_API_KEY
    }
    
    logger.info(f"🔍 Testing with countries={TARGET_COUNTRIES}...")

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return pd.DataFrame()

def fetch_tickers_test():
    """
    Test function to fetch tickers with country filters.
    """
    TARGET_COUNTRIES = "US,FR,IT,DE,CN,IN,BR,CA,JP,GB,NL,CH,TW,KR,AU,DK"
    
    base_url = "https://financialmodelingprep.com/stable/stock-screener?"
    params = {
        "marketCapMoreThan": 2000000000,
        "country": TARGET_COUNTRIES,
        "exchange": "NYSE,NASDAQ,PAR,XETRA,MIL,AMS,LSE,SIX,TSX,JPX,ASX,CPH,HKSE,BSE",
        "isEtf": "false",
        "isFund": "false",
        "isActivelyTrading": "true",
        "limit": 10000,
        "apikey": FMP_API_KEY
    }
    
    logger.info(f"🔍 Testing with countries={TARGET_COUNTRIES}...")

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return pd.DataFrame()

def check_ticker_in_list(ticker: str, bucket_list: list) -> bool:
    """
    Vérifie si un ticker est présent dans une liste (insensible à la casse).
    """
    ticker_upper = ticker.strip().upper()
    bucket_upper = [t.strip().upper() for t in bucket_list]
    return ticker_upper in bucket_upper

if __name__ == "__main__":
    TARGET_TICKER = "AXTI"
    
    df = fetch_tickers_test()
    logger.info(f"Nombre total de tickers : {len(df)}")
    logger.info(f"Tickers : {df['symbol'].tolist()}")
    logger.info(f"Ticker '{TARGET_TICKER}' présent dans la liste : {check_ticker_in_list(TARGET_TICKER, df['symbol'].tolist())}")
