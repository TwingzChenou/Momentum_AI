import os
import requests
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")

def get_available_exchanges():
    url = f"https://financialmodelingprep.com/stable/available-exchanges?apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = get_available_exchanges()
    if not df.empty:
        france_exchanges = df[df['countryCode'] == 'FR']
        print("\nExchanges for France (FR):")
        print(france_exchanges)
        
        germany_exchanges = df[df['countryCode'] == 'DE']
        print("\nExchanges for Germany (DE):")
        print(germany_exchanges)
