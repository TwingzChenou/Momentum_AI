import os
import requests
from dotenv import load_dotenv

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")

def check_key():
    url = f"https://financialmodelingprep.com/api/v3/market_capitalization/AAPL?apikey={FMP_API_KEY}"
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}")

if __name__ == "__main__":
    check_key()
