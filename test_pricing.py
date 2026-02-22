import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
res = requests.get(f"https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted?symbol=AAPL&from=2024-01-01&to=2024-01-05&apikey={FMP_API_KEY}")

try:
    data = res.json()
    if isinstance(data, list):
        print(json.dumps(data[:2], indent=2))
    else:
        print(json.dumps(data, indent=2))
except Exception as e:
    print(e)
    print(res.text)
