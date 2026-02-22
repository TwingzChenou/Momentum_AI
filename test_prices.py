import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
res = requests.get(f"https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted?symbol=AAPL&from=2024-01-01&to=2024-01-05&apikey={FMP_API_KEY}")
print(json.dumps(res.json()[:2] if isinstance(res.json(), list) else res.json(), indent=2))
