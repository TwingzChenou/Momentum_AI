import asyncio
import os
import sys
import aiohttp
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.common.logging_utils import setup_logging

# 1. Load the raw values from .env
load_dotenv()

# 2. Fetch the basics
FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = "https://financialmodelingprep.com/stable/"

async def fetch_json(session, url):
    async with session.get(url) as response:
        return await response.json()

async def get_AAPL_OHLCV(session):
    url = f"{BASE_URL}/historical-price-eod/dividend-adjusted?symbol=AAPL&&from=2000-01-01&to=2026-01-01&apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def get_sp500_income_statement(session):
    url = f"{BASE_URL}/income-statement?symbol=AAPL&limit=5000&period=quarterly&apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def get_sp500_balance_sheet(session):
    url = f"{BASE_URL}/balance-sheet-statement?symbol=AAPL&limit=5000&period=quarterly&apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def get_sp500_cash_flow(session):
    url = f"{BASE_URL}/cash-flow-statement?symbol=AAPL&limit=5000&period=quarterly&apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def get_sp500_ratios(session):
    url = f"{BASE_URL}/ratios?symbol=AAPL&limit=5000&period=quarterly&apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def get_sp500_key_metrics(session):
    url = f"{BASE_URL}/key-metrics?symbol=AAPL&limit=5000&period=quarterly&apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def get_sp500_income_statement_growth(session):
    url = f"{BASE_URL}/income-statement-growth?symbol=AAPL&limit=5000&period=quarterly&apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def get_sp500_balance_sheet_growth(session):
    url = f"{BASE_URL}/balance-sheet-statement-growth?symbol=AAPL&limit=5000&period=quarterly&apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def get_sp500_cash_flow_growth(session):
    url = f"{BASE_URL}/cash-flow-statement-growth?symbol=AAPL&limit=5000&period=quarterly&apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def get_sp500_list(session):
    url = f"{BASE_URL}/historical-sp500-constituent?apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def get_nasdaq_list(session):
    url = f"{BASE_URL}/historical-nasdaq-constituent?apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def earnings_report(session):
    url = f"{BASE_URL}/earnings?symbol=AAPL&apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def get_sp500_OHLCV(session):
    url = f"{BASE_URL}/historical-price-eod/full?symbol=^GSPC&apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def company_screener(session):
    url = f"{BASE_URL}/company-screener?apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def get_general_news(session):
    url = f"{BASE_URL}/fmp-articles?page=100000&limit=20&apikey={FMP_API_KEY}"
    response = await fetch_json(session, url)
    return pd.DataFrame(response)

async def main():
    setup_logging()
    
    async with aiohttp.ClientSession() as session:
        # Create tasks for all API calls
        tasks = {
            "prices": get_AAPL_OHLCV(session),
            "income_statement": get_sp500_income_statement(session),
            "balance_sheet": get_sp500_balance_sheet(session),
            "cash_flow": get_sp500_cash_flow(session),
            "ratios": get_sp500_ratios(session),
            "key_metrics": get_sp500_key_metrics(session),
            "income_statement_growth": get_sp500_income_statement_growth(session),
            "balance_sheet_growth": get_sp500_balance_sheet_growth(session),
            "cash_flow_growth": get_sp500_cash_flow_growth(session),
            "sp500_list": get_sp500_list(session),
            "nasdaq_list": get_nasdaq_list(session),
            "earnings_report": earnings_report(session),
            "sp500_ohlcv": get_sp500_OHLCV(session),
            "company_screener": company_screener(session),
            "general_news": get_general_news(session)
        }
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks.values())
        
        # Map results back to keys
        data = dict(zip(tasks.keys(), results))
        
        # Print and log results (similar to original script structure)
        print("Prices: ")
        logger.info(data["prices"])
        
        print("Income Statement: ")
        logger.info(data["income_statement"])
        
        print("Balance Sheet: ")
        logger.info(data["balance_sheet"])
        
        print("Cash Flow: ")
        logger.info(data["cash_flow"])
        
        print("Ratios: ")
        logger.info(data["ratios"])
        
        print("Key Metrics: ")
        logger.info(data["key_metrics"])
        
        print("Income Statement Growth: ")
        logger.info(data["income_statement_growth"])
        
        print("Balance Sheet Growth: ")
        logger.info(data["balance_sheet_growth"])
        
        print("Cash Flow Growth: ")
        logger.info(data["cash_flow_growth"])
        
        print("SP500 List: ")
        logger.info(data["sp500_list"])
        
        print("NASDAQ List: ")
        logger.info(data["nasdaq_list"])
        
        print("Earnings Report: ")
        logger.info(data["earnings_report"])
        
        print("SP500 OHLCV: ")
        logger.info(data["sp500_ohlcv"])
        
        print("Company Screener: ")
        logger.info(data["company_screener"])

        print("General News: ")
        logger.info(data["general_news"])

        print(data["general_news"]["date"])

if __name__ == "__main__":
    asyncio.run(main())