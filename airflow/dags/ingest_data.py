import os
import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from delta import *
from dotenv import load_dotenv

load_dotenv()  # This loads the variables from .env into the system

# --- CONFIGURATION ---
# TODO: Replace with your actual values or set them as Environment Variables
FMP_API_KEY = os.getenv("FMP_API_KEY")
GCP_KEY_PATH = os.getenv("GCP_KEY_PATH")
BUCKET_NAME = os.getenv("BUCKET_NAME")
LAKE_PATH = f"gs://{BUCKET_NAME}/bronze/stock_prices"

# --- 1. SETUP SPARK SESSION ---
from src.common.setup_spark import create_spark_session
spark = create_spark_session(app_name="FMP_Ingestion")

# --- 2. HELPER FUNCTIONS ---

import sys
import importlib.util
from loguru import logger

# Add project root to sys.path to allow importing from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.common.logging_utils import setup_logging

def get_sp500_history_from_wikipedia():
    """Dynamically imports the list_tickers_S&P500 module and calls its function."""
    try:
        module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../list_tickers_S&P500.py'))
        spec = importlib.util.spec_from_file_location("list_tickers_S&P500", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["list_tickers_S&P500"] = module
        spec.loader.exec_module(module)
        return module.get_sp500_history_from_wikipedia()
    except Exception as e:
        logger.error(f"❌ Error importing local script: {e}")
        return None

def get_sp500_tickers():
    """Fetches the current list of S&P 500 tickers from Wikipedia (via local script)."""
    try:
        # Get history dataframe from the local script
        df_history = get_sp500_history_from_wikipedia()
        
        if df_history is None or df_history.empty:
            logger.error("❌ Error: Received empty history from Wikipedia scraper.")
            return []

        # The DataFrame is indexed by date. 
        # We want the most recent entry (last row after sorting by date, which the script does).
        # The script returns df sorted by date ascending, so the last row is the newest.
        latest_entry = df_history.iloc[-1]
        latest_tickers = latest_entry['tickers']
        
        return latest_tickers
    except Exception as e:
        logger.error(f"❌ Error fetching S&P 500 list from local script: {e}")
        return []

def get_existing_tickers(lake_path):
    """Checks GCS to see which tickers we already downloaded."""
    try:
        # We try to read the Delta table to see what's inside
        # If the table doesn't exist yet, this will fail (which is fine)
        df = spark.read.format("delta").load(lake_path)
        # Get unique tickers efficiently
        existing = [row.ticker for row in df.select("ticker").distinct().collect()]
        return set(existing)
    except Exception:
        # If table doesn't exist, we have 0 existing tickers
        return set()

def fetch_history(ticker):
    """Downloads full price history for a single ticker."""
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'historical' in data and data['historical']:
            pdf = pd.DataFrame(data['historical'])
            pdf['ticker'] = ticker  # Add ticker column for partitioning
            return pdf
    return None

# --- 3. MAIN EXECUTION FLOW ---

# Setup logging
setup_logging()

# A. Get the list of targets
all_tickers = get_sp500_tickers()
logger.info(f"📋 Found {len(all_tickers)} S&P 500 companies.")

# B. Check what we already have (Optimization)
logger.info("🔍 Checking Data Lake for existing data...")
existing_tickers = get_existing_tickers(LAKE_PATH)
logger.info(f"✅ Already have data for {len(existing_tickers)} companies.")

# C. Determine what is missing
tickers_to_process = [t for t in all_tickers if t not in existing_tickers]
logger.info(f"🚀 Starting download for {len(tickers_to_process)} remaining companies...")

# D. The Loop (Batch Processing)
# We process one by one to handle errors gracefully
count = 0
for ticker in tickers_to_process:
    logger.info(f"Downloading {ticker}...")
    
    # 1. Fetch Data (Pandas)
    pdf = fetch_history(ticker)
    
    if pdf is not None and not pdf.empty:
        # 2. Convert to Spark
        # Enforce schema to ensure stability
        # (Spark sometimes guesses wrong types, so explicit is better)
        sdf = spark.createDataFrame(pdf) 
        
        # 3. Write to Delta Lake (Append Mode)
        # PartitionBy 'ticker' creates a folder structure like: /stock_prices/ticker=AAPL/
        sdf.write.format("delta") \
            .mode("append") \
            .partitionBy("ticker") \
            .save(LAKE_PATH)
            
        logger.success(f"Downloaded {ticker} ✅")
        count += 1
    else:
        logger.warning(f"Failed or No Data for {ticker} ❌")

    # Optional: Stop after 5 for testing to save API calls
    # if count >= 5: break 

logger.success(f"\n🎉 Job Complete. Processed {count} tickers.")