import os
import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
import requests

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

# Load environment variables
load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")

def fetch_tickers_2b():
    """
    Fetch all tickers with market cap > 2B from FMP Screener API.
    """
    logger.info("📡 Connecting to FMP API to fetch tickers > $2B...")
    
    TARGET_COUNTRIES = "US,FR,IT,DE,CN,IN,BR,CA,JP,GB,NL,CH,TW,KR,AU,DK"
    # We use a large limit to bypass the default 1000 items limit on the API
    url = (
        f"https://financialmodelingprep.com/stable/company-screener?"
        f"marketCapMoreThan=2000000000&"
        f"country={TARGET_COUNTRIES}&"
        f"exchange=NYSE,NASDAQ&" # Sécurité absolue : Uniquement coté US
        f"isEtf=false&"
        f"isFund=false&"
        f"isActivelyTrading=true&"
        f"limit=10000&"
        f"apikey={FMP_API_KEY}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data)
        
        if df.empty:
            logger.warning("❌ No data returned from FMP API.")
        else:
            logger.info(f"✅ Successfully fetched {len(df)} tickers with market cap > 2B.")
            
        return df

    except Exception as e:
        logger.error(f"❌ Error fetching tickers from FMP: {e}")
        return None

def save_to_lake(spark, pandas_df):
    """
    Saves the fetched tickers DataFrame to Delta Lake.
    """
    logger.info(f"💾 Saving to {Paths.LIST_TICKER_2B} with {pandas_df.shape[0]} rows...")
    
    try:
        sdf = spark.createDataFrame(pandas_df)
        
        # Write to Delta (Overwrite mode)
        sdf.write.format("delta") \
            .mode("overwrite") \
            .save(Paths.LIST_TICKER_2B)
            
        logger.info(f"✅ Success! Data saved to {Paths.LIST_TICKER_2B}.")
        
    except Exception as e:
        logger.error(f"❌ Error saving to Lake: {e}")

def main():
    # Setup logging
    setup_logging()

    logger.info("🚀 Starting Job: Fetch Tickers > 2B Ingestion")

    spark = None
    df_tickers = None

    try:
        # 1. Fetch the data
        df_tickers = fetch_tickers_2b()

        if df_tickers is not None and not df_tickers.empty:
            # 2. Create Spark Session
            logger.info("🚀 Creating Spark Session...")
            spark = create_spark_session(app_name="List_Tickers_2B_Ingestion")

            # 3. Save to Lake
            save_to_lake(spark, df_tickers)
        else:
            logger.warning("⚠️ Skipping write to Lake because data fetch failed or is empty.")

    except Exception as e:
        logger.error(f"❌ Critical Error in job execution: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
