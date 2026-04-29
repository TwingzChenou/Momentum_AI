import os
import sys
import pandas as pd
import requests
from loguru import logger
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def fetch_sp500_latest(api_key):
    """Fetches latest S&P 500 constituents from FMP (Returns ~503 tickers)."""
    url = f"https://financialmodelingprep.com/stable/sp500-constituent?apikey={api_key}"
    logger.info("📡 Fetching latest S&P 500 constituents from FMP...")
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        logger.info(f"✅ Found {len(data)} tickers in latest list.")
        return pd.DataFrame(data)
    else:
        logger.error(f"❌ Failed to fetch latest data: {response.status_code}")
        return pd.DataFrame()

def main():
    setup_logging()
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        logger.error("❌ FMP_API_KEY not found in environment.")
        return

    spark = create_spark_session("SP500_List_Update_Latest")
    
    try:
        # 1. Fetch latest constituents only
        df_latest = fetch_sp500_latest(api_key)
        
        if df_latest.empty:
            logger.warning("⚠️ No data fetched from FMP.")
            return

        # 2. Convert to Spark
        sdf_latest = spark.createDataFrame(df_latest)

        # 2b. Save latest constituents separately (Used as a filter for ingestion)
        logger.info(f"💾 Saving latest 500 constituents to {Paths.SP500_LATEST_TICKERS}")
        sdf_latest.write.format("delta").mode("overwrite").save(Paths.SP500_LATEST_TICKERS)

        # 3. Join with existing SP500_LIST_TICKERS directly
        # This enriches the historical table with metadata from the latest list
        logger.info(f"🔗 Joining latest list with {Paths.SP500_LIST_TICKERS}...")
        
        try:
            sdf_existing = spark.read.format("delta").load(Paths.SP500_LIST_TICKERS)
            # Full outer join to keep all historical events and add latest metadata
            # We select specific columns from latest to avoid name collisions if necessary
            sdf_final = sdf_existing.join(
                sdf_latest.select("symbol", "name", "sector", "subSector", "dateFirstAdded"), 
                on="symbol", 
                how="full"
            )
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing list, saving latest as base: {e}")
            sdf_final = sdf_latest

        # 4. Save back to SP500_LIST_TICKERS
        logger.info(f"💾 Saving enriched list to {Paths.SP500_LIST_TICKERS}")
        sdf_final.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(Paths.SP500_LIST_TICKERS)
            
        logger.success("✅ S&P 500 List updated and joined successfully.")

    except Exception as e:
        logger.critical(f"❌ Critical error: {e}")
    finally:
        if spark: spark.stop()

if __name__ == "__main__":
    main()
