import os
import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, ArrayType
from loguru import logger
import requests
from io import StringIO

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths


# 2. Fetch the basics
FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = "https://financialmodelingprep.com/stable/"

def list_sp500_history():
    """
    Fetch the historical S&P 500 list from FMP API.
    """
    logger.info("📡 Connecting to FMP API to fetch S&P 500 latest...")
    url = f"{BASE_URL}/sp500-constituent?apikey={FMP_API_KEY}"

    try:
        response = requests.get(url)
        df = pd.DataFrame(response.json())
        if df.empty:
            logger.warning("❌ No data returned from FMP API.")
        else:
            logger.info("✅ S&P 500 latest fetched successfully.")
        return df

    except Exception as e:
        logger.error(f"❌ Error fetching S&P 500 latest: {e}")
        return None


def save_history_to_lake(spark, pandas_df):
    """
    Saves the historical composition DataFrame to Delta Lake.
    """
    logger.info(f"💾 Saving History to {Paths.SP500_LATEST_TICKERS} with {pandas_df.shape[0]} rows...")
    
    # Convert Pandas to Spark (reset index to make 'date' a column)
    pandas_df_reset = pandas_df.reset_index()
    
    try:
        sdf = spark.createDataFrame(pandas_df_reset)
        
        # Write to Delta (Overwrite mode for full history refresh)
        sdf.write.format("delta") \
            .mode("overwrite") \
            .save(Paths.SP500_LATEST_TICKERS)
            
        logger.success("✅ Success! History saved.")
        
    except Exception as e:
        logger.error(f"❌ Error saving to Lake: {e}")
        

def main():
    # Setup logging
    setup_logging()

    logger.info("🚀 Starting Job: List SP500 Latest Ingestion")

    spark=None
    df_history=None

    try:
        # 1. Fetch the data
        df_history = list_sp500_history()

        # 2. Create Spark Session
        logger.info("🚀 Creating Spark Session...")
        spark = create_spark_session(app_name="List_SP500_Latest_Ingestion")

        # 3. Save to Lake
        save_history_to_lake(spark, df_history)

    except Exception as e:
        logger.critical(f"❌ Critical Error: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()