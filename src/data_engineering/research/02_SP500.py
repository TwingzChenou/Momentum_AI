import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType
from pyspark.sql.functions import col, to_date

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
# Use the stable endpoint matching the Bronze pipeline
BASE_URL = "https://financialmodelingprep.com/stable/historical-price-eod"

def split_period(start_date, end_date, max_days=4700):
    """
    Splits a period into chunks no larger than max_days. 4700 days is ~13 years.
    Returns a list of (chunk_start, chunk_end) tuples.
    """
    chunks = []
    current_start = start_date
    while True:
        current_end = current_start + timedelta(days=max_days)
        if current_end >= end_date:
            chunks.append((current_start, end_date))
            break
        else:
            chunks.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)
    return chunks

def fetch_macro_data(ticker):
    """
    Fetches the full historical data for a macro indicator/index from FMP.
    Chunks requests into smaller periods to respect the 5000 records API limit.
    """

    end_date = datetime.now()
    start_date = end_date - timedelta(days=50*365) # Fetch ~50 years
    
    chunks = split_period(start_date, end_date)
    all_records = []
    
    for c_start, c_end in chunks:
        start_date_str = c_start.strftime('%Y-%m-%d')
        end_date_str = c_end.strftime('%Y-%m-%d')
        
        url = f"{BASE_URL}/full?symbol={ticker}&from={start_date_str}&to={end_date_str}&apikey={FMP_API_KEY}"
        logger.info(f"Fetching data for {ticker} from {start_date_str} to {end_date_str}...")
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    # Add the symbol to each record
                    for record in data:
                        record['symbol'] = ticker
                    all_records.extend(data)
                elif "Error Message" in data:
                    logger.error(f"❌ API Error for {ticker}: {data['Error Message']}")
                else:
                    logger.warning(f"⚠️ Unexpected response format for {ticker}: {data}")
            else:
                logger.error(f"❌ HTTP {response.status_code} for {ticker}: {response.text}")
        except Exception as e:
            logger.error(f"❌ Request failed for {ticker}: {e}")
            
    return all_records

def main():
    setup_logging()
    logger.info("🚀 Starting Job: Fetch and Clean Macro Indicators (^GSPC)")

    spark = create_spark_session(app_name="Macro_Data_Ingestion")

    try:
        all_data = []

        # 1. Fetch Market Index (^GSPC)
        idx_data = fetch_macro_data("^GSPC")
        if idx_data:
            all_data.extend(idx_data)

        if not all_data:
            logger.error("❌ No macro data fetched. Exiting.")
            sys.exit(1)
        # Process into Pandas DataFrame
        df_prices = pd.DataFrame(all_data)
        
        # We need symbol, date, adjClose, volume, and we can keep others if present
        if 'adjClose' not in df_prices.columns and 'close' in df_prices.columns:
            df_prices['adjClose'] = df_prices['close']
        elif 'adjClose' in df_prices.columns and 'close' in df_prices.columns:
            df_prices['adjClose'] = df_prices['adjClose'].fillna(df_prices['close'])
            
        expected_cols = ['symbol', 'date', 'adjClose', 'volume']
        existing_cols = [c for c in expected_cols if c in df_prices.columns]
        
        # If expected columns are completely missing, add them as empty to avoid KeyError downstream
        for c in expected_cols:
            if c not in existing_cols:
                df_prices[c] = None
                
        df_prices = df_prices[expected_cols]

        logger.info(f"📊 Converting {df_prices.shape[0]} rows to Spark DataFrame.")
        
        # Clean pandas to avoid null issues
        df_prices = df_prices.dropna(subset=['symbol', 'date'])
        df_prices['symbol'] = df_prices['symbol'].astype(str)
        df_prices['date'] = df_prices['date'].astype(str)
        df_prices['adjClose'] = pd.to_numeric(df_prices['adjClose'], errors='coerce').fillna(0.0)
        
        if 'volume' not in df_prices.columns:
            df_prices['volume'] = 1 # give a default dummy volume if completely missing
        df_prices['volume'] = pd.to_numeric(df_prices['volume'], errors='coerce').fillna(1).astype('int64')

        # Create Spark DataFrame
        sdf = spark.createDataFrame(df_prices)
        
        # Enforce Schema
        sdf = sdf.withColumn("symbol", col("symbol").cast(StringType())) \
                 .withColumn("date", to_date(col("date"))) \
                 .withColumn("adjClose", col("adjClose").cast(DoubleType())) \
                 .withColumn("volume", col("volume").cast(LongType()))

        logger.info("🧹 Cleaning macro data...")
        
        # User requested cleanings:
        # Don't forget to clean, remove duplicate, filter with volume >0, adjClose > 0, dropna.
        sdf_cleaned = sdf \
            .dropDuplicates(["symbol", "date"]) \
            .filter(col("volume") > 0) \
            .filter(col("adjClose") > 0) \
            .dropna(subset=["adjClose", "date", "symbol"])
            
        final_count = sdf_cleaned.count()
        logger.info(f"✅ Cleaned data has {final_count} records.")
        
        if final_count > 0:
            # Drop the dummy volume column before saving if it's not needed, but we keep it to match schema
            logger.info(f"💾 Saving macro data to Silver: {Paths.MACRO_PRICES_SILVER}")
            
            sdf_cleaned.repartition("symbol").write.format("delta") \
                .option("overwriteSchema", "true") \
                .partitionBy("symbol") \
                .mode("overwrite") \
                .save(Paths.MACRO_PRICES_SILVER)
                
            logger.success("✅ Macro Prices (Silver) saved successfully!")
        else:
            logger.error("❌ No data left after cleaning filters! Check your data source.")

    except Exception as e:
        logger.critical(f"❌ Critical Error: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
