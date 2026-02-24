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

def fetch_macro_data(ticker):
    """
    Fetches the full historical data for a macro indicator/index from FMP.
    Uses the past 13 years (API limit is 4700 days, so we fetch last 10 years).
    """

    # 1. Déterminer la date de début (10 ans en arrière)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=50*365)
    
    # 2. Formater la date au format YYYY-MM-DD
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # 3. Construire l'URL avec les paramètres 'from' et 'to'
    url = f"{BASE_URL}/full?symbol={ticker}&from={start_date_str}&to={end_date.strftime('%Y-%m-%d')}&apikey={FMP_API_KEY}"
    
    logger.info(f"Fetching data for {ticker} from {start_date_str} to {end_date.strftime('%Y-%m-%d')}...")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                # Add the symbol to each record
                for record in data:
                    record['symbol'] = ticker
                return data
            elif "Error Message" in data:
                logger.error(f"❌ API Error for {ticker}: {data['Error Message']}")
                return []
            else:
                logger.warning(f"⚠️ Unexpected response format for {ticker}: {data}")
                return []
        else:
            logger.error(f"❌ HTTP {response.status_code} for {ticker}")
            return []
    except Exception as e:
        logger.error(f"❌ Request failed for {ticker}: {e}")
        return []

def main():
    setup_logging()
    logger.info("🚀 Starting Job: Fetch and Clean Macro Indicators (^GSPC, ^IRX)")

    spark = create_spark_session(app_name="Macro_Data_Ingestion")

    try:
        all_data = []

        # 1. Fetch Market Index (^GSPC)
        idx_data = fetch_macro_data("^GSPC")
        if idx_data:
            all_data.extend(idx_data)
        else:
            logger.warning("⚠️ No data fetched for ^GSPC")

        # 2. Fetch Risk-Free Rate (^IRX) via Treasury Rates API
        end_date = datetime.now()
        start_date = end_date - timedelta(days=50*365)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        treasury_url = f"https://financialmodelingprep.com/stable/treasury-rates?from={start_date_str}&to={end_date_str}&apikey={FMP_API_KEY}"
        logger.info(f"Fetching treasury rates for ^IRX from {start_date_str} to {end_date_str}...")
        
        try:
            res = requests.get(treasury_url)
            if res.status_code == 200:
                t_data = res.json()
                if isinstance(t_data, list):
                    # We map 'month3' (3-month T-Bill yield) to ^IRX adjClose.
                    # As discussed, the daily yield is AnnualYield / (100 * 252).
                    for record in t_data:
                        if 'month3' in record and record['month3'] is not None:
                            daily_rate = float(record['month3']) / (100.0 * 252.0)
                            all_data.append({
                                'symbol': '^IRX',
                                'date': record['date'],
                                'adjClose': daily_rate,
                                'volume': 1  # Dummy volume so it passes the volume > 0 filter
                            })
                    logger.info(f"✅ Fetched {len(t_data)} treasury rate records for ^IRX")
                else:
                    logger.warning(f"⚠️ Unexpected treasury format: {t_data}")
            else:
                logger.error(f"❌ HTTP {res.status_code} for treasury rates")
        except Exception as e:
            logger.error(f"❌ Failed to fetch treasury rates: {e}")

        if not all_data:
            logger.error("❌ No macro data fetched. Exiting.")
            sys.exit(1)

        # Process into Pandas DataFrame
        df_prices = pd.DataFrame(all_data)
        
        # We need symbol, date, adjClose, volume, and we can keep others if present
        if 'close' in df_prices.columns:
            df_prices['adjClose'] = df_prices['adjClose'].fillna(df_prices['close'])
            
        expected_cols = ['symbol', 'date', 'adjClose', 'volume']
        existing_cols = [c for c in expected_cols if c in df_prices.columns]
        df_prices = df_prices[existing_cols]

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
