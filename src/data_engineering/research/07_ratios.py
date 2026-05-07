import os
import sys
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql.types import StringType
from pyspark.sql.functions import col, to_date

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = "https://financialmodelingprep.com/stable"

async def fetch_ratios_for_ticker(session, semaphore, ticker):
    """
    Fetches the quarterly ratios for a single ticker.
    Limits concurrency using a semaphore and implements exponential backoff for 429 Rate Limits.
    """
    url = f"{BASE_URL}/ratios?symbol={ticker}&limit=5000&period=quarterly&apikey={FMP_API_KEY}"
    
    max_retries = 5
    base_delay = 5

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list):
                            return data
                        elif "Error Message" in data:
                            logger.error(f"❌ API Error for {ticker}: {data['Error Message']}")
                            return []
                        else:
                            logger.warning(f"⚠️ Unexpected response format for {ticker}: {data}")
                            return []
                    elif response.status == 429:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"⚠️ Rate limit 429 for {ticker}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"❌ HTTP {response.status} for {ticker}")
                        return []
            except Exception as e:
                logger.error(f"❌ Request failed for {ticker}: {e}")
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                
        logger.error(f"❌ Max retries reached for {ticker}. Giving up.")
        return []

async def fetch_all_ratios(tickers):
    """
    Manages the asynchronous fetching of ratios for the entire list of tickers.
    """
    semaphore = asyncio.Semaphore(5) 
    all_ratios = []
    tasks = []
    
    async with aiohttp.ClientSession() as session:
        for ticker in tickers:
            task = fetch_ratios_for_ticker(session, semaphore, ticker)
            tasks.append(task)
        
        logger.info(f"🚀 Launching {len(tasks)} concurrent API requests to FMP...")
        results = await asyncio.gather(*tasks)
        
        for ratio_list in results:
            if ratio_list:
                all_ratios.extend(ratio_list)
                
    return all_ratios

def load_unique_tickers_from_history(spark):
    """
    Reads the consolidated history table from Delta Lake and returns a unique list of tickers.
    """
    logger.info("📡 Loading SP500_CONSOLIDATED_HISTORY from Delta Lake...")
    try:
        df_history = spark.read.format("delta").load(Paths.SP500_CONSOLIDATED_HISTORY)
        logger.info(f"✅ Loaded history table.")
        # We only need unique tickers to fetch ratios
        rows = df_history.select("Ticker").distinct().collect()
        tickers = [row["Ticker"] for row in rows if row["Ticker"] is not None]
        logger.info(f"✅ Found {len(tickers)} unique tickers.")
        return tickers
    except Exception as e:
        logger.error(f"❌ Error loading consolidated history: {e}")
        raise e

def save_ratios_to_lake(spark, pandas_df):
    """
    Saves the fetched ratios DataFrame to Delta Lake.
    """
    logger.info(f"💾 Saving {pandas_df.shape[0]} ratio records to {Paths.SP500_RATIOS}...")
    
    # Clean up pandas df to ensure proper Spark Schema inference without exceptions
    if 'symbol' in pandas_df.columns and 'date' in pandas_df.columns:
        pandas_df = pandas_df.dropna(subset=['symbol', 'date'])
        pandas_df['symbol'] = pandas_df['symbol'].astype(str)
        pandas_df['date'] = pandas_df['date'].astype(str)

    try:
        if pandas_df.empty:
            logger.warning("⚠️ Pandas DataFrame is empty. Skipping save.")
            return

        # Pre-process columns to avoid Spark type inference issues with mixed types
        # Convert all object columns to str
        for col_name in pandas_df.columns:
            if pandas_df[col_name].dtype == 'object':
                pandas_df[col_name] = pandas_df[col_name].astype(str)

        # Create DataFrame letting Spark infer the initial schema from strongly-typed Pandas DataFrame
        sdf = spark.createDataFrame(pandas_df)
        
        # Explicitly cast fundamental columns
        if "symbol" in sdf.columns:
            sdf = sdf.withColumn("symbol", col("symbol").cast(StringType()))
        if "date" in sdf.columns:
            sdf = sdf.withColumn("date", to_date(col("date")))
                 
        sdf.write.format("delta") \
            .mode("overwrite") \
            .option("mergeSchema", "true") \
            .save(Paths.SP500_RATIOS)
            
        logger.success("✅ Success! Ratios saved.")
        
    except Exception as e:
        logger.error(f"❌ Error saving ratios to Lake: {e}")

def main():
    setup_logging()
    logger.info("🚀 Starting Job: List SP500 Ratios Ingestion")

    spark = None

    try:
        spark = create_spark_session(app_name="List_SP500_Ratios")
        
        cache_file = 'local_ratios_cache.parquet'
        if os.path.exists(cache_file):
            logger.info(f"♻️ Found local cache {cache_file}. Loading from cache...")
            df_ratios = pd.read_parquet(cache_file)
        else:
            # 1. Load active tickers from Silver
            tickers = load_unique_tickers_from_history(spark)
            
            if not tickers:
                logger.warning("⚠️ No tickers found in history. Exiting.")
                return

            # 2. Asynchronously fetch ratios from FMP API
            logger.info("⏱️ Fetching ratios... This may take a few minutes.")
            raw_data = asyncio.run(fetch_all_ratios(tickers))
            
            if not raw_data:
                 logger.error("❌ No raw data fetched from FMP.")
                 return
                 
            # 3. Process into Pandas DataFrame
            df_ratios = pd.DataFrame(raw_data)
            logger.info(f"📊 Processed a total of {df_ratios.shape[0]} ratio rows.")
            
            # Drop duplicates if any overlaps occurred
            if 'symbol' in df_ratios.columns and 'date' in df_ratios.columns:
                df_ratios = df_ratios.drop_duplicates(subset=['symbol', 'date'])
            
            logger.info(f"💾 Saving to local cache {cache_file} before writing to Delta...")
            # Convert object columns to str to avoid mixed type issues
            for str_col in df_ratios.select_dtypes(include=['object']).columns:
                df_ratios[str_col] = df_ratios[str_col].astype(str)
                
            df_ratios.to_parquet(cache_file, index=False)
        
        # 4. Save to Bronze
        save_ratios_to_lake(spark, df_ratios)

    except Exception as e:
        logger.critical(f"❌ Critical Error: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
