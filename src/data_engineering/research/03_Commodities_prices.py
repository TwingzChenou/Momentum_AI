import os
import sys
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType
from pyspark.sql.functions import col, to_date

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = "https://financialmodelingprep.com/stable/"

async def fetch_prices_for_period(session, semaphore, ticker, start_date, end_date):
    """
    Fetches the historical dividend-adjusted prices for a single ticker over a specific period.
    Limits concurrency using a semaphore and implements exponential backoff for 429 Rate Limits.
    """
    url = f"{BASE_URL}/historical-price-eod/full?symbol={ticker}&from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
    
    max_retries = 5
    base_delay = 2

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        # FMP usually returns a list of dictionaries.
                        if isinstance(data, list):
                            return data
                        elif "Error Message" in data:
                            logger.error(f"❌ API Error for {ticker} ({start_date} to {end_date}): {data['Error Message']}")
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
                        logger.error(f"❌ HTTP {response.status} for {ticker} ({start_date} to {end_date})")
                        return []
            except Exception as e:
                logger.error(f"❌ Request failed for {ticker}: {e}")
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                
        logger.error(f"❌ Max retries reached for {ticker} ({start_date} to {end_date}). Giving up.")
        return []

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

async def fetch_all_historical_prices(periods_list):
    """
    Manages the asynchronous fetching of prices for the entire list of (Ticker, Start_Date, End_Date).
    Splits periods larger than 13 years into chunks to avoid the 5000 max records API limit.
    """
    # Using a smaller concurrency to spread out requests
    semaphore = asyncio.Semaphore(5) 
    all_prices = []
    tasks = []
    
    async with aiohttp.ClientSession() as session:
        for period in periods_list:
            ticker = period['Ticker']
            start_date = period['Date_start']
            
            # If Date_end is today, use today
            end_date = period['Date_end'] if period['Date_end'] is not None else datetime.now().date()
            
            # Divide into chunks if necessary
            chunks = split_period(start_date, end_date)
            
            for chunk_start, chunk_end in chunks:
                start_str = chunk_start.strftime('%Y-%m-%d')
                end_str = chunk_end.strftime('%Y-%m-%d')
                task = fetch_prices_for_period(session, semaphore, ticker, start_str, end_str)
                tasks.append(task)
        
        # Gather all results concurrently
        logger.info(f"🚀 Launching {len(tasks)} concurrent API chunked requests to FMP...")
        results = await asyncio.gather(*tasks)
        
        # Flatten the list of lists
        for prices_list in results:
            if prices_list:
                all_prices.extend(prices_list)
                
    return all_prices

def load_consolidated_history(spark):
    """
    Reads the commodities list table from Delta Lake and returns a local list of dicts.
    """
    logger.info("📡 Loading COMMODITIES_LIST_TICKERS from Delta Lake...")
    try:
        df_history = spark.read.format("delta").load(Paths.COMMODITIES_LIST_TICKERS)
        logger.info(f"✅ Loaded {df_history.count()} commodities.")
        # Collect to driver since we will make Python HTTP requests
        rows = df_history.collect()
        
        periods = []
        start_date = datetime(1990, 1, 1).date()
        end_date = datetime.now().date()
        for row in rows:
            if 'symbol' in row and row['symbol']:
                 periods.append({
                     'Ticker': row['symbol'],
                     'Date_start': start_date,
                     'Date_end': end_date
                 })
        return periods
    except Exception as e:
        logger.error(f"❌ Error loading commodities list: {e}")
        raise e

def save_prices_to_lake(spark, pandas_df):
    """
    Saves the fetched prices DataFrame to Delta Lake.
    """
    logger.info(f"💾 Saving {pandas_df.shape[0]} price records to {Paths.COMMODITIES_STOCK_PRICES}...")
    
    # Clean up pandas df to ensure proper Spark Schema inference without NullPointerExceptions
    pandas_df = pandas_df.dropna(subset=['symbol', 'date'])
    pandas_df['symbol'] = pandas_df['symbol'].astype(str)
    pandas_df['date'] = pandas_df['date'].astype(str)
    
    pandas_df['open'] = pd.to_numeric(pandas_df['open'], errors='coerce').fillna(0.0)
    pandas_df['high'] = pd.to_numeric(pandas_df['high'], errors='coerce').fillna(0.0)
    pandas_df['low'] = pd.to_numeric(pandas_df['low'], errors='coerce').fillna(0.0)
    pandas_df['close'] = pd.to_numeric(pandas_df['close'], errors='coerce').fillna(0.0)
    pandas_df['volume'] = pd.to_numeric(pandas_df['volume'], errors='coerce').fillna(0).astype('int64')

    try:
        # Create DataFrame letting Spark infer the initial schema from strongly-typed Pandas DataFrame
        sdf = spark.createDataFrame(pandas_df)
        
        # Explicitly cast to ensure correct Delta Lake types matching
        sdf = sdf.withColumn("symbol", col("symbol").cast(StringType())) \
                 .withColumn("date", to_date(col("date"))) \
                 .withColumn("open", col("open").cast(DoubleType())) \
                 .withColumn("high", col("high").cast(DoubleType())) \
                 .withColumn("low", col("low").cast(DoubleType())) \
                 .withColumn("close", col("close").cast(DoubleType())) \
                 .withColumn("volume", col("volume").cast(LongType()))
                 
        sdf.write.format("delta") \
            .mode("overwrite") \
            .option("mergeSchema", "true") \
            .save(Paths.COMMODITIES_STOCK_PRICES)
            
        logger.success("✅ Success! Historical Prices saved.")
        
    except Exception as e:
        logger.error(f"❌ Error saving prices to Lake: {e}")

def main():
    setup_logging()
    logger.info("🚀 Starting Job: Commodities Historical Prices Ingestion")

    spark = None

    try:
        spark = create_spark_session(app_name="Commodities_Prices_Ingestion")
        
        cache_file = 'local_commodities_prices_cache.parquet'
        if os.path.exists(cache_file):
            logger.info(f"♻️ Found local cache {cache_file}. Loading from cache...")
            df_prices = pd.read_parquet(cache_file)
        else:
            # 1. Load active intervals from Silver
            periods_list = load_consolidated_history(spark)
            
            if not periods_list:
                logger.warning("⚠️ No periods found in history. Exiting.")
                return

            # 2. Asynchronously fetch prices from FMP API
            logger.info("⏱️ Fetching historical prices... This may take a few minutes.")
            raw_prices = asyncio.run(fetch_all_historical_prices(periods_list))
            
            if not raw_prices:
                 logger.error("❌ No raw prices fetched from FMP.")
                 return
                 
            # 3. Process into Pandas DataFrame
            df_prices = pd.DataFrame(raw_prices)
            
            # Ensure only relevant columns are kept (in case API returns extra fields)
            expected_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            # Filter columns to only those expected
            existing_cols = [c for c in expected_cols if c in df_prices.columns]
            df_prices = df_prices[existing_cols]

            logger.info(f"📊 Processed a total of {df_prices.shape[0]} price rows.")
            
            # Drop duplicates if any overlaps occurred
            df_prices = df_prices.drop_duplicates(subset=['symbol', 'date'])
            
            logger.info(f"💾 Saving to local cache {cache_file} before writing to Delta...")
            df_prices.to_parquet(cache_file, index=False)
        
        # 4. Save to Bronze
        save_prices_to_lake(spark, df_prices)

    except Exception as e:
        logger.critical(f"❌ Critical Error: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()

