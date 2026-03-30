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

async def fetch_crash_prices(session, semaphore, ticker, start_date, end_date):
    url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={ticker}&from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    
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
                            for row in data:
                                if 'symbol' not in row:
                                    row['symbol'] = ticker
                            return data
                        elif isinstance(data, dict) and "historical" in data:
                            res = data["historical"]
                            for row in res:
                                row['symbol'] = data.get('symbol', ticker)
                            return res
                        elif isinstance(data, dict) and "Error Message" in data:
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

async def fetch_all_crash_data(tickers):
    all_data = []
    semaphore = asyncio.Semaphore(5)
    start_date = datetime(1950, 1, 1).date()
    end_date = datetime.now().date()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for ticker in tickers:
            chunks = split_period(start_date, end_date)
            for chunk_start, chunk_end in chunks:
                start_str = chunk_start.strftime('%Y-%m-%d')
                end_str = chunk_end.strftime('%Y-%m-%d')
                tasks.append(fetch_crash_prices(session, semaphore, ticker, start_str, end_str))
                
        results = await asyncio.gather(*tasks)
        for res in results:
            if res:
                all_data.extend(res)
    return all_data

def save_to_lake(spark, pandas_df):
    logger.info(f"💾 Saving {pandas_df.shape[0]} records to {Paths.SP500_CRASH_BRONZE}...")
    
    # Keep expected columns
    expected_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'adjClose', 'volume']
    existing_cols = [c for c in expected_cols if c in pandas_df.columns]
    pandas_df = pandas_df[existing_cols]

    pandas_df = pandas_df.dropna(subset=['symbol', 'date'])
    pandas_df['symbol'] = pandas_df['symbol'].astype(str)
    pandas_df['date'] = pandas_df['date'].astype(str)
    
    for c in ['open', 'high', 'low', 'close', 'adjClose']:
        if c in pandas_df.columns:
            pandas_df[c] = pd.to_numeric(pandas_df[c], errors='coerce').fillna(0.0)
    
    if 'volume' in pandas_df.columns:
        pandas_df['volume'] = pd.to_numeric(pandas_df['volume'], errors='coerce').fillna(0).astype('int64')

    try:
        sdf = spark.createDataFrame(pandas_df)
        
        sdf = sdf.withColumn("symbol", col("symbol").cast(StringType())) \
                 .withColumn("date", to_date(col("date")))
                 
        for c in ['open', 'high', 'low', 'close', 'adjClose']:
             if c in sdf.columns:
                 sdf = sdf.withColumn(c, col(c).cast(DoubleType()))
        
        if 'volume' in sdf.columns:
             sdf = sdf.withColumn("volume", col("volume").cast(LongType()))

        sdf.write.format("delta") \
            .mode("overwrite") \
            .option("mergeSchema", "true") \
            .save(Paths.SP500_CRASH_BRONZE)
            
        logger.success("✅ Success! Crash Data saved.")
        
    except Exception as e:
        logger.error(f"❌ Error saving to Lake: {e}")

def main():
    setup_logging()
    logger.info("🚀 Starting Job: List Crash (VIX, VIX3M) Data Ingestion")
    
    tickers = ['^VIX']

    spark = None
    try:
        spark = create_spark_session(app_name="Crash_Data_Ingestion")
        
        raw_data = asyncio.run(fetch_all_crash_data(tickers))
        
        if not raw_data:
             logger.error("❌ No raw data fetched from FMP.")
             return
             
        df = pd.DataFrame(raw_data)
        logger.info(f"📊 Processed a total of {df.shape[0]} rows.")
        
        df = df.drop_duplicates(subset=['symbol', 'date'])
        
        save_to_lake(spark, df)

    except Exception as e:
        logger.critical(f"❌ Critical Error: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
