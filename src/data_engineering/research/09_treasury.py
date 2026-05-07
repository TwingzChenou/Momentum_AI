import os
import sys
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql.types import DateType, FloatType, StringType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = "https://financialmodelingprep.com/stable"

async def fetch_treasury_chunk(session, semaphore, start_date, end_date):
    """
    Fetches treasury rates for a specific date range.
    """
    url = f"{BASE_URL}/treasury-rates?from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
    
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
                            logger.error(f"❌ API Error for {start_date} to {end_date}: {data['Error Message']}")
                            return []
                        else:
                            logger.warning(f"⚠️ Unexpected response format: {data}")
                            return []
                    elif response.status == 429:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"⚠️ Rate limit 429. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"❌ HTTP {response.status} for {start_date} to {end_date}")
                        return []
            except Exception as e:
                logger.error(f"❌ Request failed: {e}")
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                
        logger.error(f"❌ Max retries reached for {start_date} to {end_date}. Giving up.")
        return []

async def fetch_all_treasury_rates(start_date, end_date):
    """
    Manages fetching treasury rates in 90-day chunks to respect the 90-record limitation.
    """
    semaphore = asyncio.Semaphore(5) 
    all_rates = []
    tasks = []
    
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    final_end = datetime.strptime(end_date, "%Y-%m-%d")
    
    async with aiohttp.ClientSession() as session:
        while current_start <= final_end:
            # 90 calendar days guarantees <= 90 trading days (no weekends/holidays)
            current_end = current_start + timedelta(days=90)
            if current_end > final_end:
                current_end = final_end
                
            task = fetch_treasury_chunk(
                session, 
                semaphore, 
                current_start.strftime("%Y-%m-%d"), 
                current_end.strftime("%Y-%m-%d")
            )
            tasks.append(task)
            
            # Move to next chunk
            current_start = current_end + timedelta(days=1)
        
        logger.info(f"🚀 Launching {len(tasks)} concurrent API requests for chunks...")
        results = await asyncio.gather(*tasks)
        
        for rates_list in results:
            if rates_list:
                all_rates.extend(rates_list)
                
    return all_rates

def save_treasury_rates_to_lake(spark, pandas_df):
    """
    Saves the fetched treasury rates DataFrame to Delta Lake.
    """
    logger.info(f"💾 Saving {pandas_df.shape[0]} treasury records to {Paths.TREASURY_BOND_BRONZE}...")
    
    try:
        if pandas_df.empty:
            logger.warning("⚠️ Pandas DataFrame is empty. Skipping save.")
            return

        # Explicit type conversion in pandas before spark
        for col_name in pandas_df.columns:
            if col_name == 'date':
                pandas_df[col_name] = pandas_df[col_name].astype(str)
            else:
                pandas_df[col_name] = pd.to_numeric(pandas_df[col_name], errors='coerce')

        # Create DataFrame
        sdf = spark.createDataFrame(pandas_df)
        
        # Explicitly cast columns
        from pyspark.sql.functions import col, to_date
        
        for column_name in sdf.columns:
            if column_name == "date":
                sdf = sdf.withColumn("date", to_date(col("date")))
            else:
                sdf = sdf.withColumn(column_name, col(column_name).cast("float"))
                 
        sdf.coalesce(1).write.format("delta") \
            .mode("overwrite") \
            .option("mergeSchema", "true") \
            .save(Paths.TREASURY_BOND_BRONZE)
            
        logger.success("✅ Success! Treasury rates saved.")
        
    except Exception as e:
        logger.error(f"❌ Error saving treasury rates to Lake: {e}")


def main():
    setup_logging()
    logger.info("🚀 Starting Job: Ingest Treasury Rates")

    spark = None

    try:
        spark = create_spark_session(app_name="Ingest_Treasury_Rates")
        
        start_date = "1900-01-01"
        end_date = datetime.today().strftime("%Y-%m-%d")
        
        # Fetch data
        logger.info(f"⏱️ Fetching treasury rates from {start_date} to {end_date}...")
        raw_data = asyncio.run(fetch_all_treasury_rates(start_date, end_date))
        
        if not raw_data:
             logger.error("❌ No raw data fetched from FMP.")
             return
             
        # Process into Pandas DataFrame
        df_rates = pd.DataFrame(raw_data)
        logger.info(f"📊 Processed a total of {df_rates.shape[0]} treasury records.")
        
        # Drop duplicates and sort by date descending
        if 'date' in df_rates.columns:
            df_rates = df_rates.drop_duplicates(subset=['date'])
            df_rates = df_rates.sort_values(by='date', ascending=False)
            
        # Save to Bronze
        save_treasury_rates_to_lake(spark, df_rates)

    except Exception as e:
        logger.critical(f"❌ Critical Error: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
