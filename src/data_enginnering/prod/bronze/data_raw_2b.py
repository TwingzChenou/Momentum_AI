import os
import sys
import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType
from pyspark.sql.functions import col, to_date

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def get_tickers_from_lake(spark):
    """
    Reads the tickers list from LIST_TICKER_2B and returns a python list of symbols.
    """
    logger.info(f"📡 Loading tickers from {Paths.LIST_TICKER_2B}...")
    try:
        df = spark.read.format("delta").load(Paths.LIST_TICKER_2B)
        symbols = [row['symbol'] for row in df.select('symbol').distinct().collect()]
        logger.info(f"✅ Loaded {len(symbols)} unique tickers.")
        return symbols
    except Exception as e:
        logger.error(f"❌ Error loading tickers: {e}")
        return []

def fetch_data_in_chunks(tickers, period="2y", chunk_size=100):
    """
    Fetches daily data from yfinance in chunks to avoid rate limits,
    then stacks it into a flat structure.
    """
    logger.info(f"🚀 Fetching data for {len(tickers)} tickers in chunks of {chunk_size}...")
    
    all_data = []
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        logger.info(f"⏱️ Fetching chunk {i // chunk_size + 1}/{(len(tickers) - 1) // chunk_size + 1} ({len(chunk)} tickers)...")
        
        try:
            # group_by='ticker' ensures Level 0 is Ticker, Level 1 is Price
            df = yf.download(
                tickers=chunk, 
                period=period, 
                interval="1d", 
                group_by="ticker", 
                auto_adjust=False, 
                progress=False,
                threads=True
            )
            
            if df.empty:
                logger.warning("⚠️ Empty DataFrame returned for this chunk.")
                continue
                
            # If only 1 ticker in chunk, yfinance doesn't use MultiIndex on columns
            if len(chunk) == 1:
                df['Ticker'] = chunk[0]
                df = df.reset_index()
            else:
                # Stack the Ticker level (level=0)
                df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
                
            all_data.append(df)
            
            # Sleep to prevent rapid rate limiting
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"❌ Error fetching chunk {chunk}: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure columns exist even if some are missing
    expected_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col_name in expected_cols:
        if col_name not in final_df.columns:
            final_df[col_name] = pd.NA
            
    final_df = final_df[expected_cols]
    final_df = final_df.rename(columns={'Adj Close': 'AdjClose'})
    return final_df

def process_data(df_daily, all_tickers):
    """
    Cleans daily data and generates both daily and weekly (W-FRI) master-indexed DataFrames.
    """
    logger.info("🔧 Processing daily data and resampling to W-FRI...")
    
    # Clean Date column
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    
    # Optional: clean numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    for col in numeric_cols:
        df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
        
    if not df_daily.empty:
        min_date = df_daily['Date'].min()
        max_date = df_daily['Date'].max()
    else:
        # Fallback
        max_date = pd.Timestamp.today()
        min_date = max_date - pd.DateOffset(years=2)

    # --- 1. DAILY PIPELINE ---
    master_dates_daily = pd.date_range(start=min_date, end=max_date, freq='B') # Business days
    master_df_daily = pd.MultiIndex.from_product([all_tickers, master_dates_daily], names=['Ticker', 'Date']).to_frame(index=False)
    final_daily_df = pd.merge(master_df_daily, df_daily, on=['Ticker', 'Date'], how='left')
    
    # --- 2. WEEKLY PIPELINE ---
    df_for_weekly = df_daily.set_index('Date')
    resampled = df_for_weekly.groupby('Ticker').resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'AdjClose': 'last',
        'Volume': 'sum'
    }).reset_index()

    master_dates_weekly = pd.date_range(start=min_date, end=max_date, freq='W-FRI')
    master_df_weekly = pd.MultiIndex.from_product([all_tickers, master_dates_weekly], names=['Ticker', 'Date']).to_frame(index=False)
    final_weekly_df = pd.merge(master_df_weekly, resampled, on=['Ticker', 'Date'], how='left')
    
    # --- 3. MONTHLY PIPELINE ---
    resampled_monthly = df_for_weekly.groupby('Ticker').resample('BME').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'AdjClose': 'last',
        'Volume': 'sum'
    }).reset_index()

    master_dates_monthly = pd.date_range(start=min_date, end=max_date, freq='BME')
    master_df_monthly = pd.MultiIndex.from_product([all_tickers, master_dates_monthly], names=['Ticker', 'Date']).to_frame(index=False)
    final_monthly_df = pd.merge(master_df_monthly, resampled_monthly, on=['Ticker', 'Date'], how='left')
    
    logger.info(f"✅ Final shape Daily: {final_daily_df.shape} | Weekly: {final_weekly_df.shape} | Monthly: {final_monthly_df.shape}")
    return final_daily_df, final_weekly_df, final_monthly_df

def save_to_lake(spark, pandas_df, path):
    """
    Saves the processed pandas DataFrame to Delta Lake.
    """
    logger.info(f"💾 Saving {pandas_df.shape[0]} rows to {path}...")
    
    # Ensure safe types for Spark conversion
    pandas_df['Date'] = pandas_df['Date'].dt.strftime('%Y-%m-%d')
    pandas_df['Ticker'] = pandas_df['Ticker'].astype(str)
    
    # Fill NAs explicitly for numerics or keep them as None
    for col_name in ['Open', 'High', 'Low', 'Close', 'AdjClose']:
        pandas_df[col_name] = pandas_df[col_name].astype(float)
        
    # Using Int64 (nullable int) to avoid Float conversion of NAs if possible, 
    # but for Spark schema, it's safer to keep it Float if missing, or handle null logic.
    pandas_df['Volume'] = pandas_df['Volume'].astype(float)

    try:
        # Create DataFrame from Pandas
        sdf = spark.createDataFrame(pandas_df)
        
        # Explicit Casting
        sdf = sdf.withColumn("Date", to_date(col("Date"))) \
                 .withColumn("Ticker", col("Ticker").cast(StringType())) \
                 .withColumn("Open", col("Open").cast(DoubleType())) \
                 .withColumn("High", col("High").cast(DoubleType())) \
                 .withColumn("Low", col("Low").cast(DoubleType())) \
                 .withColumn("Close", col("Close").cast(DoubleType())) \
                 .withColumn("AdjClose", col("AdjClose").cast(DoubleType())) \
                 .withColumn("Volume", col("Volume").cast(LongType()))
                 
        sdf.write.format("delta") \
            .mode("overwrite") \
            .option("mergeSchema", "true") \
            .save(path)
            
        logger.info(f"✅ Success! Data saved to Delta Lake at {path}.")
        
    except Exception as e:
        logger.error(f"❌ Error saving raw data to Lake: {e}")

def main():
    setup_logging()
    logger.info("🚀 Starting Job: Fetch YF Data for 2B Tickers")

    spark = None

    try:
        spark = create_spark_session(app_name="Data_Raw_2B_Ingestion")
        
        # 1. Get Tickers
        tickers = get_tickers_from_lake(spark)
        
        if not tickers:
            logger.warning("⚠️ No tickers found. Exiting.")
            return
            
        # 2. Fetch Data in Chunks
        df_daily = fetch_data_in_chunks(tickers, period="2y", chunk_size=100)
        
        if df_daily.empty:
            logger.error("❌ No data retrieved from yfinance.")
            return

        # 3. Process and Align
        df_daily_master, df_weekly_master, df_monthly_master = process_data(df_daily, tickers)
        
        # 4. Save to Delta
        save_to_lake(spark, df_daily_master, Paths.DATA_RAW_2B)
        save_to_lake(spark, df_weekly_master, Paths.DATA_RAW_2B_WEEKLY)
        save_to_lake(spark, df_monthly_master, Paths.DATA_RAW_2B_MONTHLY)

    except Exception as e:
        logger.critical(f"❌ Critical Error in job execution: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
