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

def get_max_date_from_lake(spark, path):
    """
    Checks the last available date in the Delta table for incremental loading.
    Returns (max_date_str, is_incremental)
    """
    try:
        df = spark.read.format("delta").load(path)
        max_date = df.selectExpr("max(Date)").collect()[0][0]
        if max_date:
            logger.info(f"📍 Last data in Lake: {max_date}")
            return str(max_date), True
    except Exception:
        logger.warning(f"⚠️ No existing table found at {path}. Full load required.")
        
    return None, False

ETF_TICKERS = [
    'XLP', 'XLV', 'XLU', 'XLE', 'XLK', 'XLC', 'XLI', 'XLY', 
    'XLB', 'XLRE', 'TLT', 'IEF', 'HYG', 'GLD', 'VNQ'
]

def fetch_data_in_chunks(tickers, start_date=None, period="max", chunk_size=100):
    """
    Fetches daily data from yfinance in chunks.
    If start_date is provided, uses start/end instead of period.
    """
    logger.info(f"🚀 Fetching data for {len(tickers)} ETFs (Start: {start_date if start_date else period})")
    
    all_data = []
    
    # Calculate end as today
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        logger.info(f"⏱️ Fetching chunk {i // chunk_size + 1}/{(len(tickers) - 1) // chunk_size + 1}...")
        
        try:
            if start_date:
                df = yf.download(
                    tickers=chunk, 
                    start=start_date,
                    end=end_date,
                    interval="1d", 
                    group_by="ticker", 
                    auto_adjust=False, 
                    progress=False,
                    threads=True
                )
            else:
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
                continue
                
            if len(chunk) == 1:
                df['Ticker'] = chunk[0]
                df = df.reset_index()
            else:
                df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
                
            all_data.append(df)
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Error fetching chunk: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    final_df = pd.concat(all_data, ignore_index=True)
    expected_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col_name in expected_cols:
        if col_name not in final_df.columns:
            final_df[col_name] = pd.NA
            
    final_df = final_df[expected_cols]
    final_df = final_df.rename(columns={'Adj Close': 'AdjClose'})
    return final_df

def process_data(df_daily, all_tickers):
    """
    Cleans daily data and generates daily, weekly (W-FRI), and monthly (BME) master-indexed DataFrames.
    """
    logger.info("🔧 Processing daily ETF data and resampling...")
    
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    for col in numeric_cols:
        df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
        
    if not df_daily.empty:
        min_date = df_daily['Date'].min()
        max_date = df_daily['Date'].max()
    else:
        max_date = pd.Timestamp.today()
        min_date = max_date - pd.DateOffset(years=2)

    # --- 1. DAILY PIPELINE ---
    master_dates_daily = pd.date_range(start=min_date, end=max_date, freq='B')
    master_df_daily = pd.MultiIndex.from_product([all_tickers, master_dates_daily], names=['Ticker', 'Date']).to_frame(index=False)
    final_daily_df = pd.merge(master_df_daily, df_daily, on=['Ticker', 'Date'], how='left')
    
    # --- 2. WEEKLY PIPELINE ---
    df_for_resampling = df_daily.set_index('Date')
    resampled_weekly = df_for_resampling.groupby('Ticker').resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'AdjClose': 'last',
        'Volume': 'sum'
    }).reset_index()

    master_dates_weekly = pd.date_range(start=min_date, end=max_date, freq='W-FRI')
    master_df_weekly = pd.MultiIndex.from_product([all_tickers, master_dates_weekly], names=['Ticker', 'Date']).to_frame(index=False)
    final_weekly_df = pd.merge(master_df_weekly, resampled_weekly, on=['Ticker', 'Date'], how='left')
    
    # --- 3. MONTHLY PIPELINE ---
    resampled_monthly = df_for_resampling.groupby('Ticker').resample('BM').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'AdjClose': 'last',
        'Volume': 'sum'
    }).reset_index()

    master_dates_monthly = pd.date_range(start=min_date, end=max_date, freq='BM')
    master_df_monthly = pd.MultiIndex.from_product([all_tickers, master_dates_monthly], names=['Ticker', 'Date']).to_frame(index=False)
    final_monthly_df = pd.merge(master_df_monthly, resampled_monthly, on=['Ticker', 'Date'], how='left')
    
    logger.info(f"✅ Final shape Daily: {final_daily_df.shape} | Weekly: {final_weekly_df.shape} | Monthly: {final_monthly_df.shape}")
    return final_daily_df, final_weekly_df, final_monthly_df

def save_to_lake(spark, pandas_df, path):
    """
    Saves the processed pandas DataFrame to Delta Lake.
    """
    if pandas_df.empty:
        logger.warning(f"⚠️ Skipping save to {path}: DataFrame is empty.")
        return

    logger.info(f"💾 Saving {pandas_df.shape[0]} rows to {path}...")
    
    pandas_df['Date'] = pandas_df['Date'].dt.strftime('%Y-%m-%d')
    pandas_df['Ticker'] = pandas_df['Ticker'].astype(str)
    
    for col_name in ['Open', 'High', 'Low', 'Close', 'AdjClose']:
        pandas_df[col_name] = pandas_df[col_name].astype(float)
        
    pandas_df['Volume'] = pandas_df['Volume'].astype(float)

    try:
        sdf = spark.createDataFrame(pandas_df)
        
        sdf = sdf.withColumn("Date", to_date(col("Date"))) \
                 .withColumn("Ticker", col("Ticker").cast(StringType())) \
                 .withColumn("Open", col("Open").cast(DoubleType())) \
                 .withColumn("High", col("High").cast(DoubleType())) \
                 .withColumn("Low", col("Low").cast(DoubleType())) \
                 .withColumn("Close", col("Close").cast(DoubleType())) \
                 .withColumn("AdjClose", col("AdjClose").cast(DoubleType())) \
                 .withColumn("Volume", col("Volume").cast(LongType()))
                 
        # --- DELTA MERGE FOR DATA PRESERVATION ---
        from delta.tables import DeltaTable
        if DeltaTable.isDeltaTable(spark, path):
            logger.info(f"🔄 Merging into existing Delta table at {path}...")
            dt = DeltaTable.forPath(spark, path)
            dt.alias("target").merge(
                sdf.alias("source"),
                "target.Date = source.Date AND target.Ticker = source.Ticker"
            ).whenMatchedUpdateAll() \
             .whenNotMatchedInsertAll() \
             .execute()
        else:
            logger.info(f"🆕 Creating new Delta table at {path}...")
            sdf.write.format("delta").mode("overwrite").save(path)
            
        logger.info(f"✅ Success! Data saved to {path}.")
        
    except Exception as e:
        logger.error(f"❌ Error saving raw data to Lake: {e}")
        raise e

def main():
    setup_logging()
    logger.info("🚀 Starting Job: Fetch YF Data for ETFs (MAX period)")

    spark = None

    try:
        spark = create_spark_session(app_name="Data_Raw_ETF_Ingestion")
        from delta.tables import DeltaTable
        
        # 1. Check High Water Mark
        last_date, is_inc = get_max_date_from_lake(spark, Paths.DATA_RAW_ETF)
        
        # 2. Fetch Data (Fetch last 60 days to ensure weekly/monthly resampling catch-up)
        fetch_start = None
        if last_date:
            from datetime import timedelta
            fetch_start = (datetime.strptime(last_date, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')
            logger.info(f"🔄 Incremental mode: fetching from {fetch_start} (60 days before last date {last_date})")

        df_daily = fetch_data_in_chunks(ETF_TICKERS, start_date=fetch_start, period="max", chunk_size=100)
        
        if df_daily.empty:
            logger.info("💤 No new ETF data to process.")
            return

        # 3. Process and Align
        df_daily_master, df_weekly_master, df_monthly_master = process_data(df_daily, ETF_TICKERS)
        
        # 4. Save/Merge to Lake
        save_to_lake(spark, df_daily_master, Paths.DATA_RAW_ETF)
        save_to_lake(spark, df_weekly_master, Paths.DATA_RAW_ETF_WEEKLY)
        save_to_lake(spark, df_monthly_master, Paths.DATA_RAW_ETF_MONTHLY)

    except Exception as e:
        logger.critical(f"❌ Critical Error in job execution: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
