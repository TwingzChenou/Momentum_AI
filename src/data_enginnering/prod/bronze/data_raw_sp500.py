import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
from loguru import logger
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType
from pyspark.sql.functions import col, to_date

# Force Spark match environment
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
            logger.info(f"📍 Last data in Lake for SP500: {max_date}")
            return str(max_date), True
    except Exception:
        logger.warning(f"⚠️ No existing table found at {path}. Full load required.")
        
    return None, False

def fetch_sp500(start_date=None, period="max"):
    """
    Télécharge l'historique du S&P 500 (^GSPC).
    Si start_date est fourni, télécharge depuis cette date.
    """
    ticker = "^GSPC"
    logger.info(f"🚀 Téléchargement pour {ticker} (Start: {start_date if start_date else period})...")
    
    try:
        if start_date:
            df = yf.download(
                tickers=ticker, 
                start=start_date, 
                end=datetime.today().strftime('%Y-%m-%d'),
                interval="1d", 
                auto_adjust=False, 
                progress=False
            )
        else:
            df = yf.download(
                tickers=ticker, 
                period=period, 
                interval="1d", 
                auto_adjust=False, 
                progress=False
            )
        
        if df.empty:
            return pd.DataFrame()
            
        df = df.reset_index()
        df['Ticker'] = ticker
        df = df.rename(columns={'Adj Close': 'AdjClose'})
        
        expected_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
        df = df[expected_cols]
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement: {e}")
        return pd.DataFrame()

def process_frequencies(df_raw):
    """
    Génère les versions Daily, Weekly et Monthly.
    """
    logger.info("🔧 Génération des fréquences Daily, Weekly et Monthly...")
    
    # Flatten columns if MultiIndex (sometimes yfinance returns 2 levels like ['Close', '^GSPC'])
    df_daily = df_raw.copy()
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)
    
    df_daily['Date'] = pd.to_datetime(df_daily['Date']).dt.tz_localize(None)
    df_daily = df_daily.sort_values('Date')
    ticker = "^GSPC"
    
    # 1. DAILY (Business Days)
    min_date, max_date = df_daily['Date'].min(), df_daily['Date'].max()
    master_dates_daily = pd.date_range(start=min_date, end=max_date, freq='B')
    master_df_daily = pd.DataFrame({'Date': master_dates_daily})
    master_df_daily['Ticker'] = ticker
    
    final_daily = pd.merge(master_df_daily, df_daily, on=['Date', 'Ticker'], how='left')
    
    # 2. WEEKLY (W-FRI)
    df_res = df_daily.set_index('Date')
    resampled_weekly = df_res.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'AdjClose': 'last',
        'Volume': 'sum'
    }).reset_index()
    resampled_weekly['Ticker'] = ticker
    
    # 3. MONTHLY (BME)
    resampled_monthly = df_res.resample('BME').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'AdjClose': 'last',
        'Volume': 'sum'
    }).reset_index()
    resampled_monthly['Ticker'] = ticker
    
    return final_daily, resampled_weekly, resampled_monthly

def save_to_lake(spark, pandas_df, path):
    """
    Sauvegarde en Delta Lake.
    """
    logger.info(f"💾 Sauvegarde de {len(pandas_df)} lignes vers {path}...")
    
    pandas_df['Date'] = pandas_df['Date'].dt.strftime('%Y-%m-%d')
    for col_name in ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']:
        pandas_df[col_name] = pd.to_numeric(pandas_df[col_name], errors='coerce').astype(float)

    try:
        sdf = spark.createDataFrame(pandas_df)
        sdf = sdf.withColumn("Date", to_date(col("Date"))) \
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
            
        logger.info(f"✅ Sauvegardé avec succès.")
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde: {e}")

def main():
    setup_logging()
    spark = create_spark_session(app_name="Ingestion_SP500_Bronze")
    
    try:
        from delta.tables import DeltaTable
        
        # 1. Check High Water Mark
        last_date, is_inc = get_max_date_from_lake(spark, Paths.DATA_RAW_SP500)
        
        # 2. Fetch Data
        df_raw = fetch_sp500(start_date=last_date, period="max")
        
        if not df_raw.empty:
            d, w, m = process_frequencies(df_raw)
            save_to_lake(spark, d, Paths.DATA_RAW_SP500)
            save_to_lake(spark, w, Paths.DATA_RAW_SP500_WEEKLY)
            save_to_lake(spark, m, Paths.DATA_RAW_SP500_MONTHLY)
        else:
            logger.info("💤 No new SP500 data to process.")
            
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
