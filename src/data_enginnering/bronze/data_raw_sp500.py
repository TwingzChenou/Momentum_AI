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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def fetch_sp500_max():
    """
    Télécharge l'historique MAX du S&P 500 (^GSPC).
    """
    ticker = "^GSPC"
    logger.info(f"🚀 Téléchargement de l'historique MAX pour {ticker}...")
    
    try:
        df = yf.download(
            tickers=ticker, 
            period="max", 
            interval="1d", 
            auto_adjust=False, 
            progress=False
        )
        
        if df.empty:
            logger.error("❌ Données vides retournées par yfinance.")
            return pd.DataFrame()
            
        df = df.reset_index()
        df['Ticker'] = ticker
        
        # Renommage pour compatibilité Delta
        df = df.rename(columns={'Adj Close': 'AdjClose'})
        
        # Nettoyage colonnes
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
                 
        sdf.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(path)
        logger.info(f"✅ Sauvegardé avec succès.")
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde: {e}")

def main():
    setup_logging()
    spark = create_spark_session(app_name="Ingestion_SP500_Bronze")
    
    try:
        df_raw = fetch_sp500_max()
        if not df_raw.empty:
            d, w, m = process_frequencies(df_raw)
            save_to_lake(spark, d, Paths.DATA_RAW_SP500)
            save_to_lake(spark, w, Paths.DATA_RAW_SP500_WEEKLY)
            save_to_lake(spark, m, Paths.DATA_RAW_SP500_MONTHLY)
            
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
