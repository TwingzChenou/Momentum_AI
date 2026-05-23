import os
import sys
from loguru import logger
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths
import pyspark.sql.functions as F

def check_bronze_and_history():
    logger.info("🚀 Starting GCS Bronze and History Audit for short tickers...")
    spark = create_spark_session("Bronze_History_Audit")
    try:
        short_tickers = ["DOW", "HCA", "DELL", "KMI", "AMP", "HLT", "BR", "DG", "TT", "DD"]
        
        # 1. Load History
        logger.info(f"📂 Checking {Paths.SP500_CONSOLIDATED_HISTORY}...")
        df_history = spark.read.format("delta").load(Paths.SP500_CONSOLIDATED_HISTORY)
        
        logger.info("--- HISTORY DATA FOR SHORT TICKERS ---")
        df_hist_short = df_history.filter(F.col("Ticker").isin(short_tickers))
        df_hist_short.show(truncate=False)
        
        # 2. Load Bronze prices
        logger.info(f"📂 Checking {Paths.SP500_STOCK_PRICES} (Bronze Daily Prices)...")
        df_bronze = spark.read.format("delta").load(Paths.SP500_STOCK_PRICES)
        
        logger.info("--- BRONZE ROW COUNTS AND DATE RANGES ---")
        for ticker in short_tickers:
            df_ticker_prices = df_bronze.filter(F.col("Ticker") == ticker)
            count = df_ticker_prices.count()
            if count > 0:
                stats = df_ticker_prices.select(F.min("Date"), F.max("Date")).collect()[0]
                logger.success(f"📈 {ticker} in Bronze: {count} rows. Date range: {stats[0]} to {stats[1]}")
            else:
                logger.error(f"❌ {ticker} in Bronze: 0 rows found!")
                
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    check_bronze_and_history()
