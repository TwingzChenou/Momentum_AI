import os
import sys
from loguru import logger
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths
import pyspark.sql.functions as F

def check_silver():
    logger.info("🚀 Starting GCS Silver Data Audit...")
    spark = create_spark_session("Silver_Data_Audit")
    try:
        paths_to_check = {
            "SP500 Daily Silver": Paths.SP500_STOCK_PRICES_SILVER,
            "SP500 Weekly Silver": Paths.SP500_STOCK_PRICES_WEEKLY_SILVER,
            "DATA RAW 2B Weekly Silver": Paths.DATA_RAW_2B_WEEKLY_SILVER,
            "DATA RAW ETF Weekly Silver": Paths.DATA_RAW_ETF_WEEKLY_SILVER,
            "DATA RAW SP500 Weekly Silver": Paths.DATA_RAW_SP500_WEEKLY_SILVER,
        }
        
        for name, path in paths_to_check.items():
            logger.info(f"----------------------------------------")
            logger.info(f"🔍 Auditing table: {name} ({path})")
            try:
                df = spark.read.format("delta").load(path)
                
                total_rows = df.count()
                distinct_tickers = df.select("Ticker").distinct().count()
                
                # Check date range
                date_stats = df.select(F.min("Date"), F.max("Date")).collect()[0]
                min_date = date_stats[0]
                max_date = date_stats[1]
                
                logger.success(f"✅ {name} : {total_rows} total rows, {distinct_tickers} distinct tickers.")
                logger.info(f"   📅 Date Range: {min_date} to {max_date}")
                
                # Check tickers with fewest rows
                logger.info("   📈 Tickers with fewest rows:")
                df_counts = df.groupBy("Ticker").count().orderBy("count")
                fewest = df_counts.limit(10).collect()
                for row in fewest:
                    logger.warning(f"      - {row['Ticker']}: {row['count']} rows")
                
            except Exception as e:
                logger.error(f"❌ Failed to load or audit {name}: {e}")
                
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    check_silver()
