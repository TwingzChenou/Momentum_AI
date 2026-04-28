import sys
import os
from loguru import logger

# Correct path handling for Airflow environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.setup_spark import create_spark_session
from src.common.validation import validate_df
from config.config_spark import Paths

def main():
    logger.info("🚀 Starting Bronze Data Validation...")
    spark = None
    try:
        spark = create_spark_session(app_name="Validation_Bronze")
        
        # 1. Validate SP500 Stock Prices
        logger.info(f"📥 Loading Bronze Stocks from {Paths.SP500_STOCK_PRICES}")
        df_stocks = spark.read.format("delta").load(Paths.SP500_STOCK_PRICES)
        
        if df_stocks.isEmpty():
            raise ValueError("❌ SP500_STOCK_PRICES table is empty!")
            
        validate_df(df_stocks, "bronze_stock_prices_suite")
        
        logger.success("✅ Bronze Validation Completed Successfully.")
        
    except Exception as e:
        logger.critical(f"❌ Bronze Validation Failed: {e}")
        sys.exit(1) # Ensure Airflow task fails
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()
