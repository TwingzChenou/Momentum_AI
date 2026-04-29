import sys
import os
from loguru import logger

# Correct path handling for Airflow environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.setup_spark import create_spark_session
from src.common.validation import validate_df
from config.config_spark import Paths

def main():
    logger.info("🚀 Starting Silver Data Validation...")
    spark = None
    try:
        spark = create_spark_session(app_name="Validation_Silver")
        
        # 1. Validate SP500 Stock Prices Weekly (Silver)
        logger.info(f"📥 Loading Silver Weekly Bronze from {Paths.DATA_RAW_2B_WEEKLY_SILVER}")
        df_silver = spark.read.format("delta").load(Paths.DATA_RAW_2B_WEEKLY_SILVER)
        
        if df_silver.isEmpty():
            raise ValueError("❌ DATA_RAW_2B_WEEKLY_SILVER table is empty!")
            
        validate_df(df_silver, "silver_weekly_prices_suite")
        
        logger.success("✅ Silver Validation Completed Successfully.")
        
    except Exception as e:
        logger.critical(f"❌ Silver Validation Failed: {e}")
        sys.exit(1) # Ensure Airflow task fails
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()
