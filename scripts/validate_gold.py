import sys
import os
from loguru import logger

# Correct path handling for Airflow environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.setup_spark import create_spark_session
from src.common.validation import validate_spark_df
from config.config_spark import Paths

def main():
    logger.info("🚀 Starting Gold Data Validation...")
    spark = None
    try:
        spark = create_spark_session(app_name="Validation_Gold")
        
        # 1. Validate Gold Stocks (Delta Cache)
        logger.info(f"📥 Loading Gold Stocks from {Paths.DATA_RAW_2B_WEEKLY_GOLD}")
        df_gold = spark.read.format("delta").load(Paths.DATA_RAW_2B_WEEKLY_GOLD)
        
        if df_gold.isEmpty():
            raise ValueError("❌ DATA_RAW_2B_WEEKLY_GOLD table is empty!")
            
        validate_spark_df(df_gold, "gold_momentum_features_suite")
        
        logger.success("✅ Gold Validation Completed Successfully.")
        
    except Exception as e:
        logger.critical(f"❌ Gold Validation Failed: {e}")
        sys.exit(1) # Ensure Airflow task fails
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()
