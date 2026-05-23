import os
import sys
from loguru import logger
from pyspark.sql.functions import col, to_date, lit, coalesce

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def rebuild_history():
    logger.info("🛠️ Starting Standalone Emergency Recovery of S&P 500 Consolidated History...")
    
    spark = create_spark_session("Emergency_SP500_History_Rebuild")
    
    try:
        # 1. Load the latest S&P 500 constituents from Delta Lake
        logger.info(f"📂 Loading latest constituents from {Paths.SP500_LATEST_TICKERS}...")
        df_latest = spark.read.format("delta").load(Paths.SP500_LATEST_TICKERS)
        
        # 2. Reconstruct history: map Symbol to Date_start, set Date_end to NULL
        logger.info("⚙️ Mapping constituents to original addition dates...")
        
        df_rebuilt = df_latest.select(
            col("symbol").alias("Ticker"),
            coalesce(to_date(col("dateFirstAdded")), to_date(lit("1970-01-01"))).alias("Date_start"),
            lit(None).cast("date").alias("Date_end")
        ).distinct()
        
        total_rows = df_rebuilt.count()
        logger.info(f"📊 Rebuilt table will have {total_rows} active S&P 500 constituents.")
        
        # Show a preview
        df_rebuilt.show(15, truncate=False)
        
        # 3. Overwrite Paths.SP500_CONSOLIDATED_HISTORY with correct history
        logger.warning(f"💾 OVERWRITING {Paths.SP500_CONSOLIDATED_HISTORY} with original dates...")
        df_rebuilt.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(Paths.SP500_CONSOLIDATED_HISTORY)
            
        logger.success("✅ SUCCESS! S&P 500 Consolidated History rebuilt and saved to GCS.")
        
    except Exception as e:
        logger.critical(f"❌ Rebuild failed: {e}")
        raise e
    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    rebuild_history()
