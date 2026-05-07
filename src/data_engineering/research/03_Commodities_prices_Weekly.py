import os
import sys
from loguru import logger
from pyspark.sql.functions import col, date_trunc
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def main():
    setup_logging()
    logger.info("🚀 Starting Job: Commodities Prices Weekly Resampling")
    spark = create_spark_session(app_name="Commodities_Prices_Weekly")

    try:
        # 1. Load Data
        logger.info(f"📡 Loading daily prices from {Paths.COMMODITIES_STOCK_PRICES}...")
        df_daily = spark.read.format("delta").load(Paths.COMMODITIES_STOCK_PRICES)
        
        # 2. Add WEEKLY Truncation Date
        df_daily = df_daily.withColumn("year_week", date_trunc("week", col("date")))
        
        # 3. Weekly Aggregations (OHLCV)
        logger.info("🔄 Aggregating to WEEKLY features (OHLCV)...")
        
        # Windows to find weekly min/max and orderings
        w_week = Window.partitionBy("symbol", "year_week")
        w_week_asc = Window.partitionBy("symbol", "year_week").orderBy("date")
        w_week_desc = Window.partitionBy("symbol", "year_week").orderBy(col("date").desc())
        
        df_weekly = df_daily \
            .withColumn("max_high", F.max("high").over(w_week)) \
            .withColumn("min_low", F.min("low").over(w_week)) \
            .withColumn("sum_volume", F.sum("volume").over(w_week)) \
            .withColumn("rn_asc", F.row_number().over(w_week_asc)) \
            .withColumn("rn_desc", F.row_number().over(w_week_desc))
            
        # Extract weekly close (from the last trading day)
        df_close = df_weekly.filter(col("rn_desc") == 1).select(
            "symbol", "year_week", col("close"), col("date"), 
            "max_high", "min_low", "sum_volume"
        )
        
        # Extract weekly open (from the first trading day)
        df_open = df_weekly.filter(col("rn_asc") == 1).select(
            "symbol", "year_week", col("open")
        )

        # Join to get the final OHLCV dataframe
        df_final = df_close.join(df_open, ["symbol", "year_week"], "inner")
        
        df_final = df_final.select(
            "symbol", 
            "date", # Date of the last trading day of the week
            col("open"), 
            col("max_high").alias("high"), 
            col("min_low").alias("low"), 
            col("close"), 
            col("sum_volume").alias("volume")
        )
        
        # 4. Save to Bronze
        logger.info(f"💾 Saving Weekly Commodities data to: {Paths.COMMODITIES_STOCK_PRICES_WEEKLY}")
        
        df_final.write.format("delta") \
            .option("overwriteSchema", "true") \
            .partitionBy("symbol") \
            .mode("overwrite") \
            .save(Paths.COMMODITIES_STOCK_PRICES_WEEKLY)
        
        logger.info("✅ Weekly Commodities prices created successfully!")

    except Exception as e:
        logger.critical(f"❌ Error during weekly resampling: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
