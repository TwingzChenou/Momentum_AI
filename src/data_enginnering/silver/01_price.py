import os
import sys
from loguru import logger
from pyspark.sql.functions import col, to_date

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths


def read_sp500_price(spark):
    """Reads the S&P 500 history from Delta Lake."""
    logger.info(f"Reading S&P 500 history from {Paths.SP500_STOCK_PRICES}...")
    df = spark.read.format("delta").load(Paths.SP500_STOCK_PRICES)
    logger.info(f"Found {df.count()} records in S&P 500 history.")
    logger.info(f"Columns: {df.columns}")
    logger.info(f"Schema: {df.schema}")
    return df


def main():
    setup_logging()
    logger.info("Starting Spark session...")
    spark = create_spark_session()

    try:
        logger.info(f"📥 Reading raw data from Bronze: {Paths.SP500_STOCK_PRICES}")
        df_bronze = read_sp500_price(spark)
        
        # --- THE CLEANSING LOGIC (SILVER LAYER) ---
        logger.info("🧹 Cleaning data...")
    
        df_silver = df_bronze \
        .withColumn("date", to_date(col("date"))) \
        .dropDuplicates(["symbol", "date"]) \
        .filter(col("volume") > 0) \
        .filter(col("adjClose") > 0) \
        .dropna(subset=["adjClose", "date", "symbol"])

        tickers_to_exclude = ['EF', 'JBL', 'HP', 'TMUS', 'FMCC', 'FNMA', 'CTX', 'AET', 'MXIM', 'PARA']
        df_silver = df_silver.filter(~col("symbol").isin(tickers_to_exclude))
    
        # ------------------------------------------

        logger.info(f"💾 Saving cleaned data to Silver: {Paths.SP500_STOCK_PRICES_SILVER}")
    
        # We use "overwrite" here because if we change our cleaning rules later,
        # we want to completely replace the Silver table with the newly cleaned data.
        # Repartitioning by symbol before writing prevents the "small files problem" 
        # and significantly speeds up the write process to GCS.
        df_silver.repartition("symbol").write.format("delta") \
            .option("overwriteSchema", "true") \
            .partitionBy("symbol") \
            .mode("overwrite") \
            .save(Paths.SP500_STOCK_PRICES_SILVER)
        
        logger.info("✅ Silver layer created successfully!")

    except Exception as e:
        logger.error(f"❌ Error during Bronze to Silver transition: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")



if __name__ == "__main__":
    main()