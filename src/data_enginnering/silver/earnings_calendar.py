import os
import sys
from loguru import logger
from pyspark.sql.functions import col, to_date, abs as F_abs, when

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths


def read_earnings_calendar(spark):
    """Reads the Earnings Calendar history from Delta Lake without triggering actions."""
    logger.info(f"Reading Earnings Calendar from {Paths.SP500_EARNINGS_SURPRISE}...")
    return spark.read.format("delta").load(Paths.SP500_EARNINGS_SURPRISE)


def main():
    setup_logging()
    logger.info("🚀 Starting Spark session for Earnings Surprise...")
    spark = create_spark_session()

    try:
        df_bronze = read_earnings_calendar(spark)
        
        # --- THE CLEANSING LOGIC (SILVER LAYER) ---
        logger.info("🧹 Cleaning data and calculating surprise metrics...")
        
        tickers_to_exclude = ['EF', 'JBL', 'HP', 'TMUS', 'FMCC', 'FNMA', 'CTX', 'AET', 'MXIM', 'PARA']
        
        df_silver = df_bronze \
            .withColumn("date", to_date(col("date"))) \
            .dropDuplicates(["symbol", "date"]) \
            .dropna(subset=["date", "symbol"]) \
            .filter(~col("symbol").isin(tickers_to_exclude)) \
            .withColumn(
                "suprise_eps",
                when(F_abs(col("epsEstimated")) > 0, 
                     (col("epsActual") - col("epsEstimated")) / F_abs(col("epsEstimated")))
                .otherwise(None)
            ) \
            .withColumn(
                "suprise_revenue",
                when(F_abs(col("revenueEstimated")) > 0, 
                     (col("revenueActual") - col("revenueEstimated")) / F_abs(col("revenueEstimated")))
                .otherwise(None)
            )

        # ------------------------------------------

        logger.info(f"💾 Saving cleaned data to Silver: {Paths.SP500_EARNINGS_SURPRISE_SILVER}")
    
        # 🛠️ CORRECTION : Suppression de repartition() et partitionBy()
        # Un ou deux gros fichiers Delta seront infiniment plus rapides à lire pour tes backtests
        df_silver.write.format("delta") \
            .option("overwriteSchema", "true") \
            .mode("overwrite") \
            .save(Paths.SP500_EARNINGS_SURPRISE_SILVER)
        
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
