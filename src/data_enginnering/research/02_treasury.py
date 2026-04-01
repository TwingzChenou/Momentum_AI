import os
import sys
from loguru import logger
from pyspark.sql.functions import col

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
    logger.info("🚀 Starting Job: Calculate Gold Treasury Risk-Free Rates")

    spark = None

    try:
        spark = create_spark_session(app_name="Gold_Treasury_Rates")

        logger.info(f"📡 Reading Bronze Treasury data from {Paths.TREASURY_BOND_BRONZE}...")
        df_bronze = spark.read.format("delta").load(Paths.TREASURY_BOND_BRONZE)
        logger.info(f"✅ Loaded {df_bronze.count()} records.")

        logger.info("📐 Calculating daily, weekly, and monthly risk-free rates from `month1`...")
        
        # We need the `date` and the `month1` yield
        df_gold = df_bronze.select(
            col("date"),
            col("month3")
        ).dropna(subset=["month3"])

        # Convert the annualized percentage (e.g., 3.72) to a decimal rate (0.0372)
        df_gold = df_gold.withColumn("annualized_rate", col("month3") / 100.0)

        # Calculate the periodic risk-free rates using simple division as requested
        df_gold = df_gold \
            .withColumn("daily_risk_free_rate", col("annualized_rate") / 252.0) \
            .withColumn("weekly_risk_free_rate", col("annualized_rate") / 52.0) \
            .withColumn("monthly_risk_free_rate", col("annualized_rate") / 12.0) \
            .drop("annualized_rate", "month3")

        logger.info(f"💾 Saving complete Gold Treasury data to {Paths.TREASURY_BOND_GOLD}...")
        
        # Save to Gold layer
        df_gold.coalesce(1).write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(Paths.TREASURY_BOND_GOLD)
        
        logger.success("✅ Gold layer for Treasury Risk-Free Rates created successfully!")

    except Exception as e:
        logger.critical(f"❌ Error during Bronze to Gold transition for Treasury data: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
