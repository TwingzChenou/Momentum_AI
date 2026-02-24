import os
import sys
from loguru import logger

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths


def read_sp500_history(spark):
    """Reads the S&P 500 history from Delta Lake."""
    logger.info(f"Reading S&P 500 history from {Paths.MACRO_PRICES_SILVER}...")
    df = spark.read.format("delta").load(Paths.MACRO_PRICES_SILVER)
    logger.info(f"Found {df.count()} records in S&P 500 history.")
    logger.info(f"Columns: {df.columns}")
    logger.info(f"Schema: {df.schema}")
    return df

def main():
    setup_logging()
    logger.info("Starting Spark session...")
    spark = create_spark_session()

    logger.info("Reading S&P 500 history...")
    df = read_sp500_history(spark)

    logger.info("Showing S&P 500 history...")
    df.orderBy("date", ascending=False).show()
    df.orderBy("date", ascending=True).show()
    logger.info("Stopping Spark session...")
    spark.stop()

if __name__ == "__main__":
    main()