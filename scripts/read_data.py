import os
import sys
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt

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
    logger.info(f"Reading S&P 500 history from {Paths.DATA_RAW_SP500_WEEKLY_SILVER}...")
    df = spark.read.format("delta").load(Paths.DATA_RAW_SP500_WEEKLY_SILVER)
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
    df.createOrReplaceTempView("sp500_history")
    df = df.toPandas()

    logger.info("Showing S&P 500 history...")
    pd.set_option('display.max_columns', None)

    print(df.head(10))
    print(df.describe())
    print(df.sort_values(by="Date", ascending=False).head(5))
    print(df.sort_values(by="Date", ascending=True).head(5))
    #print(df['symbol'].unique())

    logger.info("Stopping Spark session...")
    spark.stop()

if __name__ == "__main__":
    main()