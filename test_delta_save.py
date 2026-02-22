import os
import sys
import pandas as pd
from loguru import logger
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType
from pyspark.sql.functions import col, to_date

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def main():
    spark = create_spark_session(app_name="Test_Spark_NPE")
    cache_file = 'local_prices_cache.parquet'
    
    print("Reading parquet cache...")
    pandas_df = pd.read_parquet(cache_file)
    print(f"Loaded {len(pandas_df)} rows. Memory info: ")
    print(pandas_df.info())
    
    print("Dropping NaNs and casting...")
    pandas_df = pandas_df.dropna(subset=['symbol', 'date'])
    pandas_df['symbol'] = pandas_df['symbol'].astype(str)
    pandas_df['date'] = pandas_df['date'].astype(str)
    
    pandas_df['adjOpen'] = pd.to_numeric(pandas_df['adjOpen'], errors='coerce').fillna(0.0)
    pandas_df['adjHigh'] = pd.to_numeric(pandas_df['adjHigh'], errors='coerce').fillna(0.0)
    pandas_df['adjLow'] = pd.to_numeric(pandas_df['adjLow'], errors='coerce').fillna(0.0)
    pandas_df['adjClose'] = pd.to_numeric(pandas_df['adjClose'], errors='coerce').fillna(0.0)
    pandas_df['volume'] = pd.to_numeric(pandas_df['volume'], errors='coerce').fillna(0).astype('int64')
    
    # Try Explicit schema directly instead of inferring
    print("Creating explicit schema...")
    schema = StructType([
        StructField("symbol", StringType(), True),
        StructField("date", StringType(), True),
        StructField("adjOpen", DoubleType(), True),
        StructField("adjHigh", DoubleType(), True),
        StructField("adjLow", DoubleType(), True),
        StructField("adjClose", DoubleType(), True),
        StructField("volume", LongType(), True)
    ])
    
    print("Creating Spark DataFrame...")
    try:
        # Pass schema directly to avoid Arrow inference issues
        sdf = spark.createDataFrame(pandas_df, schema=schema)
        print("Spark DF Schema:")
        sdf.printSchema()
        
        print("Casting types...")
        sdf = sdf.withColumn("date", to_date(col("date")))
        
        print("Showing 5 rows to trigger action...")
        sdf.show(5)
        
        print("Writing to Delta...")
        sdf.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(Paths.SP500_STOCK_PRICES)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
