import os
import sys
from pyspark.sql import SparkSession

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

spark = create_spark_session(app_name="Schema_Check")
df = spark.read.format("delta").load(Paths.COMMODITIES_STOCK_PRICES)
print("CURRENT SCHEMA:")
df.printSchema()
spark.stop()
