import os
import sys
from pyspark.sql import SparkSession

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

spark = create_spark_session(app_name="Cleanup_Commodities")
df = spark.read.format("delta").load(Paths.COMMODITIES_STOCK_PRICES)
cols_to_keep = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
df_clean = df.select(*cols_to_keep)
df_clean.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(Paths.COMMODITIES_STOCK_PRICES)
print("CLEANUP DONE")
spark.stop()
