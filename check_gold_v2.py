import os
import sys
sys.path.append(os.path.abspath('.'))
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

spark = create_spark_session(app_name="Check_Gold_V2")
for p in [Paths.DATA_RAW_2B_WEEKLY_GOLD, Paths.DATA_RAW_ETF_WEEKLY_GOLD, Paths.DATA_RAW_SP500_WEEKLY_SILVER]:
    try:
        df = spark.read.format("delta").load(p)
        print(f"Table {p}: {df.count()} rows")
    except Exception as e:
        print(f"Error loading {p}: {e}")
spark.stop()
