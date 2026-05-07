import os
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# Configuration pour accéder à GCS
sys.path.append(os.getcwd())
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def audit_data():
    spark = create_spark_session("Data_Audit")
    
    paths_to_check = [
        ("BRONZE Stocks", Paths.SP500_STOCK_PRICES),
        ("SILVER Daily Stocks", Paths.SP500_STOCK_PRICES_SILVER + ".parquet"),
        ("SILVER Weekly Stocks", Paths.SP500_STOCK_PRICES_WEEKLY_SILVER),
        ("GOLD Features", Paths.STOCK_FEATURES_GOLD + ".parquet")
    ]
    
    print("\n" + "="*80)
    print(f"{'Table':<25} | {'Rows':<12} | {'Min Date':<12} | {'Max Date':<12}")
    print("-"*80)
    
    for name, path in paths_to_check:
        try:
            if "parquet" in path:
                df = spark.read.parquet(path)
            else:
                df = spark.read.format("delta").load(path)
            
            stats = df.select(
                F.count("*").alias("count"),
                F.min("Date").alias("min_date"),
                F.max("Date").alias("max_date")
            ).collect()[0]
            
            print(f"{name:<25} | {stats['count']:<12} | {str(stats['min_date']):<12} | {str(stats['max_date']):<12}")
        except Exception as e:
            print(f"{name:<25} | {'ERROR':<12} | {'N/A':<12} | {str(e)[:30]}")
            
    print("="*80 + "\n")
    spark.stop()

if __name__ == "__main__":
    audit_data()
