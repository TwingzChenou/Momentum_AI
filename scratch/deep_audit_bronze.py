import os
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# Configuration
sys.path.append(os.getcwd())
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def deep_audit():
    spark = create_spark_session("Deep_Audit_Bronze")
    path = Paths.SP500_STOCK_PRICES
    
    print(f"\n🔍 Analyse de la table : {path}")
    
    try:
        df = spark.read.format("delta").load(path)
        
        # Stats globales
        stats = df.select(
            F.count("*").alias("total_rows"),
            F.countDistinct("Ticker").alias("distinct_tickers"),
            F.min("Date").alias("min_date"),
            F.max("Date").alias("max_date")
        ).collect()[0]
        
        print("-" * 50)
        print(f"📈 Nombre total de lignes : {stats['total_rows']}")
        print(f"ticker Nombre de tickers distincts : {stats['distinct_tickers']}")
        print(f"📅 Plage de dates : du {stats['min_date']} au {stats['max_date']}")
        print("-" * 50)
        
        # Top 10 Tickers par nombre de points
        print("\n🔝 Top 10 Tickers avec le plus de données :")
        df.groupBy("Ticker") \
          .agg(F.count("*").alias("points"), F.min("Date").alias("start"), F.max("Date").alias("end")) \
          .orderBy(F.desc("points")) \
          .show(10)
          
    except Exception as e:
        print(f"❌ Erreur lors de la lecture de la table : {e}")
        
    spark.stop()

if __name__ == "__main__":
    deep_audit()
