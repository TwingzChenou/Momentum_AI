import os
import sys
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def main():
    spark = create_spark_session("Audit_AdjClose")
    
    try:
        # Load Silver Daily
        path = "gs://finance-data-lake-unique-id/silver/sp500_stock_prices.parquet"
        df = spark.read.parquet(path)
        
        import pyspark.sql.functions as F
        
        print("\n=== ANALYSE DES COLONNES ADJCLOSE ===")
        stats = df.select(
            F.count("*").alias("total"),
            F.count("AdjClose").alias("count_adj"),
            F.min("Date").alias("min_date"),
            F.min(F.when(F.col("AdjClose").isNotNull(), F.col("Date"))).alias("min_date_adj")
        ).collect()[0]
        
        print(f"Total lignes       : {stats['total']}")
        print(f"Lignes avec AdjClose: {stats['count_adj']}")
        print(f"Date début globale : {stats['min_date']}")
        print(f"Date début AdjClose: {stats['min_date_adj']}")
        
        if stats['count_adj'] < stats['total']:
            print("\n⚠️ ATTENTION : Des lignes n'ont pas de prix ajusté (AdjClose).")
            print("Cela explique pourquoi le rééchantillonnage supprime l'historique ancien !")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
