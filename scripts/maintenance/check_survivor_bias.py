
import sys
import os
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths
from pyspark.sql import functions as F

def check_survivor_bias():
    spark = create_spark_session()
    print("\n🔍 --- ANALYSE DÉTAILLÉE DU BIAIS DU SURVIVANT ---")
    
    try:
        df = spark.read.format("delta").load(Paths.SP500_CONSOLIDATED_HISTORY)
        
        n_tickers = df.select("Ticker").distinct().count()
        n_exits = df.filter(F.col("Date_end").isNotNull()).count()
        
        print(f"Total Tickers enregistrés : {n_tickers}")
        print(f"Nombre de sorties enregistrées : {n_exits}")
        
        if n_tickers <= 505:
            print("\n⚠️ ALERTE : BIAIS DU SURVIVANT CONFIRMÉ.")
            print("Explication : Votre base ne contient que les membres ACTUELS du S&P 500.")
            print("Pour un backtest sans biais sur 50 ans, vous devriez avoir environ 1500 à 2000 tickers différents.")
        else:
            print("\n✅ VOTRE BASE EST SAINE.")
            print(f"Vous avez {n_tickers} tickers, ce qui prouve que vous conservez l'historique des membres sortis.")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'accès aux données : {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    check_survivor_bias()
