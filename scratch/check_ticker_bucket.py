import os
import sys
import pandas as pd
from loguru import logger

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def check_ticker_in_list(ticker: str, bucket_list: list) -> bool:
    """
    Vérifie si un ticker est présent dans une liste (insensible à la casse).
    """
    ticker_upper = ticker.strip().upper()
    bucket_upper = [t.strip().upper() for t in bucket_list]
    return ticker_upper in bucket_upper

def get_tickers_from_silver(spark, path: str) -> list:
    """
    Récupère la liste unique des tickers depuis une table Delta Silver.
    """
    try:
        df = spark.read.format("delta").load(path)
        tickers = df.select("Ticker").distinct().toPandas()["Ticker"].tolist()
        return tickers
    except Exception as e:
        logger.error(f"Erreur lors du chargement des tickers depuis {path}: {e}")
        return []

def main():
    # --- CONFIGURATION ---
    TARGET_TICKER = "AXTI"
    PATH_TO_CHECK = Paths.LIST_TICKER_2B
    
    logger.info(f"🔍 Recherche de '{TARGET_TICKER}' dans le bucket '{PATH_TO_CHECK}'...")
    
    # Initialisation Spark avec fix pour Mac
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    spark = create_spark_session("Check_INTC_2B")
    
    try:
        # Chargement du bucket
        logger.info(f"📥 Chargement des données depuis {PATH_TO_CHECK}...")
        df = spark.read.format("delta").load(PATH_TO_CHECK)
        
        # Extraction des tickers uniques (la colonne s'appelle 'symbol' dans ce bucket)
        tickers = df.select("symbol").distinct().toPandas()["symbol"].tolist()
        
        # Vérification
        if check_ticker_in_list(TARGET_TICKER, tickers):
            logger.success(f"✅ '{TARGET_TICKER}' EST PRÉSENT dans le bucket list_ticker_2b.")
        else:
            logger.warning(f"❌ '{TARGET_TICKER}' N'EST PAS PRÉSENT dans le bucket list_ticker_2b.")
            
        logger.info(f"📊 Nombre total de tickers dans ce bucket : {len(tickers)}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la vérification : {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
