import os
import sys
from loguru import logger

# Configuration Spark
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def verify():
    spark = create_spark_session(app_name="Verify_Lake")
    try:
        logger.info(f"📊 Analyse de la table : {Paths.DATA_RAW_2B}")
        df = spark.read.format("delta").load(Paths.DATA_RAW_2B)
        
        stats = df.selectExpr(
            "min(Date) as min_date", 
            "max(Date) as max_date", 
            "count(*) as total_rows",
            "count(distinct Ticker) as total_tickers"
        ).collect()[0]
        
        logger.success("✅ Vérification terminée !")
        print(f"\n--- Statistiques Bronze (Daily) ---")
        print(f"📅 Date début    : {stats['min_date']}")
        print(f"📅 Date fin      : {stats['max_date']}")
        print(f"📈 Nb Tickers    : {stats['total_tickers']}")
        print(f"🔢 Total lignes  : {stats['total_rows']:,}")
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    verify()
