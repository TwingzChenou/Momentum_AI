import os
import sys
from loguru import logger
import great_expectations as gx

# Setup path to access project modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '')))

from pyspark.sql import functions as F
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths
from src.common.sql_queries import QUERY_RESAMPLE_WEEKLY
from src.common.quality_manager import QualityManager

def process_layer(spark, bronze_path, silver_path, history_path, label):
    logger.info(f"🔄 Traitement de la couche Silver pour : {label}")
    
    # 1. Lecture Bronze
    df_bronze = spark.read.format("delta").load(bronze_path)
    
    # 2. Filtrage par Historique (Anti-Biais de Survie)
    if history_path:
        logger.info(f"🛡️ Application du filtre historique (Anti-Biais) depuis {history_path}")
        df_history = spark.read.format("delta").load(history_path)
        
        # Jointure et filtrage temporel
        # Note: on utilise Date_end IS NULL pour les actions encore présentes
        df_bronze = df_bronze.join(df_history, on="Ticker", how="inner") \
            .filter((F.col("Date") >= F.col("Date_start")) & 
                    ((F.col("Date") <= F.col("Date_end")) | F.col("Date_end").isNull()))
        
        logger.info(f"✅ Filtrage terminé. Lignes restantes après join : {df_bronze.count()}")

    df_bronze.createOrReplaceTempView("bronze_data")
    
    # 3. Transformation SQL (Rééchantillonnage Hebdomadaire)
    df_silver = spark.sql(QUERY_RESAMPLE_WEEKLY)
    
    # 4. Validation GX
    QualityManager.validate_silver_data(df_silver, label)
    
    # 5. Écriture Silver
    logger.info(f"📤 Sauvegarde de {label} dans {silver_path}...")
    df_silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path)

def main():
    spark = create_spark_session(app_name="Silver_Layer_Unified")
    
    # Cibles : (Source_Bronze, Destination_Silver, Historique_Optionnel, Label)
    targets = [
        (Paths.SP500_STOCK_PRICES, Paths.SP500_STOCK_PRICES_WEEKLY_SILVER, Paths.SP500_CONSOLIDATED_HISTORY, "Stocks S&P 500"),
        (Paths.DATA_RAW_2B, Paths.DATA_RAW_2B_WEEKLY_SILVER, Paths.TICKERS_2B_CONSOLIDATED_HISTORY, "Universe 2B"),
        (Paths.DATA_RAW_ETF, Paths.DATA_RAW_ETF_WEEKLY_SILVER, None, "ETFs"),
        (Paths.DATA_RAW_SP500, Paths.DATA_RAW_SP500_WEEKLY_SILVER, None, "Index ^GSPC")
    ]
    
    for bronze, silver, history, label in targets:
        try:
            process_layer(spark, bronze, silver, history, label)
        except Exception as e:
            logger.error(f"❌ Erreur sur {label}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.success("✨ Couche SILVER unifiée, historisée et validée !")

    logger.success("✨ Couche SILVER unifiée et validée !")

if __name__ == "__main__":
    main()
