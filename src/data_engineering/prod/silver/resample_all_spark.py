import os
import sys
from loguru import logger
import great_expectations as gx

# Setup path to access project modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '')))

from src.common.setup_spark import create_spark_session
from config.config_spark import Paths
from src.common.sql_queries import QUERY_RESAMPLE_WEEKLY
from src.common.quality_manager import QualityManager

def process_layer(spark, bronze_path, silver_path, label):
    logger.info(f"🔄 Traitement de la couche Silver pour : {label}")
    
    # 1. Lecture Bronze
    df_bronze = spark.read.format("delta").load(bronze_path)
    df_bronze.createOrReplaceTempView("bronze_data")
    
    # 2. Transformation SQL
    df_silver = spark.sql(QUERY_RESAMPLE_WEEKLY)
    
    # 3. Validation
    QualityManager.validate_silver_data(df_silver, label)
    
    # 4. Écriture Silver
    logger.info(f"📤 Sauvegarde de {label} dans {silver_path}...")
    df_silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path)

def main():
    spark = create_spark_session(app_name="Silver_Layer_Unified")
    
    targets = [
        (Paths.SP500_STOCK_PRICES, Paths.SP500_STOCK_PRICES_WEEKLY_SILVER, "Stocks S&P 500"),
        (Paths.DATA_RAW_ETF, Paths.DATA_RAW_ETF_WEEKLY_SILVER, "ETFs"),
        (Paths.DATA_RAW_2B, Paths.DATA_RAW_2B_WEEKLY_SILVER, "Universe 2B"),
        (Paths.DATA_RAW_SP500, Paths.DATA_RAW_SP500_WEEKLY_SILVER, "Index ^GSPC")
    ]
    
    for bronze, silver, label in targets:
        try:
            process_layer(spark, bronze, silver, label)
        except Exception as e:
            logger.error(f"❌ Erreur sur {label}: {e}")

    logger.success("✨ Couche SILVER unifiée et validée !")

if __name__ == "__main__":
    main()
