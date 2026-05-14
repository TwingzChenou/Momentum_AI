import os
import sys
from loguru import logger

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '')))

from src.common.setup_spark import create_spark_session
from config.config_spark import Paths
from src.common.quality_manager import QualityManager

def main():
    spark = create_spark_session(app_name="Data_Audit_Full")
    
    logger.info("🧐 Démarrage de l'Audit Complet du Data Lake...")
    
    # 1. Audit Silver
    targets_silver = [
        (Paths.SP500_STOCK_PRICES_WEEKLY_SILVER, "Stocks Silver"),
        (Paths.DATA_RAW_ETF_WEEKLY_SILVER, "ETFs Silver"),
    ]
    for path, label in targets_silver:
        df = spark.read.format("delta").load(path)
        QualityManager.validate_silver_data(df, label)
        
    # 2. Audit Gold (BigQuery)
    targets_gold = [
        (Paths.BQ_STOCKS_GOLD, "Stocks Gold BQ"),
        (Paths.BQ_ETF_GOLD, "ETFs Gold BQ"),
    ]
    for table, label in targets_gold:
        df = spark.read.format("bigquery").option("table", table).load()
        QualityManager.validate_gold_data(df, label)

    logger.success("✅ Audit terminé. Consultez les logs pour le détail.")

if __name__ == "__main__":
    main()
