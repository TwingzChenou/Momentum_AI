import os
import sys
import pandas as pd
from loguru import logger
import time
import pyspark.sql.functions as F

# Configuration
sys.path.append(os.getcwd())
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def migrate_parquet_to_gcs():
    start_time = time.time()
    logger.info("🎬 Démarrage de la migration : Parquet Local -> Delta GCS")
    
    spark = create_spark_session("Parquet_to_GCS_Migration")
    local_path = "gs://finance-data-lake-unique-id/temp/data_SP500.parquet"
    gcs_path = Paths.SP500_STOCK_PRICES
    
    try:
        # 1. Lecture du Parquet depuis GCS
        logger.info(f"📥 Lecture du fichier GCS : {local_path}")
        sdf_raw = spark.read.parquet(local_path)
        
        # Audit rapide avant transformation
        raw_count = sdf_raw.count()
        logger.info(f"📊 Données brutes chargées : {raw_count} lignes")
        
        # 2. Standardisation (via Spark directement pour la performance)
        logger.info("🛠️ Standardisation du schéma...")
        if 'Adj Close' in sdf_raw.columns:
            sdf_standard = sdf_raw.withColumnRenamed('Adj Close', 'AdjClose')
        else:
            sdf_standard = sdf_raw
            
        # Conversion Date si nécessaire
        from pyspark.sql.functions import col, to_date
        sdf_standard = sdf_standard.withColumn("Date", to_date(col("Date")))
        
        # Audit rapide avant envoi
        stats = sdf_standard.select(
            F.min("Date").alias("min_date"),
            F.max("Date").alias("max_date"),
            F.countDistinct("Ticker").alias("tickers_count")
        ).collect()[0]
        
        logger.info(f"📊 Données prêtes : {raw_count} lignes, {stats['tickers_count']} tickers, du {stats['min_date']} au {stats['max_date']}")
        
        # 3. Sauvegarde GCS
        logger.info(f"🚀 Envoi vers GCS : {gcs_path}")
        
        # Sauvegarde en mode Overwrite pour repartir sur une base saine et complète
        sdf_standard.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(gcs_path)
        
        logger.success(f"✅ Migration terminée avec succès en {time.time() - start_time:.2f}s !")
        
    except Exception as e:
        logger.critical(f"❌ Erreur lors de la migration : {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if spark: spark.stop()

if __name__ == "__main__":
    migrate_parquet_to_gcs()
