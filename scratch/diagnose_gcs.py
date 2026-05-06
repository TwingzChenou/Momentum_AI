import sys
import os
from pyspark.sql import SparkSession
from loguru import logger

def diagnose_gcs():
    bucket = "finance-data-lake-unique-id"
    paths_to_check = [
        f"gs://{bucket}/silver/data_raw_2b_weekly",
        f"gs://{bucket}/silver/data_raw_etf_weekly",
        f"gs://{bucket}/silver/data_raw_sp500_weekly",
        f"gs://{bucket}/silver/sp500_stock_prices_weekly"
    ]

    # Création session Spark avec accès GCS
    spark = SparkSession.builder \
        .appName("GCS_Diagnostic") \
        .config("spark.jars.ivy", "/tmp/ivy_cache") \
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", "/opt/airflow/config/keys_gcp/finance-ml-project-486410-f5aa9a641051.json") \
        .getOrCreate()

    try:
        for path in paths_to_check:
            logger.info(f"🔍 Vérification du chemin : {path}")
            try:
                # On essaie de lister le contenu via Hadoop FS
                fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
                    spark._jvm.java.net.URI(path), 
                    spark._jsc.hadoopConfiguration()
                )
                
                status = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(path))
                files = [str(s.getPath()) for s in status]
                
                if not files:
                    logger.error(f"❌ LE DOSSIER EST VIDE : {path}")
                else:
                    logger.success(f"✅ {len(files)} fichiers trouvés dans {path}")
                    # On vérifie si c'est une table Delta (présence de _delta_log)
                    is_delta = any("_delta_log" in f for f in files)
                    if is_delta:
                        logger.success(f"   ✨ C'est bien une table Delta valide.")
                    else:
                        logger.warning(f"   ⚠️ Ce n'est PAS une table Delta valide (manque _delta_log).")
            
            except Exception as e:
                logger.error(f"❌ Erreur d'accès au chemin {path} : {e}")

    finally:
        spark.stop()

if __name__ == "__main__":
    diagnose_gcs()
