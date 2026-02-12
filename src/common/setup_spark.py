import os
import sys
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from loguru import logger


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.config_spark import Paths, GCP_KEY_PATH

def create_spark_session(app_name: str = "SparkApp", log_level: str = "ERROR") -> SparkSession:

    # Use the shaded JAR directly to avoid Guava conflicts and Maven coordinate parsing errors
    gcs_jar_url = "https://repo1.maven.org/maven2/com/google/cloud/bigdataoss/gcs-connector/hadoop3-2.2.6/gcs-connector-hadoop3-2.2.6-shaded.jar"
    
    logger.info(f"🛠️ Configurant Spark avec le connecteur GCS : {gcs_jar_url}")

    builder = SparkSession.builder.master("local[*]") \
        .appName(app_name) \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", GCP_KEY_PATH) \
        .config("spark.jars", gcs_jar_url)

    # CRUCIAL : On passe le package GCS à la configuration Delta
    spark = configure_spark_with_delta_pip(
        builder
    ).getOrCreate()

    logger.success(f"✅ Spark Session '{app_name}' créée avec succès ! (Version: {spark.version})")

    spark.sparkContext.setLogLevel(log_level)
    return spark
    
def main():
    create_spark_session()  

if __name__ == "__main__":
    main()