import os
import sys
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from loguru import logger


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.config_spark import Paths, GCP_KEY_PATH, BQ_TEMP_BUCKET

def create_spark_session(app_name: str = "SparkApp", log_level: str = "ERROR") -> SparkSession:
    # Prevent PYSPARK_PYTHON mismatch errors by forcing workers to match the driver executable
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_KEY_PATH

    # Use the shaded JAR directly to avoid Guava conflicts and Maven coordinate parsing errors
    gcs_jar_url = "https://repo1.maven.org/maven2/com/google/cloud/bigdataoss/gcs-connector/hadoop3-2.2.6/gcs-connector-hadoop3-2.2.6-shaded.jar"
    bq_jar_url = "https://repo1.maven.org/maven2/com/google/cloud/spark/spark-bigquery-with-dependencies_2.12/0.40.0/spark-bigquery-with-dependencies_2.12-0.40.0.jar"
    
    combined_jars = f"{gcs_jar_url},{bq_jar_url}"
    logger.info(f"🛠️ Configurant Spark avec GCS et BigQuery Jars...")

    builder = SparkSession.builder.master("local[*]") \
        .appName(app_name) \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=jdk.management/com.sun.management.internal=ALL-UNNAMED") \
        .config("spark.executor.extraJavaOptions", "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=jdk.management/com.sun.management.internal=ALL-UNNAMED") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", GCP_KEY_PATH) \
        .config("spark.jars", combined_jars)

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