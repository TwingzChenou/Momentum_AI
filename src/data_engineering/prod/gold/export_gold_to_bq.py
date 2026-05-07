import os
import sys
from pyspark.sql import SparkSession
from config.config_spark import Paths, BUCKET_NAME

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# On récupère le chemin de la clé depuis l'environnement ou on utilise le chemin par défaut Airflow
GCP_KEY = os.getenv('GCP_KEY_PATH', '/opt/airflow/config/keys_gcp/finance-ml-project-486410-f5aa9a641051.json')

def create_spark_session():
    """Initialise la session Spark avec les configurations GCS et BigQuery."""
    # Note: Les JARs et packages sont passés via PYSPARK_SUBMIT_ARGS dans la DAG
    return SparkSession.builder \
        .appName("ExportGoldToBigQuery") \
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", GCP_KEY) \
        .getOrCreate()

def export_table(spark, gcs_path, bq_table):
    # On pointe vers le fichier .parquet généré par DuckDB
    parquet_path = f"{gcs_path}.parquet"
    
    print(f"--- Exporting {parquet_path} to BigQuery {bq_table} ---")
    try:
        print(f"Chargement des données Parquet depuis {parquet_path}...")
        df = spark.read.parquet(parquet_path)
    
        print(f"Export vers BigQuery : {bq_table}...")
        df.write \
            .format("bigquery") \
            .option("table", bq_table) \
            .option("temporaryGcsBucket", BUCKET_NAME) \
            .mode("overwrite") \
            .save()
        print(f"Succès : {bq_table} est à jour.")
    except Exception as e:
        print(f"ERREUR lors de l'export de {parquet_path} : {str(e)}")
        raise e

if __name__ == "__main__":
    spark = create_spark_session()
    
    # Configuration des chemins
    tables_to_export = [
        {"gcs": "gs://" + BUCKET_NAME + "/gold/stock_features", "bq": "Dataset_Strategy_Momentum.gold_stock_features"},
        {"gcs": "gs://" + BUCKET_NAME + "/gold/etf_features", "bq": "Dataset_Strategy_Momentum.gold_etf_features"},
        {"gcs": "gs://" + BUCKET_NAME + "/gold/sp500_index_features", "bq": "Dataset_Strategy_Momentum.gold_sp500_index_features"},
    ]
    
    try:
        for table in tables_to_export:
            export_table(spark, table["gcs"], table["bq"])
    finally:
        spark.stop()
