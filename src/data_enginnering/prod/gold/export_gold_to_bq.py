import os
import sys
from pyspark.sql import SparkSession
from config.config_spark import Paths, BUCKET_NAME

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

def create_spark_session():
    """Initialise la session Spark avec les configurations GCS et BigQuery."""
    return SparkSession.builder \
        .appName("ExportGoldToBigQuery") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

def export_table(spark, gcs_path, bq_table):
    print(f"--- Exporting {gcs_path} to BigQuery {bq_table} ---")
    try:
        # Debug: check if directory exists via spark's hadoop config
        sc = spark.sparkContext
        Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
        FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
        conf = sc._jsc.hadoopConfiguration()
        fs = FileSystem.get(Path(gcs_path).toUri(), conf)
        
        if not fs.exists(Path(gcs_path)):
            print(f"WARNING: Path {gcs_path} DOES NOT EXIST according to Hadoop FileSystem.")
            # List parent directory to see what's there
            parent = Path(gcs_path).getParent()
            print(f"Listing parent directory: {parent}")
            if fs.exists(parent):
                statuses = fs.listStatus(parent)
                for s in statuses:
                    print(f"  - {s.getPath()}")
            else:
                print(f"Parent directory {parent} also does not exist.")
        
        print(f"Chargement des données depuis {gcs_path}...")
        df = spark.read.format("delta").load(gcs_path)
    
        print(f"Export vers BigQuery : {bq_table}...")
        df.write \
            .format("bigquery") \
            .option("table", bq_table) \
            .option("temporaryGcsBucket", BUCKET_NAME) \
            .mode("overwrite") \
            .save()
        print(f"Succès : {bq_table} est à jour.")
    except Exception as e:
        print(f"ERREUR lors de l'export de {gcs_path} : {str(e)}")
        raise e

if __name__ == "__main__":
    spark = create_spark_session()
    
    # Configuration des chemins synchronisée avec config_spark.py
    tables_to_export = [
        {"gcs": Paths.STOCK_FEATURES_GOLD, "bq": Paths.BQ_STOCKS_GOLD},
        {"gcs": Paths.ETF_FEATURES_GOLD, "bq": Paths.BQ_ETF_GOLD},
        {"gcs": Paths.INDEX_FEATURES_GOLD, "bq": Paths.BQ_SP500_GOLD},
    ]
    
    try:
        for table in tables_to_export:
            export_table(spark, table["gcs"], table["bq"])
    finally:
        spark.stop()
