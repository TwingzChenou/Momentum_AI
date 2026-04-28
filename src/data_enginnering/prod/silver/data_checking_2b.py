import os
import sys
from loguru import logger
from pyspark.sql.functions import col

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths
import pyspark.sql.functions as F

def standardize_columns(df):
    """
    Standardise les colonnes OHLCV : Standardise vers Close 
    pour s'assurer que 'Close' contient toujours le prix ajusté.
    """
    cols = df.columns
    if "AdjClose" in cols and "Close" in cols:
        df = df.drop("Close")
    
    if "AdjClose" in cols:
        df = df.withColumnRenamed("AdjClose", "Close")
        
    for c in df.columns:
        if c.lower() == "adjclose" and "Close" not in df.columns:
            df = df.withColumnRenamed(c, "Close")
            
    return df

def run_data_checks(df, name="Silver Data"):
    """
    Applies Data Quality constraints on OHLCV data.
    Keeps rows where all prices are Null (to maintain Master Index) 
    or where prices respect strict mathematical rules.
    """
    logger.info(f"🔎 Début des vérifications Data Quality pour {name}...")
    
    count_before = df.count()
    
    # 1. Doublons
    df_dedup = df.dropDuplicates(['Ticker', 'Date'])
    count_dedup = df_dedup.count()
    if count_before > count_dedup:
        logger.warning(f"⚠️ {count_before - count_dedup} doublons supprimés sur [Ticker, Date].")

    # 2. Règles Métiers & Nulls (On utilise 'Close' car c'est le standard après standardize_columns)
    all_null_condition = (
        F.col("Open").isNull() & 
        F.col("High").isNull() & 
        F.col("Low").isNull() & 
        F.col("Close").isNull() & 
        F.col("Volume").isNull()
    )
    
    none_null_condition = (
        F.col("Open").isNotNull() & 
        F.col("High").isNotNull() & 
        F.col("Low").isNotNull() & 
        F.col("Close").isNotNull() & 
        F.col("Volume").isNotNull()
    )
    
    positive_prices = (
        (F.col("Open") > 0) & 
        (F.col("High") > 0) & 
        (F.col("Low") > 0) & 
        (F.col("Close") > 0) & 
        (F.col("Volume") >= 0)
    )
    
    logical_prices = (
        (F.col("High") >= F.col("Low")) &
        (F.col("High") >= F.col("Open")) &
        (F.col("High") >= F.col("Close")) &
        (F.col("Low") <= F.col("Open")) &
        (F.col("Low") <= F.col("Close"))
    )
    
    valid_row = all_null_condition | (none_null_condition & positive_prices & logical_prices)
    
    df_filtered = df_dedup.filter(valid_row)
    count_after = df_filtered.count()
    
    diff = count_dedup - count_after
    if diff > 0:
        logger.warning(f"🚨 {diff} lignes supprimées (Erreurs Mathématiques ou Partiellement Nulles).")
    else:
        logger.info("✅ Aucune ligne mathermatiquement invalide !")
        
    return df_filtered

def process_bronze_to_silver(spark, bronze_path, silver_path, name_label):
    logger.info(f"📥 Loading Bronze Data: {bronze_path}")
    try:
        df_bronze = spark.read.format("delta").load(bronze_path)
        if df_bronze.isEmpty(): # PySpark method to check if DF has rows
            logger.warning(f"⚠️ La table Bronze est vide : {bronze_path}")
            return
            
        # --- NOUVEAUTÉ : STANDARDISATION COLONNES ---
        df_standard = standardize_columns(df_bronze)
        
        df_silver = run_data_checks(df_standard, name=name_label)
        
        logger.info(f"💾 Saving to Silver Lake: {silver_path} ({df_silver.count()} rows)")
        
        df_silver.write.format("delta") \
                 .mode("overwrite") \
                 .option("overwriteSchema", "true") \
                 .save(silver_path)
                 
        logger.info(f"✅ Success! Table enregistrée dans {silver_path}.")

    except Exception as e:
        logger.error(f"❌ Erreur sur la pipeline {name_label}: {e}")

def main():
    setup_logging()
    logger.info("🚀 Starting Job: Silver Data Quality (2B Universe)")

    spark = None
    try:
        spark = create_spark_session(app_name="Data_Checking_2B_Silver")
        
        pipelines = [
            (Paths.DATA_RAW_2B, Paths.DATA_RAW_2B_SILVER, "Journalier 2B"),
            (Paths.DATA_RAW_2B_WEEKLY, Paths.DATA_RAW_2B_WEEKLY_SILVER, "Hebdo 2B"),
            (Paths.DATA_RAW_2B_MONTHLY, Paths.DATA_RAW_2B_MONTHLY_SILVER, "Mensuel 2B")
        ]
        
        for bronze, silver, label in pipelines:
            process_bronze_to_silver(spark, bronze, silver, label)

    except Exception as e:
        logger.critical(f"❌ Critical Error in job execution: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
