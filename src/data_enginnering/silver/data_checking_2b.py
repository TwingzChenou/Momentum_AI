import os
import sys
from loguru import logger
from pyspark.sql.functions import col

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

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

    # 2. Règles Métiers & Nulls
    all_null_condition = (
        col("Open").isNull() & 
        col("High").isNull() & 
        col("Low").isNull() & 
        col("Close").isNull() & 
        col("AdjClose").isNull() & 
        col("Volume").isNull()
    )
    
    none_null_condition = (
        col("Open").isNotNull() & 
        col("High").isNotNull() & 
        col("Low").isNotNull() & 
        col("Close").isNotNull() & 
        col("AdjClose").isNotNull() & 
        col("Volume").isNotNull()
    )
    
    positive_prices = (
        (col("Open") > 0) & 
        (col("High") > 0) & 
        (col("Low") > 0) & 
        (col("Close") > 0) & 
        (col("AdjClose") > 0) & 
        (col("Volume") >= 0)
    )
    
    logical_prices = (
        (col("High") >= col("Low")) &
        (col("High") >= col("Open")) &
        (col("High") >= col("Close")) &
        (col("Low") <= col("Open")) &
        (col("Low") <= col("Close"))
    )
    
    valid_row = all_null_condition | (none_null_condition & positive_prices & logical_prices)
    
    df_filtered = df_dedup.filter(valid_row)
    count_after = df_filtered.count()
    
    diff = count_dedup - count_after
    if diff > 0:
        logger.warning(f"🚨 {diff} lignes supprimées (Erreurs Mathématiques ou Partiellement Nulles).")
    else:
        logger.info("✅ Aucune ligne mathématiquement invalide !")
        
    return df_filtered

def process_bronze_to_silver(spark, bronze_path, silver_path, name_label):
    logger.info(f"📥 Loading Bronze Data: {bronze_path}")
    try:
        df_bronze = spark.read.format("delta").load(bronze_path)
        if df_bronze.isEmpty(): # PySpark method to check if DF has rows
            logger.warning(f"⚠️ La table Bronze est vide : {bronze_path}")
            return
            
        df_silver = run_data_checks(df_bronze, name=name_label)
        
        logger.info(f"💾 Saving to Silver Lake: {silver_path} ({df_silver.count()} rows)")
        
        df_silver.write.format("delta") \
                 .mode("overwrite") \
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
