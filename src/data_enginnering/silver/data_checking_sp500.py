import os
import sys
from loguru import logger
from pyspark.sql.functions import col

# Force Spark match environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def run_data_checks(df, name="Silver Data"):
    """
    Applique les contrôles de qualité OHLCV.
    """
    logger.info(f"🔎 Début des vérifications Data Quality pour {name}...")
    
    count_before = df.count()
    
    # 1. Doublons
    df_dedup = df.dropDuplicates(['Ticker', 'Date'])
    
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
    
    diff = count_before - count_after
    if diff > 0:
        logger.warning(f"🚨 {diff} lignes supprimées (Doublons/Invalides).")
    else:
        logger.info("✅ Données saines.")
        
    return df_filtered

def process_bronze_to_silver(spark, bronze_path, silver_path, label):
    logger.info(f"📥 Loading Bronze: {bronze_path}")
    try:
        df_bronze = spark.read.format("delta").load(bronze_path)
        df_silver = run_data_checks(df_bronze, name=label)
        
        logger.info(f"💾 Saving to Silver: {silver_path}")
        df_silver.write.format("delta").mode("overwrite").save(silver_path)
        logger.success(f"✅ Success {label}.")

    except Exception as e:
        logger.error(f"❌ Erreur sur {label}: {e}")

def main():
    setup_logging()
    spark = create_spark_session(app_name="Data_Checking_SP500_Silver")
    
    try:
        pipelines = [
            (Paths.DATA_RAW_SP500, Paths.DATA_RAW_SP500_SILVER, "SP500 Daily"),
            (Paths.DATA_RAW_SP500_WEEKLY, Paths.DATA_RAW_SP500_WEEKLY_SILVER, "SP500 Hebdo"),
            (Paths.DATA_RAW_SP500_MONTHLY, Paths.DATA_RAW_SP500_MONTHLY_SILVER, "SP500 Mensuel")
        ]
        
        for b, s, l in pipelines:
            process_bronze_to_silver(spark, b, s, l)
            
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
