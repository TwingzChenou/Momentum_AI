import os
import sys
from loguru import logger
from pyspark.sql.functions import col

# Force Spark match environment
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
    # Si on a les deux, on dégage le Close brut pour garder l'ajusté
    if "AdjClose" in cols and "Close" in cols:
        df = df.drop("Close")
    
    if "AdjClose" in cols:
        df = df.withColumnRenamed("AdjClose", "Close")
        
    # Normalisation case-insensitive si besoin (optionnel mais recommandé)
    for c in df.columns:
        if c.lower() == "adjclose" and "Close" not in df.columns:
            df = df.withColumnRenamed(c, "Close")
            
    return df

def run_data_checks(df, name="Silver Data"):
    """
    Applique les contrôles de qualité OHLCV.
    """
    logger.info(f"🔎 Début des vérifications Data Quality pour {name}...")
    
    count_before = df.count()
    
    # 1. Doublons
    df_dedup = df.dropDuplicates(['Ticker', 'Date'])
    
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
        
        # --- NOUVEAUTÉ : STANDARDISATION COLONNES ---
        df_standard = standardize_columns(df_bronze)
        
        df_silver = run_data_checks(df_standard, name=label)
        
        logger.info(f"💾 Saving to Silver: {silver_path}")
        df_silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path)
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
