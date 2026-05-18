import os
import sys
from loguru import logger
from pyspark.sql import functions as F

# Configuration du chemin projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.common.setup_spark import create_spark_session
from config.config_spark import Paths, BQ_TEMP_BUCKET
from src.common.config_utils import get_champion_config
from src.common.indicators_spark import calculate_technical_indicators

def main():
    spark = create_spark_session(app_name="Great_Gold_Restoration")
    config = get_champion_config()
    
    spark.conf.set("spark.sql.parquet.writeLegacyFormat", "true")
    
    # 1. RÉPARATION DES ACTIONS (S&P 500 + UNIVERSE 2B)
    logger.info("🛠️ Restauration des ACTIONS (S&P 500 + 2B)...")
    df_sp500 = spark.read.format("delta").load(Paths.SP500_STOCK_PRICES_WEEKLY_SILVER)
    df_2b = spark.read.format("delta").load(Paths.DATA_RAW_2B_WEEKLY_SILVER)
    
    # Union des deux univers
    df_stocks = df_sp500.unionByName(df_2b, allowMissingColumns=True).dropDuplicates(['Ticker', 'Date'])
    df_stocks = df_stocks.withColumn("Date", F.col("Date").cast("date"))
    
    # Calcul des indicateurs
    df_stocks = calculate_technical_indicators(
        df_stocks, 
        sma_fast_p=config.get('stock_sma_fast', 13),
        sma_slow_p=config.get('stock_sma_slow', 50),
        adx_p=9,  # Force le nouveau ADX 9
        atr_p=4   # Force le nouveau ATR 4
    )
    
    # Add the Eligibility flag based on:
    # Trend: Close > SMA_slow and SMA_fast > SMA_slow
    # Force : ADX > Seuil (20 par défaut)
    # Risque : ATR_pct < Seuil (20% par défaut)
    df_stocks = df_stocks.withColumn("Eligible", 
        (df_stocks.Close > df_stocks.SMA_slow) & 
        (df_stocks.SMA_fast > df_stocks.SMA_slow) & 
        (df_stocks.ADX > config.get('stock_adx_threshold', 20.0)) & 
        (df_stocks.ATR_pct < config.get('stock_atr_threshold', 20.0))
    )
    
    # Écriture dans BigQuery
    logger.info(f"📤 Injection de {df_stocks.count()} lignes d'Actions dans BigQuery...")
    df_stocks.write.format("bigquery") \
        .option("table", Paths.BQ_STOCKS_GOLD) \
        .option("temporaryGcsBucket", BQ_TEMP_BUCKET) \
        .mode("overwrite") \
        .save()

    # 2. RÉPARATION DES ETFS
    logger.info("🛠️ Restauration des ETFS...")
    df_etf = spark.read.format("delta").load(Paths.DATA_RAW_ETF_WEEKLY_SILVER)
    df_etf = df_etf.withColumn("Date", F.col("Date").cast("date"))
    df_etf = calculate_technical_indicators(
        df_etf, 
        sma_fast_p=config.get('etf_sma_fast', 13),
        sma_slow_p=config.get('etf_sma_slow', 50),
        adx_p=9,
        atr_p=4
    )
    
    logger.info(f"📤 Injection de {df_etf.count()} lignes d'ETFs dans BigQuery...")
    df_etf.write.format("bigquery") \
        .option("table", Paths.BQ_ETF_GOLD) \
        .option("temporaryGcsBucket", BQ_TEMP_BUCKET) \
        .mode("overwrite") \
        .save()

    # 3. RÉPARATION DE L'INDICE S&P 500
    logger.info("🛠️ Restauration de l'INDICE S&P 500 (^GSPC)...")
    df_idx = spark.read.format("delta").load(Paths.DATA_RAW_SP500_WEEKLY_SILVER)
    df_idx = df_idx.withColumn("Date", F.col("Date").cast("date"))
    df_idx = calculate_technical_indicators(
        df_idx, 
        sma_fast_p=config.get('sp500_sma_fast', 7),
        sma_slow_p=config.get('sp500_sma_slow', 30),
        adx_p=9,
        atr_p=4
    )
    
    logger.info(f"📤 Injection de {df_idx.count()} lignes d'Indice dans BigQuery...")
    df_idx.write.format("bigquery") \
        .option("table", Paths.BQ_SP500_GOLD) \
        .option("temporaryGcsBucket", BQ_TEMP_BUCKET) \
        .mode("overwrite") \
        .save()

    logger.success("✨ LA GRANDE RESTAURATION EST TERMINÉE !")

if __name__ == "__main__":
    main()
