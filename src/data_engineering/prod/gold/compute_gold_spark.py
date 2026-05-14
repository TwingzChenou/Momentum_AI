import os
import sys
from loguru import logger
import great_expectations as gx

# Setup path to access project modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '')))

from src.common.setup_spark import create_spark_session
from config.config_spark import Paths, BQ_TEMP_BUCKET
from src.common.sql_queries import QUERY_COMPUTE_GOLD_INDICATORS
from src.common.config_utils import get_champion_config
from src.common.quality_manager import QualityManager

def process_gold_layer(spark, silver_path, bq_table, label, config, is_stock=False, df_input=None):
    logger.info(f"🏆 Calcul de la couche Gold pour : {label}")
    
    # 1. Lecture Silver (depuis chemin ou DF direct)
    if df_input is not None:
        df_silver = df_input
    else:
        df_silver = spark.read.format("delta").load(silver_path)
        
    df_silver.createOrReplaceTempView("temp_silver_data")
    
    # 2. Transformation SQL (Injection des paramètres via f-string)
    query = QUERY_COMPUTE_GOLD_INDICATORS.format(
        sma_fast_p=config.get('stock_sma_fast' if is_stock else 'etf_sma_fast', 13) - 1,
        sma_slow_p=config.get('stock_sma_slow' if is_stock else 'etf_sma_slow', 50) - 1,
        mom_p=config.get('stock_mom_period' if is_stock else 'etf_mom_period', 20),
        atr_p=3, # Fenêtre ATR 4 (N-1 = 3)
        adx_p=8  # Fenêtre ADX 9 (N-1 = 8)
    )
    
    df_gold = spark.sql(query)
    
    # 3. Calcul Éligibilité (uniquement pour les actions)
    if is_stock:
        df_gold = df_gold.withColumn("Eligible", 
            (df_gold.Close > df_gold.SMA_slow) & 
            (df_gold.SMA_fast > df_gold.SMA_slow) & 
            (df_gold.ADX > config.get('stock_adx_threshold', 20.0)) & 
            (df_gold.ATR_pct < config.get('stock_atr_threshold', 20.0))
        )
    
    # 4. Validation
    QualityManager.validate_gold_data(df_gold, label)
    
    # 5. Écriture BigQuery
    logger.info(f"📤 Injection de {label} dans BigQuery {bq_table}...")
    df_gold.write.format("bigquery") \
        .option("table", bq_table) \
        .option("temporaryGcsBucket", BQ_TEMP_BUCKET) \
        .mode("overwrite") \
        .save()

def main():
    spark = create_spark_session(app_name="Gold_Layer_Unified")
    spark.conf.set("spark.sql.parquet.writeLegacyFormat", "true")
    config = get_champion_config()
    
    # 1. Unification des Univers Actions (S&P 500 + Universe 2B)
    logger.info("🔗 Fusion des univers S&P 500 et 2B...")
    df_sp500 = spark.read.format("delta").load(Paths.SP500_STOCK_PRICES_WEEKLY_SILVER)
    df_2b = spark.read.format("delta").load(Paths.DATA_RAW_2B_WEEKLY_SILVER)
    
    # Union des deux DataFrames et suppression des doublons (Ticker + Date)
    df_unified_stocks = df_sp500.unionByName(df_2b, allowMissingColumns=True) \
                                .dropDuplicates(['Ticker', 'Date'])
    
    # 2. Traitement de la couche Gold
    process_gold_layer(spark, None, Paths.BQ_STOCKS_GOLD, "Combined Stocks (SP500 + 2B)", config, is_stock=True, df_input=df_unified_stocks)
    process_gold_layer(spark, Paths.DATA_RAW_ETF_WEEKLY_SILVER, Paths.BQ_ETF_GOLD, "ETFs", config, is_stock=False)
    process_gold_layer(spark, Paths.DATA_RAW_SP500_WEEKLY_SILVER, Paths.BQ_SP500_GOLD, "Index ^GSPC", config, is_stock=False)

    logger.success("✨ Couche GOLD unifiée (SP500 + 2B) injectée dans BigQuery !")

if __name__ == "__main__":
    main()
