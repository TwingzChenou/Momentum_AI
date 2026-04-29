import os
import sys
import pandas as pd
from loguru import logger

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

import time

def process_daily_to_weekly(df_daily):
    """Resamples daily data to weekly (Friday) and cleans it."""
    start_resample = time.time()
    logger.info(f"🔧 Rééchantillonnage de {len(df_daily)} lignes quotidiennes en hebdomadaire (W-FRI)...")
    
    # --- STANDARDIZATION PASCAL CASE ---
    rename_map = {
        'symbol': 'Ticker', 'ticker': 'Ticker',
        'date': 'Date',
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
        'adjClose': 'AdjClose', 'adj_close': 'AdjClose'
    }
    df_daily = df_daily.rename(columns=rename_map)
    
    # Ensure mandatory columns exist
    for col in ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']:
        if col not in df_daily.columns:
            logger.warning(f"⚠️ Colonne {col} manquante, recherche par nom insensible à la casse...")
            for c in df_daily.columns:
                if c.lower() == col.lower():
                    df_daily = df_daily.rename(columns={c: col})
                    break

    # Ensure Date is datetime
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily = df_daily.set_index('Date')
    
    # Resample by Ticker
    resampled = df_daily.groupby('Ticker').resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'AdjClose': 'last',
        'Volume': 'sum'
    }).reset_index()
    
    # Cleaning: Use AdjClose as the main Close for backtesting
    resampled = resampled.rename(columns={'Close': 'Close_Raw', 'AdjClose': 'Close'})
    
    # Final PascalCase verification
    expected_cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close_Raw', 'Close', 'Volume']
    resampled = resampled[expected_cols]
    
    # Drop rows where Close is NaN (tickers not active)
    initial_len = len(resampled)
    resampled = resampled.dropna(subset=['Close'])
    
    duration = time.time() - start_resample
    logger.info(f"✨ Rééchantillonnage terminé en {duration:.2f}s. {len(resampled)} points hebdomadaires générés (NaN supprimés: {initial_len - len(resampled)}).")
    return resampled

def main():
    start_time = time.time()
    setup_logging()
    logger.info("🎬 Démarrage de la pipeline Silver : Transformation Hebdomadaire")
    
    spark = create_spark_session("SP500_Prices_Weekly_Silver")
    
    try:
        # 1. Load Stocks (Bronze Raw)
        logger.info(f"📥 Chargement des prix Bronze depuis {Paths.SP500_STOCK_PRICES}")
        sdf_stocks = spark.read.format("delta").load(Paths.SP500_STOCK_PRICES)
        
        # 2. Load History (Silver) to filter by inclusion dates
        logger.info(f"📥 Chargement de l'historique consolidé depuis {Paths.SP500_CONSOLIDATED_HISTORY}")
        sdf_history = spark.read.format("delta").load(Paths.SP500_CONSOLIDATED_HISTORY)
        
        # 3. Filter stocks: only keep data when the stock was in the index
        import pyspark.sql.functions as F
        sdf_stocks = sdf_stocks.withColumnRenamed("symbol", "Ticker").withColumnRenamed("date", "Date")
        
        initial_count = sdf_stocks.count()
        logger.info(f"⚡ Filtrage de {initial_count} lignes par périodes d'inclusion historique...")
        
        sdf_stocks_filtered = sdf_stocks.join(
            sdf_history,
            on="Ticker",
            how="inner"
        ).filter(
            (F.col("Date") >= F.col("Date_start")) & 
            (F.col("Date") <= F.col("Date_end"))
        ).select(sdf_stocks.columns)
        
        filtered_count = sdf_stocks_filtered.count()
        logger.info(f"✅ Filtrage terminé : {filtered_count} lignes conservées ({initial_count - filtered_count} lignes hors-index supprimées).")
        
        # 4. Load Index (^GSPC) benchmark
        logger.info(f"📥 Chargement du benchmark S&P 500 Index...")
        sdf_index = spark.read.format("delta").load(Paths.DATA_RAW_SP500)
        sdf_index = sdf_index.withColumnRenamed("symbol", "Ticker").withColumnRenamed("date", "Date")
        
        # 5. Combine Stocks and Index
        sdf_combined = sdf_stocks_filtered.unionByName(sdf_index, allowMissingColumns=True)
        
        # 6. Convert to Pandas for resampling
        logger.info("🐼 Conversion Spark -> Pandas pour rééchantillonnage...")
        df_daily = sdf_combined.toPandas()
        
        if df_daily.empty:
            logger.warning("⚠️ Aucune donnée disponible après filtrage.")
            return

        # 7. Process
        df_weekly = process_daily_to_weekly(df_daily)
        
        # 8. Save to Delta
        save_start = time.time()
        sdf_weekly = spark.createDataFrame(df_weekly)
        
        logger.info(f"💾 Sauvegarde de {len(df_weekly)} lignes dans les tables Silver...")
        sdf_weekly.write.format("delta").mode("overwrite").save(Paths.SP500_STOCK_PRICES_WEEKLY_SILVER)
        sdf_weekly.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(Paths.SP500_STOCK_PRICES_SILVER)
        
        save_duration = time.time() - save_start
        logger.success(f"💾 Sauvegarde Silver terminée en {save_duration:.2f}s")

    except Exception as e:
        logger.critical(f"❌ Erreur critique Silver : {e}")
        sys.exit(1)
    finally:
        total_duration = time.time() - start_time
        logger.info(f"🏁 Fin de la pipeline Silver. Durée totale : {total_duration:.2f}s")
        if spark: spark.stop()

if __name__ == "__main__":
    main()
