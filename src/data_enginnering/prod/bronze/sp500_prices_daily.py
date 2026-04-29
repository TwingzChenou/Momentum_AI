import os
import sys
import pandas as pd
import yfinance as yf
from loguru import logger
from datetime import datetime, timedelta
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths
import pyspark.sql.functions as F

def get_incremental_tasks(spark):
    """
    Determines which tickers need updates.
    Logic: 
    - ONLY current members (503 tickers).
    - Start date is at least March 19, 2026.
    """
    logger.info(f"📡 Calculating incremental tasks (STRICT: 503 Current Members)...")
    
    # 1. Load Current Members List
    df_latest = spark.read.format("delta").load(Paths.SP500_LATEST_TICKERS).select(F.col("symbol").alias("Ticker"))
    
    global_max_date = None
    try:
        df_existing = spark.read.format("delta").load(Paths.SP500_STOCK_PRICES)
        if "symbol" in df_existing.columns:
            df_existing = df_existing.withColumnRenamed("symbol", "Ticker").withColumnRenamed("date", "Date")
        
        df_max = df_existing.groupBy("Ticker").agg(F.max("Date").alias("LastDate"))
    except:
        df_max = spark.createDataFrame([], "Ticker string, LastDate date")

    # On fixe la date de début minimale au 1er Janvier 2025
    start_floor = datetime(2025, 1, 1).date()
    
    # 2. Join (Left Join to keep ALL current members)
    df_tasks = df_latest.join(df_max, on="Ticker", how="left")
    
    # 3. Calculate Effective Start
    df_tasks = df_tasks.withColumn("EffectiveStart", 
        F.when(F.col("LastDate").isNotNull(), F.date_add(F.col("LastDate"), 1))
         .otherwise(F.lit(start_floor))
    )
    
    # Final check: Start must be >= start_floor
    df_tasks = df_tasks.withColumn("EffectiveStart", 
        F.when(F.col("EffectiveStart") < F.lit(start_floor), F.lit(start_floor)).otherwise(F.col("EffectiveStart"))
    )
    
    # Final filter: Start < Today
    df_tasks = df_tasks.filter(F.col("EffectiveStart") < today)
    
    tasks = [(row['Ticker'], str(row['EffectiveStart'])) for row in df_tasks.collect()]
    return tasks

def fetch_yf_data_incremental(tasks, chunk_size=20):
    """Fetches data for specific tickers starting from their respective last dates."""
    all_data = []
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # To keep it efficient, we group by start_date
    from collections import defaultdict
    date_groups = defaultdict(list)
    for ticker, start_date in tasks:
        date_groups[start_date].append(ticker)
        
    for start_date, tickers in date_groups.items():
        logger.info(f"📅 Downloading {len(tickers)} tickers starting from {start_date}...")
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            
            for attempt in range(3):
                try:
                    df = yf.download(tickers=chunk, start=start_date, end=end_date, interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
                    if not df.empty:
                        if len(chunk) == 1:
                            df['Ticker'] = chunk[0]
                            df = df.reset_index()
                        else:
                            df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
                        all_data.append(df)
                        break
                    else:
                        time.sleep(2)
                except Exception as e:
                    logger.error(f"❌ Error: {e}")
                    time.sleep(5)
            time.sleep(1)
            
    if not all_data: return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def main():
    start_time = time.time()
    setup_logging()
    logger.info("🎬 Démarrage de la pipeline Bronze : Ingestion des prix S&P 500")
    
    spark = create_spark_session("SP500_Prices_Daily_Incremental")
    
    try:
        # 1. Identify what needs to be downloaded
        tasks = get_incremental_tasks(spark)
        
        if not tasks:
            logger.success("✅ Tout est à jour. Aucun téléchargement nécessaire.")
            return

        logger.info(f"📋 {len(tasks)} tickers nécessitent une mise à jour.")
        
        # 2. Fetch Data
        fetch_start = time.time()
        df_new = fetch_yf_data_incremental(tasks)
        fetch_duration = time.time() - fetch_start
        
        if df_new.empty:
            logger.warning("⚠️ Aucun nouveau prix récupéré après tentative de téléchargement.")
            return

        logger.info(f"📊 Téléchargement terminé : {len(df_new)} lignes récupérées en {fetch_duration:.2f}s")
        
        # 3. Préparation des données
        logger.info("🛠️ Alignement du schéma sur le standard CamelCase...")
        final_df = pd.DataFrame()
        final_df['symbol'] = df_new['Ticker'] if 'Ticker' in df_new.columns else df_new['symbol']
        final_df['date'] = pd.to_datetime(df_new['Date'] if 'Date' in df_new.columns else df_new['date']).dt.date
        final_df['adjOpen'] = df_new['Open']
        final_df['adjHigh'] = df_new['High']
        final_df['adjLow'] = df_new['Low']
        final_df['adjClose'] = df_new['Adj Close'] if 'Adj Close' in df_new.columns else df_new['Close']
        final_df['volume'] = df_new['Volume']
        
        sdf_new = spark.createDataFrame(final_df)
        
        # 4. Sauvegarde avec Merge (pour éviter les doublons)
        from delta.tables import DeltaTable
        save_start = time.time()
        if DeltaTable.isDeltaTable(spark, Paths.SP500_STOCK_PRICES):
            logger.info(f"🔄 Upsert en cours via Delta Merge dans {Paths.SP500_STOCK_PRICES}...")
            dt = DeltaTable.forPath(spark, Paths.SP500_STOCK_PRICES)
            
            dt.alias("old").merge(
                sdf_new.alias("new"),
                "old.symbol = new.symbol AND old.date = new.date"
            ).whenNotMatchedInsertAll().execute()
        else:
            logger.info(f"🆕 Création de la table Delta : {Paths.SP500_STOCK_PRICES}")
            sdf_new.write.format("delta").mode("overwrite").save(Paths.SP500_STOCK_PRICES)
        
        save_duration = time.time() - save_start
        logger.success(f"💾 Sauvegarde terminée avec succès en {save_duration:.2f}s")

    except Exception as e:
        logger.critical(f"❌ Erreur critique lors de l'exécution : {e}")
        sys.exit(1)
    finally:
        total_duration = time.time() - start_time
        logger.info(f"🏁 Fin de la pipeline Bronze. Durée totale : {total_duration:.2f}s")
        if spark: spark.stop()

if __name__ == "__main__":
    main()
