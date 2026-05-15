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
    Determines which tickers need updates, including historical members to avoid survivor bias.
    """
    logger.info(f"📡 Calculating incremental tasks (NO BIAS: 1173+ Historical Members)...")
    today = datetime.today().date()
    
    # 1. Load ALL Members (Current + Historical)
    try:
        df_history = spark.read.format("delta").load(Paths.SP500_CONSOLIDATED_HISTORY) \
                          .select("Ticker", "Date_start", "Date_end")
        logger.info(f"✅ Loaded {df_history.count()} unique tickers from consolidated history.")
    except Exception as e:
        logger.error(f"❌ Could not load history: {e}")
        return []
    
    # 2. Get existing max dates
    try:
        df_existing = spark.read.format("delta").load(Paths.SP500_STOCK_PRICES)
        if "symbol" in df_existing.columns:
            df_existing = df_existing.withColumnRenamed("symbol", "Ticker").withColumnRenamed("date", "Date")
        df_max = df_existing.groupBy("Ticker").agg(F.max("Date").alias("LastDate"))
    except:
        df_max = spark.createDataFrame([], "Ticker string, LastDate date")

    # 3. Join History with existing data
    df_tasks = df_history.join(df_max, on="Ticker", how="left")
    
    # 4. Global floor: 1976
    global_floor = datetime(1976, 1, 1).date()
    
    # 5. Calculate Effective Start and End
    # - Start: max(global_floor, Date_start, LastDate + 1)
    # - End: min(today, Date_end if not null)
    df_tasks = df_tasks.withColumn("Start", 
        F.when(F.col("LastDate").isNotNull(), F.date_add(F.col("LastDate"), 1))
         .otherwise(F.greatest(F.lit(global_floor), F.col("Date_start")))
    )
    
    df_tasks = df_tasks.withColumn("End", 
        F.when(F.col("Date_end").isNotNull(), F.col("Date_end"))
         .otherwise(F.lit(today))
    )
    
    # Filter: Start < End
    df_tasks = df_tasks.filter(F.col("Start") < F.col("End"))
    
    tasks = [(row['Ticker'], str(row['Start']), str(row['End'])) for row in df_tasks.collect()]
    return tasks

def fetch_yf_data_incremental(tasks, chunk_size=5):
    """Fetches data with User-Agent rotation and medium-conservative timing."""
    import requests
    import random
    
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    ]
    
    session = requests.Session()
    all_dfs = []
    task_groups = {}
    for ticker, start, end in tasks:
        key = (start, end)
        if key not in task_groups: task_groups[key] = []
        task_groups[key].append(ticker)

    for (start, end), tickers in task_groups.items():
        logger.info(f"📅 Fetching {len(tickers)} tickers for range {start} to {end}")
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            session.headers.update({"User-Agent": random.choice(user_agents)})
            
            success = False
            for attempt in range(5):
                try:
                    df = yf.download(chunk, start=start, end=end, group_by='ticker', threads=False, progress=False, session=session)
                    if not df.empty:
                        if len(chunk) > 1:
                            df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
                        else:
                            df['Ticker'] = chunk[0]
                            df = df.reset_index()
                        all_dfs.append(df)
                        success = True
                        break
                    else:
                        success = True 
                        break
                except Exception as e:
                    wait = (attempt + 1) * 30
                    logger.warning(f"⚠️ Attempt {attempt+1} failed for {chunk}: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            
            if i + chunk_size < len(tickers):
                time.sleep(5) # Polite wait of 5s between chunks

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def main():
    start_time = time.time()
    setup_logging()
    logger.info("🎬 Démarrage de la pipeline Bronze : Ingestion des prix S&P 500")
    
    spark = create_spark_session("SP500_Prices_Daily_Incremental")
    
    try:
        # 1. Identify what needs to be downloaded
        logger.info("🔍 Étape 1 : Identification des données manquantes (Incrémental)...")
        tasks = get_incremental_tasks(spark)
        
        if not tasks:
            logger.success("✅ Tout est à jour. Aucun téléchargement nécessaire.")
            return

        logger.info(f"📋 {len(tasks)} tickers nécessitent une mise à jour.")
        
        # 2. Fetch Data
        logger.info("🌐 Étape 2 : Téléchargement des données depuis Yahoo Finance...")
        fetch_start = time.time()
        df_new = fetch_yf_data_incremental(tasks, chunk_size=2)
        fetch_duration = time.time() - fetch_start
        
        if df_new.empty:
            logger.warning("⚠️ Aucun nouveau prix récupéré après tentative de téléchargement.")
            return

        logger.info(f"📊 Téléchargement terminé : {len(df_new)} lignes récupérées en {fetch_duration:.2f}s")
        
        # 3. Préparation des données
        logger.info("🛠️ Étape 3 : Transformation et Standardisation des colonnes...")
        final_df = pd.DataFrame()
        
        # Mapping robuste et typage forcé pour éviter les erreurs Spark (CANNOT_ACCEPT_OBJECT_IN_TYPE)
        final_df['Ticker'] = df_new['Ticker']
        final_df['Date'] = pd.to_datetime(df_new['Date']).dt.date
        
        # Forcer les prix en float (DoubleType)
        for col in ['Open', 'High', 'Low', 'Close']:
            final_df[col] = pd.to_numeric(df_new[col], errors='coerce').astype(float)
        
        final_df['AdjClose'] = pd.to_numeric(df_new['Adj Close'] if 'Adj Close' in df_new.columns else df_new['Close'], errors='coerce').astype(float)
        
        # Forcer le Volume en int (LongType) - Remplissage des NaNs par 0 pour permettre la conversion
        final_df['Volume'] = pd.to_numeric(df_new['Volume'], errors='coerce').fillna(0).astype(int)
        
        logger.info(f"✨ Colonnes standardisées et typées : {list(final_df.columns)}")
        
        # 4. Sauvegarde Sécurisée (AUCUN OVERWRITE AUTORISÉ)
        from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType
        from delta.tables import DeltaTable
        
        # Définition explicite du schéma pour éviter les erreurs d'inférence (CANNOT_MERGE_TYPE)
        schema = StructType([
            StructField("Ticker", StringType(), True),
            StructField("Date", DateType(), True),
            StructField("Open", DoubleType(), True),
            StructField("High", DoubleType(), True),
            StructField("Low", DoubleType(), True),
            StructField("Close", DoubleType(), True),
            StructField("AdjClose", DoubleType(), True),
            StructField("Volume", LongType(), True)
        ])

        logger.info("🚀 Conversion du DataFrame Pandas en Spark avec schéma explicite...")
        # S'assurer que les dates sont au format datetime pour Spark
        final_df['Date'] = pd.to_datetime(final_df['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'AdjClose']:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').astype('float64')
        # -----------------------------------------------

        logger.info("🚀 Conversion du DataFrame Pandas en Spark avec schéma explicite...")
        sdf_new = spark.createDataFrame(final_df, schema=schema)
        
        save_start = time.time()
        
        if DeltaTable.isDeltaTable(spark, Paths.SP500_STOCK_PRICES):
            logger.info(f"🔄 Étape 4 : Upsert (Merge) dans la table Delta : {Paths.SP500_STOCK_PRICES}")
            dt = DeltaTable.forPath(spark, Paths.SP500_STOCK_PRICES)
            
            dt.alias("old").merge(
                sdf_new.alias("new"),
                "old.Ticker = new.Ticker AND old.Date = new.Date"
            ).whenNotMatchedInsertAll().execute()
        else:
            # Si la table n'existe pas, on utilise 'append' au lieu de 'overwrite' 
            # pour éviter d'effacer des fichiers qui pourraient être là par erreur
            logger.info(f"🆕 Étape 4 : Création/Append dans la table Delta : {Paths.SP500_STOCK_PRICES}")
            sdf_new.write.format("delta").mode("append").save(Paths.SP500_STOCK_PRICES)
        
        save_duration = time.time() - save_start
        logger.success(f"💾 Sauvegarde terminée avec succès en {save_duration:.2f}s")

    except Exception as e:
        logger.critical(f"❌ Erreur critique lors de l'exécution : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        total_duration = time.time() - start_time
        logger.info(f"🏁 Fin de la pipeline Bronze. Durée totale : {total_duration:.2f}s")
        if spark: spark.stop()

if __name__ == "__main__":
    # Augmenter la résilience pour le téléchargement massif
    # Utiliser un chunk_size très petit et des pauses pour éviter le Rate Limit
    main()
