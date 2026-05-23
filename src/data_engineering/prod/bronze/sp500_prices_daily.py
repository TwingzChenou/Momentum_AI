import os
import sys
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from loguru import logger
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType
from pyspark.sql.functions import col, to_date
import pyspark.sql.functions as F
from delta.tables import DeltaTable

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def get_active_tickers(spark):
    """
    Load all active S&P 500 tickers currently present in the index from consolidated history.
    """
    logger.info(f"📡 Loading active S&P 500 tickers from {Paths.SP500_CONSOLIDATED_HISTORY}...")
    try:
        df_history = spark.read.format("delta").load(Paths.SP500_CONSOLIDATED_HISTORY)
        # Tickers currently in the index (Date_end is null)
        tickers = [row['Ticker'] for row in df_history.filter(col("Date_end").isNull()).select("Ticker").distinct().collect()]
        if tickers:
            logger.info(f"✅ Loaded {len(tickers)} active S&P 500 tickers from consolidated history.")
            return tickers
    except Exception as e:
        logger.warning(f"⚠️ Could not load active tickers from history: {e}")
        
    # Fallback to the latest constituents list
    try:
        logger.info(f"📡 Falling back to latest constituents list from {Paths.SP500_LATEST_TICKERS}...")
        df_latest = spark.read.format("delta").load(Paths.SP500_LATEST_TICKERS)
        tickers = [row['symbol'] for row in df_latest.select("symbol").distinct().collect()]
        logger.info(f"✅ Loaded {len(tickers)} active S&P 500 tickers from latest constituents list.")
        return tickers
    except Exception as e:
        logger.error(f"❌ Failed to load tickers from fallback list: {e}")
        return []

def get_max_date_from_lake(spark, path):
    """
    Gets the maximum Date present in the destination price table for incremental sync.
    """
    try:
        df = spark.read.format("delta").load(path)
        max_date = df.selectExpr("max(Date)").collect()[0][0]
        if max_date:
            logger.info(f"📍 Last price date in Lake: {max_date}")
            return max_date
    except Exception:
        logger.warning(f"⚠️ No existing table found at {path}. Full load required.")
    return None

def fetch_individual_with_retry(ticker, start_date, end_date):
    """
    Individual fallback download when a multi-ticker chunk download fails for a specific ticker.
    """
    logger.info(f"🔄 Retry individuel pour : {ticker}...")
    try:
        time.sleep(1.0)
        if start_date:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        else:
            df = yf.download(ticker, period="2y", progress=False, auto_adjust=False)
            
        if df.empty or df['Close'].isnull().all():
            logger.error(f"❌ Échec persistant pour {ticker}")
            return pd.DataFrame()
            
        df['Ticker'] = ticker
        return df.reset_index()
    except Exception as e:
        logger.error(f"⚠️ Erreur retry {ticker} : {e}")
        return pd.DataFrame()

def fetch_sp500_prices(tickers, start_date=None, chunk_size=100):
    """
    Downloads stock prices in efficient bulk chunks using yfinance.
    """
    logger.info(f"🚀 Ingestion Bronze S&P 500 pour {len(tickers)} tickers (Début : {start_date if start_date else '2y'})")
    all_data = []
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        logger.info(f"⏱️ Chunk {i // chunk_size + 1}/{(len(tickers) - 1) // chunk_size + 1} (Taille: {len(chunk)})")
        
        try:
            if start_date:
                df = yf.download(tickers=chunk, start=start_date, end=end_date, group_by="ticker", auto_adjust=False, progress=False, threads=True)
            else:
                df = yf.download(tickers=chunk, period="2y", group_by="ticker", auto_adjust=False, progress=False, threads=True)
            
            if df.empty:
                logger.error(f"❌ Résultat vide pour le chunk {i // chunk_size + 1}.")
                continue
                
            chunk_processed = []
            failed_in_chunk = []
            
            if len(chunk) == 1:
                ticker = chunk[0]
                if df['Close'].isnull().all():
                    logger.warning(f"⚠️ Données vides pour {ticker}")
                    failed_in_chunk.append(ticker)
                else:
                    df['Ticker'] = ticker
                    chunk_processed.append(df.reset_index())
            else:
                downloaded_tickers = df.columns.levels[0] if hasattr(df.columns, 'levels') else []
                
                for ticker in chunk:
                    if ticker not in downloaded_tickers:
                        logger.warning(f"❌ Ticker {ticker} ABSENT de la réponse Yahoo Finance.")
                        failed_in_chunk.append(ticker)
                        continue
                    
                    ticker_df = df[ticker].dropna(subset=['Close'])
                    if ticker_df.empty:
                        logger.warning(f"⚠️ Ticker {ticker} trouvé mais 100% de valeurs NaN.")
                        failed_in_chunk.append(ticker)
                    else:
                        ticker_df = ticker_df.copy()
                        ticker_df['Ticker'] = ticker
                        chunk_processed.append(ticker_df.reset_index())

            # Retry process for failed/Rate-Limited tickers
            if failed_in_chunk:
                logger.warning(f"🛑 Échecs détectés sur {len(failed_in_chunk)} tickers : {failed_in_chunk}")
                for t in failed_in_chunk:
                    retry_df = fetch_individual_with_retry(t, start_date, end_date)
                    if not retry_df.empty:
                        chunk_processed.append(retry_df)
                        logger.success(f"✅ Retry réussi pour {t}")
                    else:
                        logger.error(f"💀 Échec définitif pour {t}")

            all_data.extend(chunk_processed)
            time.sleep(1.0)
            
        except Exception as e:
            logger.error(f"🔥 Erreur inattendue lors du traitement du chunk : {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    final_df = pd.concat(all_data, ignore_index=True)
    if 'Adj Close' in final_df.columns:
        final_df = final_df.rename(columns={'Adj Close': 'AdjClose'})
        
    expected_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    for col_name in expected_cols:
        if col_name not in final_df.columns:
            final_df[col_name] = pd.NA
            
    return final_df[expected_cols]

def save_to_lake(spark, pandas_df, path):
    """
    Saves daily prices securely using a precise Spark Schema and Delta Table Merge.
    """
    if pandas_df.empty:
        logger.warning("⚠️ DataFrame vide, rien à sauvegarder.")
        return
    logger.info(f"💾 Sauvegarde de {pandas_df.shape[0]} lignes vers {path}...")
    
    # Standardize data types for Spark inference and schema compatibility
    pandas_df['Date'] = pd.to_datetime(pandas_df['Date'])
    pandas_df['Ticker'] = pandas_df['Ticker'].astype(str)
    for col_name in ['Open', 'High', 'Low', 'Close', 'AdjClose']:
        pandas_df[col_name] = pd.to_numeric(pandas_df[col_name], errors='coerce').astype(float)
    pandas_df['Volume'] = pd.to_numeric(pandas_df['Volume'], errors='coerce').fillna(0).astype('int64')

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

    try:
        sdf = spark.createDataFrame(pandas_df, schema=schema)
        
        if DeltaTable.isDeltaTable(spark, path):
            logger.info(f"🔄 Upsert (Merge) dans la table Delta : {path}")
            dt = DeltaTable.forPath(spark, path)
            dt.alias("old").merge(
                sdf.alias("new"),
                "old.Ticker = new.Ticker AND old.Date = new.Date"
            ).whenNotMatchedInsertAll().execute()
        else:
            logger.info(f"🆕 Création/Append dans la table Delta : {path}")
            sdf.write.format("delta").mode("append").save(path)
        logger.success(f"✅ Sauvegarde réussie : {path}")
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde : {e}")
        raise e

def main():
    start_time = time.time()
    setup_logging()
    logger.info("🎬 Démarrage de la pipeline Bronze : Ingestion RAPIDE et incrémentale du S&P 500")
    
    spark = create_spark_session("SP500_Prices_Daily_Fast_Incremental")
    try:
        # 1. Identify active constituents
        tickers = get_active_tickers(spark)
        if not tickers:
            logger.error("❌ Aucun ticker actif trouvé. Arrêt du job.")
            return
            
        # 2. Get existing last date in Lake
        max_date = get_max_date_from_lake(spark, Paths.SP500_STOCK_PRICES)
        fetch_start = None
        if max_date:
            # We go back 7 days to cover weekends, holidays, and minor lag. Delta merge handles deduplication.
            fetch_start = (max_date - timedelta(days=7)).strftime('%Y-%m-%d')
            logger.info(f"📅 Mise à jour incrémentale à partir du : {fetch_start}")
        else:
            logger.warning("⚠️ Pas d'historique trouvé, chargement complet (2 ans)...")
            
        # 3. Download daily prices in optimized chunks
        df_new = fetch_sp500_prices(tickers, start_date=fetch_start, chunk_size=100)
        
        # 4. Save/Merge into Delta Lake
        if not df_new.empty:
            save_to_lake(spark, df_new, Paths.SP500_STOCK_PRICES)
        else:
            logger.warning("⚠️ Aucun nouveau prix récupéré.")
            
    except Exception as e:
        logger.critical(f"❌ Erreur critique lors de l'exécution : {e}")
        sys.exit(1)
    finally:
        total_duration = time.time() - start_time
        logger.info(f"🏁 Fin de la pipeline Bronze. Durée totale : {total_duration:.2f}s")
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()
