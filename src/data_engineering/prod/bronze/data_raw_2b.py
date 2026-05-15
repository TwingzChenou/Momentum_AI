import os
import sys
import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType
from pyspark.sql.functions import col, to_date

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def get_max_date_from_lake(spark, path):
    try:
        df = spark.read.format("delta").load(path)
        max_date = df.selectExpr("max(Date)").collect()[0][0]
        if max_date:
            logger.info(f"📍 Last data in Lake: {max_date}")
            return str(max_date), True
    except Exception:
        logger.warning(f"⚠️ No existing table found at {path}. Full load required.")
    return None, False

def get_tickers_from_lake(spark):
    logger.info(f"📡 Loading active tickers from {Paths.TICKERS_2B_CONSOLIDATED_HISTORY}...")
    try:
        # On ne prend que les tickers qui sont actuellement dans l'univers (Date_end est NULL)
        df = spark.read.format("delta").load(Paths.TICKERS_2B_CONSOLIDATED_HISTORY)
        symbols = [row['Ticker'] for row in df.filter(col("Date_end").isNull()).select('Ticker').distinct().collect()]
        logger.info(f"✅ Loaded {len(symbols)} active tickers from consolidated history.")
        return symbols
    except Exception as e:
        logger.error(f"❌ Error loading tickers from history: {e}")
        # Fallback sur la liste brute si l'historique n'existe pas encore
        try:
            logger.warning(f"⚠️ Falling back to raw list: {Paths.LIST_TICKER_2B}")
            df_raw = spark.read.format("delta").load(Paths.LIST_TICKER_2B)
            return [row['symbol'] for row in df_raw.select('symbol').distinct().collect()]
        except:
            return []

def fetch_individual_with_retry(ticker, start_date, end_date, period):
    logger.info(f"🔄 Retry individuel pour : {ticker}...")
    try:
        time.sleep(1.0)
        if start_date:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        else:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
            
        if df.empty or df['Close'].isnull().all():
            logger.error(f"❌ Échec persistant pour {ticker}")
            return pd.DataFrame()
            
        df['Ticker'] = ticker
        return df.reset_index()
    except Exception as e:
        logger.error(f"⚠️ Erreur retry {ticker} : {e}")
        return pd.DataFrame()

def fetch_data_in_chunks(tickers, start_date=None, period="2y", chunk_size=100):
    logger.info(f"🚀 Ingestion Bronze pour {len(tickers)} tickers (Start: {start_date if start_date else period})")
    all_data = []
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        logger.info(f"⏱️ Chunk {i // chunk_size + 1}/{(len(tickers) - 1) // chunk_size + 1} (Taille: {len(chunk)})")
        
        try:
            # Note: yf.download affiche souvent des erreurs dans la console sans lever d'exception
            if start_date:
                df = yf.download(tickers=chunk, start=start_date, end=end_date, group_by="ticker", auto_adjust=False, progress=False, threads=True)
            else:
                df = yf.download(tickers=chunk, period=period, group_by="ticker", auto_adjust=False, progress=False, threads=True)
            
            if df.empty:
                logger.error(f"❌ Erreur critique : Le téléchargement du chunk {i} a renvoyé un résultat vide. (Probable Rate Limit)")
                continue
                
            chunk_processed = []
            failed_in_chunk = []
            
            # Analyse par ticker pour identifier les "trous"
            if len(chunk) == 1:
                ticker = chunk[0]
                if df['Close'].isnull().all():
                    logger.warning(f"⚠️ Données vides pour {ticker}")
                    failed_in_chunk.append(ticker)
                else:
                    df['Ticker'] = ticker
                    chunk_processed.append(df.reset_index())
            else:
                # Dans un téléchargement multi-tickers, df a un MultiIndex en colonnes
                downloaded_tickers = df.columns.levels[0] if hasattr(df.columns, 'levels') else []
                
                for ticker in chunk:
                    if ticker not in downloaded_tickers:
                        logger.warning(f"❌ Ticker {ticker} ABSENT de la réponse Yahoo Finance.")
                        failed_in_chunk.append(ticker)
                        continue
                    
                    ticker_df = df[ticker].dropna(subset=['Close'])
                    if ticker_df.empty:
                        logger.warning(f"⚠️ Ticker {ticker} trouvé mais 100% de valeurs NaN (Rate Limit ?).")
                        failed_in_chunk.append(ticker)
                    else:
                        ticker_df = ticker_df.copy()
                        ticker_df['Ticker'] = ticker
                        chunk_processed.append(ticker_df.reset_index())

            # --- LOG ET RETRY ---
            if failed_in_chunk:
                logger.warning(f"🛑 Échecs détectés sur {len(failed_in_chunk)} tickers : {failed_in_chunk}")
                for t in failed_in_chunk:
                    retry_df = fetch_individual_with_retry(t, start_date, end_date, period)
                    if not retry_df.empty:
                        chunk_processed.append(retry_df)
                        logger.success(f"✅ Retry réussi pour {t}")
                    else:
                        logger.error(f"💀 Échec définitif pour {t}")

            all_data.extend(chunk_processed)
            time.sleep(1.0) # Augmentation du délai pour plus de sécurité
            
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

def process_data(df_daily, all_tickers):
    logger.info("🔧 Traitement des données et rééchantillonnage...")
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    for col in numeric_cols:
        df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
        
    if df_daily.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    min_date = df_daily['Date'].min()
    max_date = df_daily['Date'].max()

    # --- 1. DAILY ---
    master_dates_daily = pd.date_range(start=min_date, end=max_date, freq='B')
    master_df_daily = pd.MultiIndex.from_product([all_tickers, master_dates_daily], names=['Ticker', 'Date']).to_frame(index=False)
    final_daily_df = pd.merge(master_df_daily, df_daily, on=['Ticker', 'Date'], how='left')
    
    # --- 2. WEEKLY ---
    df_for_weekly = df_daily.set_index('Date')
    resampled = df_for_weekly.groupby('Ticker').resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'AdjClose': 'last', 'Volume': 'sum'
    }).reset_index()
    master_dates_weekly = pd.date_range(start=min_date, end=max_date, freq='W-FRI')
    master_df_weekly = pd.MultiIndex.from_product([all_tickers, master_dates_weekly], names=['Ticker', 'Date']).to_frame(index=False)
    final_weekly_df = pd.merge(master_df_weekly, resampled, on=['Ticker', 'Date'], how='left')
    
    # --- 3. MONTHLY ---
    resampled_monthly = df_for_weekly.groupby('Ticker').resample('BM').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'AdjClose': 'last', 'Volume': 'sum'
    }).reset_index()
    master_dates_monthly = pd.date_range(start=min_date, end=max_date, freq='BM')
    master_df_monthly = pd.MultiIndex.from_product([all_tickers, master_dates_monthly], names=['Ticker', 'Date']).to_frame(index=False)
    final_monthly_df = pd.merge(master_df_monthly, resampled_monthly, on=['Ticker', 'Date'], how='left')
    
    return final_daily_df, final_weekly_df, final_monthly_df

def save_to_lake(spark, pandas_df, path):
    if pandas_df.empty:
        return
    logger.info(f"💾 Sauvegarde de {pandas_df.shape[0]} lignes vers {path}...")
    pandas_df['Date'] = pandas_df['Date'].dt.strftime('%Y-%m-%d')
    pandas_df['Ticker'] = pandas_df['Ticker'].astype(str)
    for col_name in ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']:
        pandas_df[col_name] = pandas_df[col_name].astype(float)

    try:
        sdf = spark.createDataFrame(pandas_df)
        sdf = sdf.withColumn("Date", to_date(col("Date"))) \
                 .withColumn("Ticker", col("Ticker").cast(StringType())) \
                 .withColumn("Open", col("Open").cast(DoubleType())) \
                 .withColumn("High", col("High").cast(DoubleType())) \
                 .withColumn("Low", col("Low").cast(DoubleType())) \
                 .withColumn("Close", col("Close").cast(DoubleType())) \
                 .withColumn("AdjClose", col("AdjClose").cast(DoubleType())) \
                 .withColumn("Volume", col("Volume").cast(LongType()))
                 
        from delta.tables import DeltaTable
        if DeltaTable.isDeltaTable(spark, path):
            dt = DeltaTable.forPath(spark, path)
            dt.alias("target").merge(
                sdf.alias("source"),
                "target.Date = source.Date AND target.Ticker = source.Ticker"
            ).whenMatchedUpdateAll() \
             .whenNotMatchedInsertAll() \
             .execute()
        else:
            sdf.write.format("delta").mode("overwrite").save(path)
        logger.info(f"✅ Sauvegarde réussie : {path}")
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde : {e}")
        raise e

def main():
    setup_logging()
    logger.info("🚀 Job Bronze 2B")
    spark = create_spark_session(app_name="Data_Raw_2B_Ingestion")
    try:
        tickers = get_tickers_from_lake(spark)
        if not tickers: return
        last_date, is_inc = get_max_date_from_lake(spark, Paths.DATA_RAW_2B)
        fetch_start = None
        if last_date:
            from datetime import timedelta
            fetch_start = (datetime.strptime(last_date, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')
        
        df_daily = fetch_data_in_chunks(tickers, start_date=fetch_start, period="2y", chunk_size=100)
        if df_daily.empty: return
        df_d, df_w, df_m = process_data(df_daily, tickers)
        save_to_lake(spark, df_d, Paths.DATA_RAW_2B)
        save_to_lake(spark, df_w, Paths.DATA_RAW_2B_WEEKLY)
        save_to_lake(spark, df_m, Paths.DATA_RAW_2B_MONTHLY)
    except Exception as e:
        logger.critical(f"❌ Erreur : {e}")
        sys.exit(1)
    finally:
        if spark: spark.stop()

if __name__ == "__main__":
    main()
