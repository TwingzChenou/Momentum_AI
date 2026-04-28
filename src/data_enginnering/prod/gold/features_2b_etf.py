import os
import sys
import pandas as pd
import ta
import numpy as np
import warnings
from loguru import logger
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType

warnings.filterwarnings('ignore')

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from src.common.config_utils import get_champion_config
from config.config_spark import Paths, BQ_TEMP_BUCKET, GCP_PROJECT_ID, GCP_KEY_PATH
from src.common.validation import validate_df

# --- ETF PIPELINE ---
etf_schema = StructType([
    StructField("Date", DateType(), True),
    StructField("Ticker", StringType(), True),
    StructField("Close", DoubleType(), True),
    StructField("SMA_fast", DoubleType(), True),
    StructField("SMA_slow", DoubleType(), True),
    StructField("Momentum_XM", DoubleType(), True)
])

def process_etf_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    from loguru import logger
    if len(df) < max(config['etf_sma_slow'], config['etf_mom_period']) + 5:
        return pd.DataFrame(columns=[f.name for f in etf_schema.fields]).astype({
            'Date': 'str', 'Ticker': 'str', 'Close': 'float', 'SMA_fast': 'float', 
            'SMA_slow': 'float', 'Momentum_XM': 'float'
        })
    df = df.sort_values("Date")
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    # On utilise 'Close' comme seul standard
    price_col = 'Close'
    logger.info(f"📊 [ETF/Index] Ticker: {df['Ticker'].iloc[0] if not df.empty else 'Unknown'}, Rows: {len(df)}, Price Col: {price_col}")

    # Sécurité : On s'assure d'avoir des prix numériques
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    
    df['SMA_fast'] = ta.trend.sma_indicator(df[price_col], window=config['etf_sma_fast'])
    df['SMA_slow'] = ta.trend.sma_indicator(df[price_col], window=config['etf_sma_slow'])
    df['Momentum_XM'] = df[price_col].pct_change(periods=config['etf_mom_period'])
    
    df['Close'] = df[price_col]
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    # Debug dropna
    cols = [f.name for f in etf_schema.fields]
    before = len(df)
    res_df = df[cols].dropna()
    after = len(res_df)
    if after == 0 and before > 0:
        logger.warning(f"🚨 [ETF/Index] Tout a été filtré! Ticker: {df['Ticker'].iloc[0] if not df.empty else 'Unknown'}")
        logger.info(f"NaN count per col: {df[cols].isna().sum().to_dict()}")
        logger.info(f"Sample data (head 5):\n{df[price_col].head(5)}")
        
    return res_df

# --- STOCKS PIPELINE ---
stock_schema = StructType([
    StructField("Date", DateType(), True),
    StructField("Ticker", StringType(), True),
    StructField("Close", DoubleType(), True),
    StructField("ADX", DoubleType(), True),
    StructField("ATR", DoubleType(), True),
    StructField("SMA_fast", DoubleType(), True),
    StructField("SMA_slow", DoubleType(), True),
    StructField("ATR_pct", DoubleType(), True),
    StructField("Momentum_XM", DoubleType(), True)
])

def process_stock_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    from loguru import logger
    if len(df) < max(config['stock_sma_slow'], config['stock_mom_period']) + 10:
        logger.warning(f"⚠️ [Stock] {df['Ticker'].iloc[0] if not df.empty else 'Unknown'} : Pas assez de données ({len(df)} rows)")
        return pd.DataFrame(columns=[f.name for f in stock_schema.fields]).astype({
            'Date': 'str', 'Ticker': 'str', 'Close': 'float', 'ADX': 'float',
            'ATR': 'float', 'SMA_fast': 'float', 'SMA_slow': 'float',
            'ATR_pct': 'float', 'Momentum_XM': 'float'
        })
    # On utilise 'Close' comme seul standard
    p_col = 'Close'
    logger.info(f"📊 [Stock] Ticker: {df['Ticker'].iloc[0]}, Rows: {len(df)}, Price Col: {p_col}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    
    # 1. Daily ADX and ATR
    df['ADX_20'] = ta.trend.adx(df['High'], df['Low'], df[p_col], window=20, fillna=True)
    df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df[p_col], window=14, fillna=True)
        
    # 2. Resample to Weekly W-FRI
    weekly_df = df.resample('W-FRI', on='Date').agg({
        p_col: 'last',
        'ADX_20': 'last',
        'ATR_14': 'last'
    }).dropna(subset=[p_col])
    
    # 3. Calculate Weekly indicators 
    if len(weekly_df) >= 1:
        weekly_df['SMA_fast'] = ta.trend.sma_indicator(weekly_df[p_col], window=config['stock_sma_fast'])
        weekly_df['SMA_slow'] = ta.trend.sma_indicator(weekly_df[p_col], window=config['stock_sma_slow'])
        weekly_df['ATR_pct'] = weekly_df['ATR_14'] / weekly_df[p_col]
        weekly_df['Momentum_XM'] = weekly_df[p_col].pct_change(periods=config['stock_mom_period'])
    else:
        for col in ['SMA_fast', 'SMA_slow', 'ATR_pct', 'Momentum_XM']:
            weekly_df[col] = np.nan
            
    weekly_df = weekly_df.reset_index()
    weekly_df['Date'] = pd.to_datetime(weekly_df['Date']).dt.date
    weekly_df['Ticker'] = df['Ticker'].iloc[0] if not df.empty else ""
    
    # Renommage explicite pour ADX/ATR
    weekly_df = weekly_df.rename(columns={'ADX_20': 'ADX', 'ATR_14': 'ATR'})
    
    # Renommage final pour coller au schéma (Close est attendu en Gold)
    weekly_df = weekly_df.rename(columns={p_col: 'Close'})
    
    # Cast back to match Arrow schema
    for col_name in ['Close', 'ADX', 'ATR', 'SMA_fast', 'SMA_slow', 'ATR_pct', 'Momentum_XM']:
        weekly_df[col_name] = weekly_df[col_name].astype('float64')
        
    # Debug dropna
    cols = [f.name for f in stock_schema.fields]
    before = len(weekly_df)
    res_df = weekly_df[cols].dropna()
    after = len(res_df)
    if after == 0 and before > 0:
        logger.warning(f"🚨 [Stock] Tout a été filtré! Ticker: {weekly_df['Ticker'].iloc[0] if not weekly_df.empty else 'Unknown'}")
        logger.info(f"NaN count per col: {weekly_df[cols].isna().sum().to_dict()}")
        
    return res_df


def save_to_delta(df, target_path, table_name=""):
    logger.info(f"💾 Sauvegarde Delta de {table_name} vers {target_path}")
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    logger.success(f"✅ Enregistré dans {target_path}.")

def save_to_bigquery(df, bq_table, table_name=""):
    logger.info(f"💾 Sauvegarde BigQuery de {table_name} vers {bq_table}")
    df.write.format("bigquery") \
        .option("table", bq_table) \
        .option("temporaryGcsBucket", BQ_TEMP_BUCKET) \
        .option("parentProject", GCP_PROJECT_ID) \
        .option("credentialsFile", GCP_KEY_PATH) \
        .mode("overwrite") \
        .save()
    logger.success(f"✅ Enregistré dans BigQuery: {bq_table}")

def main():
    setup_logging()
    logger.info("🚀 Starting Job: Gold Feature Engineering (Dynamic Champion Config)")
    
    # 0. Récupération de la configuration Champion
    config = get_champion_config()
    logger.info(f"📋 Configuration Champion utilisée: {config}")

    spark = None
    try:
        spark = create_spark_session(app_name="Features_Gold_Dynamic")
        
        # --- 1. PROCESS ETFS ---
        logger.info(f"📥 Loading ETF Weekly from {Paths.DATA_RAW_ETF_WEEKLY_SILVER}")
        try:
            df_etf_silver = spark.read.format("delta").load(Paths.DATA_RAW_ETF_WEEKLY_SILVER)
            if not df_etf_silver.isEmpty():
                df_etf_gold = df_etf_silver.groupby("Ticker").applyInPandas(lambda df: process_etf_features(df, config), schema=etf_schema)
                df_etf_gold = df_etf_gold.withColumn("Date", col("Date").cast("date"))
                save_to_bigquery(df_etf_gold, Paths.BQ_ETF_GOLD, "ETFs Gold (Weekly)")
                
                # VALIDATION DE QUALITÉ
                try:
                    logger.info("🧪 Validation de la qualité des données ETF Gold...")
                    validate_df(df_etf_gold, "suite_etf_gold")
                except Exception as ve:
                    logger.error(f"⚠️ Échec de la validation de qualité ETF : {ve}")
                
                save_to_delta(df_etf_gold, Paths.DATA_RAW_ETF_WEEKLY_GOLD, "ETFs Gold (Delta Cache)")
            else:
                logger.warning("⚠️ Table ETF Weekly Silver est vide, passe...")
        except Exception as e:
            logger.warning(f"⚠️ Impossible de traiter les ETFs : {e}")
        
        # --- 2. PROCESS SP500 INDEX (Silver Weekly) ---
        logger.info(f"📥 Loading S&P 500 Weekly from {Paths.DATA_RAW_SP500_WEEKLY_SILVER}")
        try:
            df_sp500_silver = spark.read.format("delta").load(Paths.DATA_RAW_SP500_WEEKLY_SILVER)
            if not df_sp500_silver.isEmpty():
                # On utilise les params SP500 de la config
                sp500_config = config.copy()
                sp500_config['etf_sma_fast'] = config['sp500_sma_fast']
                sp500_config['etf_sma_slow'] = config['sp500_sma_slow']
                sp500_config['etf_mom_period'] = 1 # Non utilisé pour l'indice mais requis par la fonction
                
                # Conversion en Pandas pour le traitement des indicateurs
                df_sp500_pd = df_sp500_silver.toPandas()
                
                if 'Ticker' not in df_sp500_pd.columns:
                    df_sp500_pd['Ticker'] = '^GSPC'
                
                # Sécurité : On s'assure de ne pas avoir de doublons si 'Close' existe déjà
                # (On ne fait rien si Close est déjà là, on évite juste de s'emmêler avec des vieilles colonnes)
                pass
                
                logger.info(f"🧐 Pré-traitement SP500 : {len(df_sp500_pd)} lignes, Colonnes: {df_sp500_pd.columns.tolist()}")
                
                df_sp500_res = process_etf_features(df_sp500_pd, sp500_config)
                
                if not df_sp500_res.empty:
                    df_sp500_gold = spark.createDataFrame(df_sp500_res, schema=etf_schema)
                    df_sp500_gold = df_sp500_gold.withColumn("Date", col("Date").cast("date"))
                    save_to_bigquery(df_sp500_gold, Paths.BQ_SP500_GOLD, "SP500 Index Gold (Weekly)")
                    
                    # VALIDATION DE QUALITÉ
                    try:
                        logger.info("🧪 Validation de la qualité des données SP500 Gold...")
                        validate_df(df_sp500_gold, "suite_sp500_gold")
                    except Exception as ve:
                        logger.error(f"⚠️ Échec de la validation de qualité SP500 : {ve}")
                else:
                    logger.warning("⚠️ Résultats vides après traitement Gold pour le S&P 500")
            else:
                logger.warning("⚠️ Table SP500 Weekly Silver est vide")
        except Exception as e:
            logger.warning(f"⚠️ Impossible de traiter le SP500 via la couche Silver : {e}")
        
        # --- 3. PROCESS STOCKS ---
        logger.info(f"📥 Loading Stocks Daily from {Paths.DATA_RAW_2B_SILVER}")
        try:
            df_stock_silver = spark.read.format("delta").load(Paths.DATA_RAW_2B_SILVER)
            if not df_stock_silver.isEmpty():
                p_col_stock = 'Close'
                logger.info(f"🔍 Colonne de prix pour Stocks Silver : {p_col_stock}")
                
                df_stock_clean = df_stock_silver.dropna(subset=['High', 'Low', p_col_stock])
                
                # Pre-filter: On a besoin de SMA_slow semaines, donc SMA_slow * 5 jours (Daily)
                # min_rows: 64 weeks * 5 days + buffer
                min_rows = max(config['stock_sma_slow'] * 5 + 20, 350)
                from pyspark.sql.functions import count
                ticker_counts = df_stock_clean.groupBy("Ticker").agg(count("*").alias("cnt"))
                valid_tickers = ticker_counts.filter(col("cnt") > min_rows).select("Ticker")
                logger.info(f"📊 Filtrage Stocks : {valid_tickers.count()} tickers valides sur {ticker_counts.count()} (min_rows={min_rows})")
                df_stock_clean = df_stock_clean.join(valid_tickers, on="Ticker", how="inner")
                
                # Tri en amont par Ticker et Date
                df_stock_clean = df_stock_clean.orderBy("Ticker", "Date")
                
                df_stock_gold = df_stock_clean.groupBy("Ticker").applyInPandas(lambda df: process_stock_features(df, config), schema=stock_schema)
                df_stock_gold = df_stock_gold.withColumn("Date", col("Date").cast("date"))
                
                save_to_bigquery(df_stock_gold, Paths.BQ_STOCKS_GOLD, "Stocks Gold (Weekly Aggregated)")
                save_to_delta(df_stock_gold, Paths.DATA_RAW_2B_WEEKLY_GOLD, "Stocks Gold (Delta Cache)")
                
                # VALIDATION DE QUALITÉ
                try:
                    logger.info("🧪 Validation de la qualité des données Stocks Gold...")
                    validate_df(df_stock_gold, "suite_stocks_gold")
                except Exception as ve:
                    logger.error(f"⚠️ Échec de la validation de qualité : {ve}")
            else:
                logger.warning("⚠️ Table Stocks Daily Silver est vide, passe...")
        except Exception as e:
            logger.warning(f"⚠️ Impossible de traiter les actions 2B : {e}")

    except Exception as e:
        logger.critical(f"❌ Critical Error in job execution: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
