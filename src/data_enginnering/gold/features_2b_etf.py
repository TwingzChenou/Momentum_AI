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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

# --- ETF PIPELINE ---
etf_schema = StructType([
    StructField("Date", StringType(), True),
    StructField("Ticker", StringType(), True),
    StructField("Close", DoubleType(), True),
    StructField("SMA_12", DoubleType(), True),
    StructField("SMA_26", DoubleType(), True),
    StructField("SMA_50", DoubleType(), True),
    StructField("Momentum_3M", DoubleType(), True)
])

def process_etf_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df_empty = pd.DataFrame(columns=[f.name for f in etf_schema.fields])
        df_empty['Date'] = df_empty['Date'].astype(str)
        for c in ['Close', 'SMA_12', 'SMA_26', 'SMA_50', 'Momentum_3M']:
            df_empty[c] = pd.to_numeric(df_empty[c], errors='coerce').astype('float64')
        return df_empty
        
    df = df.sort_values("Date")
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    price_col = 'AdjClose' if 'AdjClose' in df.columns else 'Close'

    df['SMA_12'] = ta.trend.sma_indicator(df[price_col], window=12)
    df['SMA_26'] = ta.trend.sma_indicator(df[price_col], window=26)
    df['SMA_50'] = ta.trend.sma_indicator(df[price_col], window=50)
    df['Momentum_3M'] = df[price_col].pct_change(periods=13)
    
    df['Close'] = df[price_col]
    
    return df[[f.name for f in etf_schema.fields]]

# --- STOCKS PIPELINE ---
stock_schema = StructType([
    StructField("Date", StringType(), True),
    StructField("Ticker", StringType(), True),
    StructField("AdjClose", DoubleType(), True),
    StructField("ADX_20", DoubleType(), True),
    StructField("ATR_14", DoubleType(), True),
    StructField("SMA_12", DoubleType(), True),
    StructField("SMA_26", DoubleType(), True),
    StructField("SMA_50", DoubleType(), True),
    StructField("ATR_pct", DoubleType(), True),
    StructField("Momentum_3M", DoubleType(), True),
    StructField("Momentum_1W", DoubleType(), True)
])

def process_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applied to grouped Pandas dataframe per Ticker.
    df in input is daily data. Output must be weekly aggregated.
    """
    if df.empty:
        df_empty = pd.DataFrame(columns=[f.name for f in stock_schema.fields])
        df_empty['Date'] = df_empty['Date'].astype(str)
        for c in ['AdjClose', 'ADX_20', 'ATR_14', 'SMA_12', 'SMA_26', 'SMA_50', 'ATR_pct', 'Momentum_3M', 'Momentum_1W']:
            df_empty[c] = pd.to_numeric(df_empty[c], errors='coerce').astype('float64')
        return df_empty
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date").set_index("Date")
    
    # 1. Daily ADX and ATR
    if len(df) > 20:
        df['ADX_20'] = ta.trend.adx(df['High'], df['Low'], df['AdjClose'], window=20, fillna=True)
        df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['AdjClose'], window=14, fillna=True)
    else:
        df['ADX_20'] = np.nan
        df['ATR_14'] = np.nan
        
    # 2. Resample to Weekly W-FRI
    weekly_df = df.resample('W-FRI').agg({
        'AdjClose': 'last',
        'ADX_20': 'last',
        'ATR_14': 'last'
    }).dropna(subset=['AdjClose'])
    
    # 3. Calculate Weekly indicators 
    if len(weekly_df) >= 1:
        weekly_df['SMA_12'] = ta.trend.sma_indicator(weekly_df['AdjClose'], window=12)
        weekly_df['SMA_26'] = ta.trend.sma_indicator(weekly_df['AdjClose'], window=26)
        weekly_df['SMA_50'] = ta.trend.sma_indicator(weekly_df['AdjClose'], window=50)
        weekly_df['ATR_pct'] = weekly_df['ATR_14'] / weekly_df['AdjClose']
        weekly_df['Momentum_3M'] = weekly_df['AdjClose'].pct_change(periods=13)
        weekly_df['Momentum_1W'] = weekly_df['AdjClose'].pct_change(periods=1)
    else:
        for col in ['SMA_12', 'SMA_26', 'SMA_50', 'ATR_pct', 'Momentum_3M', 'Momentum_1W']:
            weekly_df[col] = np.nan
            
    weekly_df = weekly_df.reset_index()
    weekly_df['Date'] = weekly_df['Date'].dt.strftime('%Y-%m-%d')
    weekly_df['Ticker'] = df['Ticker'].iloc[0] if not df.empty else ""
    
    # Cast back to match Arrow schema explicitly
    for col_name in ['AdjClose', 'ADX_20', 'ATR_14', 'SMA_12', 'SMA_26', 'SMA_50', 'ATR_pct', 'Momentum_3M', 'Momentum_1W']:
        weekly_df[col_name] = weekly_df[col_name].astype('float64')
        
    result_df = weekly_df[[f.name for f in stock_schema.fields]].copy()
    
    return result_df


def save_to_delta(df, target_path, table_name=""):
    logger.info(f"💾 Sauvegarde Delta de {table_name} vers {target_path}")
    df.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(target_path)
    logger.success(f"✅ Enregistré dans {target_path}. (Prêt pour import BigQuery)")

def main():
    setup_logging()
    logger.info("🚀 Starting Job: Gold Feature Engineering (ETFs + 2B Stocks)")

    spark = None
    try:
        spark = create_spark_session(app_name="Features_Gold")
        
        # --- 1. PROCESS ETFS ---
        logger.info(f"📥 Loading ETF Weekly from {Paths.DATA_RAW_ETF_WEEKLY_SILVER}")
        try:
            df_etf_silver = spark.read.format("delta").load(Paths.DATA_RAW_ETF_WEEKLY_SILVER)
            if not df_etf_silver.isEmpty():
                df_etf_gold = df_etf_silver.groupby("Ticker").applyInPandas(process_etf_features, schema=etf_schema)
                df_etf_gold = df_etf_gold.withColumn("Date", col("Date").cast("date"))
                save_to_delta(df_etf_gold, Paths.DATA_RAW_ETF_WEEKLY_GOLD, "ETFs Gold (Weekly)")
            else:
                logger.warning("⚠️ Table ETF Weekly Silver est vide, passe...")
        except Exception as e:
            logger.warning(f"⚠️ Impossible de traiter les ETFs : {e}")
        
        # --- 2. PROCESS STOCKS ---
        logger.info(f"📥 Loading Stocks Daily from {Paths.DATA_RAW_2B_SILVER}")
        try:
            df_stock_silver = spark.read.format("delta").load(Paths.DATA_RAW_2B_SILVER)
            if not df_stock_silver.isEmpty():
                df_stock_clean = df_stock_silver.dropna(subset=['High', 'Low', 'AdjClose'])
                df_stock_gold = df_stock_clean.groupby("Ticker").applyInPandas(process_stock_features, schema=stock_schema)
                df_stock_gold = df_stock_gold.withColumn("Date", col("Date").cast("date"))
                save_to_delta(df_stock_gold, Paths.DATA_RAW_2B_WEEKLY_GOLD, "Stocks Gold (Weekly Aggregated)")
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
