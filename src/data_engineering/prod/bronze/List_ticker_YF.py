import os
import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
import requests
from tradingview_screener import Query, Column

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

# Load environment variables
load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")

def fetch_tickers_2b():
    """
    Fetch all tickers with market cap > 2B from TradingView Screener.
    """
    logger.info("📡 Connecting to TradingView to fetch tickers > $2B...")
    
    query = (Query()
        .set_markets('america', 'france', 'germany', 'italy', 'uk', 'canada', 'japan', 'hongkong')
        .select('name', 'description', 'market_cap_basic', 'exchange', 'type', 'subtype')
        .where(
            Column('type') == 'stock',
            Column('is_primary') == True,
            Column('market_cap_basic') >= 2e9  
        )
        .limit(30000))

    try:
        n_total, df = query.get_scanner_data()
        
        if df.empty:
            logger.warning("❌ No data returned from TradingView.")
            return None
            
        logger.info(f"✅ Successfully fetched {len(df)} tickers from TradingView.")

        # --- Maintien de la compatibilité du schéma (format FMP) ---
        df = df.rename(columns={
            'name': 'symbol',
            'description': 'companyName',
            'market_cap_basic': 'marketCap',
            'exchange': 'exchangeShortName'
        })

        # Ajout des colonnes attendues par Silver/Gold (Valeurs nulles)
        cols_to_add = [
            'sector', 'industry', 'beta', 'price', 'lastAnnualDividend', 
            'volume', 'exchange', 'country', 'isEtf', 'isFund', 'isActivelyTrading'
        ]
        for col in cols_to_add:
            if col not in df.columns:
                if col == 'exchange':
                    df[col] = df['exchangeShortName']
                elif col == 'isActivelyTrading':
                    df[col] = True
                elif col in ['isEtf', 'isFund']:
                    df[col] = False
                else:
                    df[col] = None
            
        return df

    except Exception as e:
        logger.error(f"❌ Error fetching tickers from TradingView: {e}")
        return None

def save_to_lake(spark, pandas_df):
    """
    Saves the fetched tickers DataFrame to Delta Lake.
    """
    logger.info(f"💾 Saving to {Paths.LIST_TICKER_2B} with {pandas_df.shape[0]} rows...")
    
    try:
        sdf = spark.createDataFrame(pandas_df)
        
        # Write to Delta (Overwrite mode)
        sdf.write.format("delta") \
            .mode("overwrite") \
            .save(Paths.LIST_TICKER_2B)
            
        logger.info(f"✅ Success! Data saved to {Paths.LIST_TICKER_2B}.")
        
    except Exception as e:
        logger.error(f"❌ Error saving to Lake: {e}")

def main():
    # Setup logging
    setup_logging()

    logger.info("🚀 Starting Job: Fetch Tickers > 2B Ingestion")

    spark = None
    df_tickers = None

    try:
        # 1. Fetch the data
        df_tickers = fetch_tickers_2b()

        if df_tickers is not None and not df_tickers.empty:
            # 2. Create Spark Session
            logger.info("🚀 Creating Spark Session...")
            spark = create_spark_session(app_name="List_Tickers_2B_Ingestion")

            # 3. Save to Lake
            save_to_lake(spark, df_tickers)
        else:
            logger.warning("⚠️ Skipping write to Lake because data fetch failed or is empty.")

    except Exception as e:
        logger.error(f"❌ Critical Error in job execution: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
