import os
import sys
import pandas as pd
from loguru import logger
from tradingview_screener import Query, Column

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from pyspark.sql import functions as F
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from src.common.quality_manager import QualityManager
from config.config_spark import Paths
def fetch_sp500_from_tradingview():
    """
    Fetches S&P 500 constituents by first getting tickers and addition dates from Wikipedia 
    and then querying TradingView for the specified metadata.
    """
    logger.info("🌐 Fetching S&P 500 tickers and addition dates from Wikipedia...")
    try:
        import requests
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        tables = pd.read_html(response.text)
        df_wiki = tables[0]
        
        # Keep Symbol and Date added columns, renaming to match our schema
        df_wiki_cleaned = df_wiki[['Symbol', 'Date added']].copy()
        df_wiki_cleaned['Symbol'] = df_wiki_cleaned['Symbol'].str.replace('.', '-', regex=False)
        df_wiki_cleaned = df_wiki_cleaned.rename(columns={'Symbol': 'symbol', 'Date added': 'dateFirstAdded'})
        
        tickers = df_wiki_cleaned['symbol'].tolist()
        logger.info(f"✅ Found {len(tickers)} tickers on Wikipedia.")
    except Exception as e:
        logger.error(f"❌ Failed to fetch from Wikipedia: {e}")
        return pd.DataFrame()

    logger.info("📡 Querying TradingView for metadata in chunks...")
    all_metadata = []
    chunk_size = 100
    
    try:
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            logger.info(f"⏱️ Fetching chunk {i//chunk_size + 1}/{(len(tickers)-1)//chunk_size + 1}...")
            
            q = (Query()
                 .set_markets('america')
                 .select('name', 'description', 'market_cap_basic', 'exchange', 'type', 'subtype', 'sector', 'industry')
                 .where(Column('name').isin(chunk))
                 .limit(chunk_size))
            
            n_total, df_chunk = q.get_scanner_data()
            if not df_chunk.empty:
                all_metadata.append(df_chunk)
        
        if not all_metadata:
            logger.warning("⚠️ No metadata returned from TradingView.")
            return pd.DataFrame()

        df_tv = pd.concat(all_metadata, ignore_index=True)

        # Rename columns to match FMP format for compatibility with existing pipeline
        df_tv = df_tv.rename(columns={
            'name': 'symbol',
            'description': 'companyName',
            'market_cap_basic': 'marketCap',
            'exchange': 'exchangeShortName'
        })
        
        # Merge Wikipedia 'dateFirstAdded' with TradingView metadata
        df_merged = pd.merge(df_tv, df_wiki_cleaned, on='symbol', how='left')
        
        logger.info(f"✅ Successfully enriched {len(df_merged)} tickers with TradingView metadata and Wikipedia dates.")
        return df_merged
        
    except Exception as e:
        logger.error(f"❌ Failed to query TradingView: {e}")
        return pd.DataFrame()

def main():
    setup_logging()
    
    spark = create_spark_session("SP500_List_Update_TradingView")
    
    try:
        # 1. Fetch data using our new hybrid logic
        df_latest = fetch_sp500_from_tradingview()
        
        if df_latest.empty:
            logger.warning("⚠️ No data fetched. Aborting ingestion.")
            return

        # 2. Data Cleaning for Spark
        for col in df_latest.columns:
            if df_latest[col].dtype == 'object':
                df_latest[col] = df_latest[col].fillna("").astype(str)
            elif df_latest[col].dtype == 'float64' or df_latest[col].dtype == 'int64':
                df_latest[col] = df_latest[col].fillna(0.0)

        # 3. Convert to Spark
        sdf_latest = spark.createDataFrame(df_latest)
        
        # 3a. Ajouter la dimension temporelle
        sdf_latest = sdf_latest.withColumn("ingestion_date", F.current_date())

        # 3b. Validation GX
        QualityManager.validate_ticker_list(sdf_latest, label="SP500 Tickers", min_rows=450)

        logger.info(f"🚀 DEBUG: Writing {sdf_latest.count()} rows to {Paths.SP500_LATEST_TICKERS}")

        # 4. Save to Lake (Overwrite with Schema)
        logger.info(f"💾 Saving latest 500 constituents to {Paths.SP500_LATEST_TICKERS}")
        sdf_latest.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(Paths.SP500_LATEST_TICKERS)

        logger.info(f"💾 Saving master list to {Paths.SP500_LIST_TICKERS}")
        sdf_latest.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(Paths.SP500_LIST_TICKERS)

        logger.success("✅ S&P 500 List updated successfully via Wikipedia + TradingView.")

    except Exception as e:
        logger.critical(f"❌ Critical error in S&P 500 Ingestion: {e}")
    finally:
        if 'spark' in locals() and spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
