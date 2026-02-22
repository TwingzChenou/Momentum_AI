import os
import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, ArrayType
from loguru import logger
import requests
from io import StringIO

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths


def get_sp500_history_from_wikipedia():
    """
    Reconstructs the historical S&P 500 composition by 'undoing' changes
    listed on Wikipedia, starting from the current list.
    """
    logger.info("🌐 Scraping Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        # Wikipedia tables: 0 = Current Constituents, 1 = Selected Changes
        dfs = pd.read_html(
            url,
            storage_options={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                           "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"}
        )
        current_df = dfs[0]
        changes_df = dfs[1]
    except Exception as e:
        logger.error(f"❌ Error scraping Wikipedia: {e}")
        return None

    # 1. Get Current List (The "Anchor")
    current_tickers = set(current_df['Symbol'].unique())
    logger.info(f"✅ Loaded {len(current_tickers)} current tickers.")

    # 2. Prepare the Changes Table
    changes_df = changes_df.copy()

    # Flatten MultiIndex columns if present (common in Wikipedia tables)
    if isinstance(changes_df.columns, pd.MultiIndex):
        changes_df.columns = ['Date', 'Added_Ticker', 'Added_Security', 'Removed_Ticker', 'Removed_Security', 'Reason']
    else:
        # Fallback if structure changes, though usually it's consistent
        pass
    
    # Ensure Date parsing
    changes_df['Date'] = pd.to_datetime(changes_df['Date'], errors='coerce')
    changes_df = changes_df.dropna(subset=['Date'])
    
    # Sort by Date DESCENDING (Newest first) - Crucial for "Undoing"
    changes_df = changes_df.sort_values('Date', ascending=False)

    # 3. The "Time Machine" Loop
    history = []
    
    # Record current state (Effective from today)
    history.append({
        'date': datetime.today(),
        'tickers': list(current_tickers)
    })

    for _, row in changes_df.iterrows():
        change_date = row['Date']
        added = row['Added_Ticker']
        removed = row['Removed_Ticker']
        
        # LOGIC: To go BACK in time:
        # If ADDED on this date -> it wasn't there before -> REMOVE it.
        if pd.notna(added):
            current_tickers.discard(added)
            
        # If REMOVED on this date -> it WAS there before -> ADD it back.
        if pd.notna(removed):
            current_tickers.add(removed)
            
        # Record state just BEFORE this change
        history.append({
            'date': change_date,
            'tickers': list(current_tickers)
        })

    # 4. Final Formatting
    history_df = pd.DataFrame(history)
    history_df = history_df.sort_values('date', ascending=True).set_index('date')
    
    return history_df

def save_history_to_lake(spark, pandas_df):
    """Saves the historical composition DataFrame to Delta Lake."""
    logger.info(f"💾 Saving History to {Paths.SP500_LIST_TICKERS}...")
    
    # Define Schema Explicitly
    schema = StructType([
        StructField("date", DateType(), True),
        StructField("tickers", ArrayType(StringType()), True)
    ])
    
    # Convert Pandas to Spark (reset index to make 'date' a column)
    pandas_df_reset = pandas_df.reset_index()
    
    try:
        sdf = spark.createDataFrame(pandas_df_reset, schema=schema)
        
        # Write to Delta (Overwrite mode for full history refresh)
        sdf.write.format("delta") \
            .mode("overwrite") \
            .save(Paths.SP500_LIST_TICKERS)
            
        logger.success("✅ Success! History saved.")
        
    except Exception as e:
        logger.error(f"❌ Error saving to Lake: {e}")
        

def main():
    # Setup logging
    setup_logging()

    # 1. Fetch Data
    df_history = get_sp500_history_from_wikipedia()
    
    if df_history is None:
        logger.error("❌ Failed to fetch history. Exiting.")
        return

    logger.success("\n📜 History Reconstruction Complete!")

    # From 2000 to today
    df_history = df_history.loc["2000-01-01":]
    print(df_history.head())

    # Quick sanity check
    try:
        # Check a known historical date roughly
        count_2000 = len(df_history.iloc[0]['tickers'])
        logger.info(f"ℹ️ Companies in early script history (approx 2000 or earliest): {count_2000}")
    except Exception:
        logger.warning("ℹ️ Could not perform historical count check.")


    # 2. Initialize Spark
    spark = create_spark_session(app_name="SP500_History_Ingestion")
    
    try:
        # 3. Save to Lake
        save_history_to_lake(spark, df_history)
    finally:
        spark.stop()
        logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()