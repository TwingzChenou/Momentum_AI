import os
import sys
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from loguru import logger

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from src.common.quality_manager import QualityManager
from config.config_spark import Paths

def process_2b_consolidated_history(spark):
    """
    Reads LATEST 2B tickers and compares with existing history to detect 
    membership changes (stocks entering or leaving the >$2B universe).
    """
    logger.info("📡 Loading LIST_TICKER_2B and current history...")
    
    # 1. Load Current Snapshot
    df_latest = spark.read.format("delta").load(Paths.LIST_TICKER_2B)
    latest_tickers = df_latest.select(F.col("name").alias("Ticker")).distinct()
    
    today = F.current_date()

    # 2. Load Existing Consolidated History (if exists)
    try:
        df_history = spark.read.format("delta").load(Paths.TICKERS_2B_CONSOLIDATED_HISTORY)
        logger.info(f"📜 Loaded existing history with {df_history.count()} records.")
    except Exception:
        logger.warning("⚠️ No existing history found. Starting fresh.")
        # Create empty history with correct schema
        return latest_tickers.select(
            F.col("Ticker"),
            today.alias("Date_start"),
            F.lit(None).cast("date").alias("Date_end")
        )

    # 3. Detect Drift
    # Find currently active tickers in history (Date_end is NULL)
    active_history = df_history.filter(F.col("Date_end").isNull()).select("Ticker")

    # New ADDS: In Latest but not in active history
    new_adds = latest_tickers.join(active_history, "Ticker", "left_anti") \
        .select("Ticker", today.alias("Date_start"), F.lit(None).cast("date").alias("Date_end"))
    n_adds = new_adds.count()
    if n_adds > 0:
        logger.info(f"🆕 Found {n_adds} new tickers entering the 2B universe.")

    # New REMOVES: In active history but not in latest
    new_removes = active_history.join(latest_tickers, "Ticker", "left_anti") \
        .select("Ticker")
    n_removes = new_removes.count()
    if n_removes > 0:
        logger.info(f"❌ Found {n_removes} tickers leaving the 2B universe.")

    # Update history: 
    # - Keep old closed periods as they are
    # - Close periods for stocks in new_removes
    # - Add new periods for stocks in new_adds
    
    logger.info("🔄 Reconstructing history with new events...")
    old_closed = df_history.filter(F.col("Date_end").isNotNull())
    
    current_active_staying = df_history.filter(F.col("Date_end").isNull()) \
        .join(new_removes, "Ticker", "left_anti")
    logger.info(f"🏠 {current_active_staying.count()} tickers remained active.")
        
    current_active_closing = df_history.filter(F.col("Date_end").isNull()) \
        .join(new_removes, "Ticker", "inner") \
        .withColumn("Date_end", today)

    df_final = old_closed.unionByName(current_active_staying) \
                         .unionByName(current_active_closing) \
                         .unionByName(new_adds)

    return df_final

def main():
    setup_logging()
    logger.info("🚀 Starting Job: 2B Tickers Consolidated History")

    spark = create_spark_session(app_name="2B_Tickers_Consolidated_History")

    try:
        df_consolidated = process_2b_consolidated_history(spark)
        
        # 3b. Validation GX
        # On vérifie que la table finale est saine (minimum 2000 tickers actifs)
        QualityManager.validate_ticker_list(
            df_consolidated.filter(F.col("Date_end").isNull()).withColumnRenamed("Ticker", "symbol").withColumn("marketCap", F.lit(2e9)), 
            label="2B History (Active)", 
            min_rows=2000
        )

        logger.info(f"💾 Saving 2B History to {Paths.TICKERS_2B_CONSOLIDATED_HISTORY}")
        df_consolidated.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(Paths.TICKERS_2B_CONSOLIDATED_HISTORY)

        logger.success(f"✅ 2B History updated. Total records: {df_consolidated.count()}")
        df_consolidated.orderBy(F.col("Date_start").desc()).show(10)

    except Exception as e:
        logger.critical(f"❌ Error in 2B history consolidation: {e}")
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()
