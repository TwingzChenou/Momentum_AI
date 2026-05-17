import os
import sys
from dotenv import load_dotenv
from pyspark.sql.functions import col, lit, to_date, row_number, lead, when
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from loguru import logger

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def process_sp500_consolidated_history(spark):
    """
    Reads LATEST and HISTORY SP500 ticker data, consolidates ADD/REMOVE dates,
    and constructs a unified history DataFrame with (Ticker, Date_start, Date_end).
    """
    logger.info("📡 Loading SP500_LATEST_TICKERS and SP500_LIST_TICKERS from Delta Lake...")
    
    try:
        df_latest = spark.read.format("delta").load(Paths.SP500_LATEST_TICKERS)
        df_history = spark.read.format("delta").load(Paths.SP500_LIST_TICKERS)
    except Exception as e:
        logger.error(f"❌ Error loading input Delta tables: {e}")
        raise e

    logger.info("⚙️ Transforming data to build consolidated history...")

    # 1. Extract events from FMP History (Base historical data)
    # Note: We now use 'ingestion_date' instead of 'date'
    adds_hist = df_history.filter((col("symbol").isNotNull()) & (col("symbol") != "")) \
        .select(col("symbol").alias("Ticker"), to_date(col("ingestion_date")).alias("event_date"), lit("ADD").alias("event_type"))
        
    # If 'removedTicker' exists in history, use it. Otherwise, our drift detection will handle it.
    if "removedTicker" in df_history.columns:
        removes_hist = df_history.filter((col("removedTicker").isNotNull()) & (col("removedTicker") != "")) \
            .select(col("removedTicker").alias("Ticker"), to_date(col("ingestion_date")).alias("event_date"), lit("REMOVE").alias("event_type"))
    else:
        removes_hist = spark.createDataFrame([], adds_hist.schema)

    # 2. Extract events from Wikipedia Latest (Ground truth for TODAY)
    # We use the ingestion_date we just added
    today = F.current_date()
    latest_tickers = df_latest.select(col("symbol").alias("Ticker")).distinct()

    # 3. Combine FMP events
    events = adds_hist.unionByName(removes_hist)

    # 4. Logic to detect current membership drift
    # If a stock is in Latest but has no active record in events -> It was added today (or recently)
    # If a stock has an active record (last event was ADD) but is NOT in Latest -> It was removed today
    
    # Simple consolidation first
    windowSpec = Window.partitionBy("Ticker").orderBy("event_date", "event_type")
    consolidated = events.withColumn("prev_event_type", F.lag("event_type").over(windowSpec)) \
                         .filter((col("prev_event_type").isNull()) | (col("event_type") != col("prev_event_type")))
    
    # Find last state for each ticker
    last_state = consolidated.withColumn("rn", F.row_number().over(Window.partitionBy("Ticker").orderBy(col("event_date").desc()))) \
                             .filter(col("rn") == 1) \
                             .select("Ticker", col("event_type").alias("last_type"), col("event_date").alias("last_date"))

    # Join with Latest to detect new ADDS/REMOVES
    drift_adds = latest_tickers.join(last_state, "Ticker", "left") \
        .filter((col("last_type").isNull()) | (col("last_type") == "REMOVE")) \
        .select("Ticker", today.alias("event_date"), lit("ADD").alias("event_type"))

    drift_removes = last_state.join(latest_tickers, "Ticker", "left_anti") \
        .filter(col("last_type") == "ADD") \
        .select("Ticker", today.alias("event_date"), lit("REMOVE").alias("event_type"))

    n_adds = drift_adds.count()
    if n_adds > 0:
        logger.info(f"🆕 Found {n_adds} new tickers to add to S&P 500 history.")
        
    n_removes = drift_removes.count()
    if n_removes > 0:
        logger.info(f"❌ Found {n_removes} tickers to remove from S&P 500 history.")

    # 5. Final event union
    final_events = consolidated.select("Ticker", "event_date", "event_type") \
                               .unionByName(drift_adds) \
                               .unionByName(drift_removes) \
                               .dropDuplicates(["Ticker", "event_date", "event_type"])

    # 6. Re-consolidate to build (Date_start, Date_end)
    final_window = Window.partitionBy("Ticker").orderBy("event_date")
    
    df_history_final = final_events.withColumn("next_date", lead("event_date").over(final_window)) \
                                   .withColumn("next_type", lead("event_type").over(final_window)) \
                                   .filter(col("event_type") == "ADD") \
                                   .select(
                                       col("Ticker"),
                                       col("event_date").alias("Date_start"),
                                       when(col("next_type") == "REMOVE", col("next_date")).otherwise(None).alias("Date_end")
                                   )

    # Clean up and exclude specific tickers as per your requirement
    tickers_to_exclude = ['EF', 'JBL', 'HP', 'TMUS', 'FMCC', 'FNMA', 'CTX', 'AET', 'MXIM', 'PARA']
    df_final = df_history_final.filter(~col("Ticker").isin(tickers_to_exclude))

    return df_final



def save_history_to_lake(df, output_path, spark):
    """
    Saves the consolidated composition DataFrame to Delta Lake using MERGE to preserve history.
    """
    from delta.tables import DeltaTable
    logger.info(f"💾 Merging Consolidated History into {output_path}...")
    
    try:
        if DeltaTable.isDeltaTable(spark, output_path):
            dt = DeltaTable.forPath(spark, output_path)
            dt.alias("target").merge(
                df.alias("source"),
                "target.Ticker = source.Ticker AND target.Date_start = source.Date_start"
            ).whenMatchedUpdateAll() \
             .whenNotMatchedInsertAll() \
             .execute()
            logger.success("✅ Success! History merged (Upsert).")
        else:
            df.write.format("delta").mode("overwrite").save(output_path)
            logger.success("✅ Success! History table created.")
            
    except Exception as e:
        logger.error(f"❌ Error saving to Lake: {e}")
        raise e

def main():
    # Setup logging
    setup_logging()
    load_dotenv()

    logger.info("🚀 Starting Job: SP500 Consolidated History Generation")

    spark = None
    try:
        spark = create_spark_session(app_name="SP500_Consolidated_History")

        # 1. Load Existing History as Base
        logger.info("📡 Loading existing base history...")
        df_base = spark.read.format("delta").load(Paths.SP500_CONSOLIDATED_HISTORY)
        
        # 2. Detect drift from LATEST
        df_latest = spark.read.format("delta").load(Paths.SP500_LATEST_TICKERS)
        latest_tickers = df_latest.select(F.col("symbol").alias("Ticker")).distinct()
        
        # Current active tickers in base (Date_end is NULL)
        active_in_base = df_base.filter(F.col("Date_end").isNull())
        
        # New ADDS: in Latest but NOT in active_in_base
        new_adds = latest_tickers.join(active_in_base, "Ticker", "left_anti") \
                                 .withColumn("Date_start", F.current_date()) \
                                 .withColumn("Date_end", F.lit(None).cast("date"))
        
        # New REMOVES: in active_in_base but NOT in Latest
        new_removes = active_in_base.join(latest_tickers, "Ticker", "left_anti") \
                                    .withColumn("Date_end", F.current_date())
        
        # 3. Combine
        # For removes, we need to update the existing row, not add a new one. 
        # So we union new_adds and the UPDATED rows for new_removes.
        df_updates = new_adds.unionByName(new_removes)
        
        if df_updates.count() > 0:
            logger.info(f"🔄 Detected {df_updates.count()} changes to apply.")
            save_history_to_lake(df_updates, Paths.SP500_CONSOLIDATED_HISTORY, spark)
        else:
            logger.success("✅ History is already up to date with Latest tickers.")

    except Exception as e:
        logger.critical(f"❌ Critical Error: {e}")
        sys.exit(1)
    finally:
        if spark: spark.stop()

if __name__ == "__main__":
    main()
