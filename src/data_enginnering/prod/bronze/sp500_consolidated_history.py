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

    # Extract Adds from History
    adds_hist = df_history.filter((col("symbol").isNotNull()) & (col("symbol") != "")) \
        .select(col("symbol").alias("Ticker"), to_date(col("date")).alias("Date_start"))
        
    # Extract Removes from History
    removes_hist = df_history.filter((col("removedTicker").isNotNull()) & (col("removedTicker") != "")) \
        .select(col("removedTicker").alias("Ticker"), to_date(col("date")).alias("Date_end"))

    # Also Extract "Implicit" Adds from Latest (for stocks added before history data starts)
    adds_latest = df_latest.select(
        col("symbol").alias("Ticker"), 
        to_date(col("dateFirstAdded")).alias("Date_start")
    )
    
    # Clean up and combine adds Date_start
    all_adds = adds_hist.unionByName(adds_latest).dropDuplicates(["Ticker", "Date_start"]).filter(col("Date_start").isNotNull())
    all_removes = removes_hist.dropDuplicates(["Ticker", "Date_end"]).filter(col("Date_end").isNotNull())

    # Map directly via chronological event logging to resolve any ordering
    events_add = all_adds.select("Ticker", col("Date_start").alias("event_date"), lit("ADD").alias("event_type"))
    events_remove = all_removes.select("Ticker", col("Date_end").alias("event_date"), lit("REMOVE").alias("event_type"))
    
    events = events_add.unionByName(events_remove).dropDuplicates(["Ticker", "event_date", "event_type"])
    
    # --- FIX: Deduplicate consecutive events of the same type ---
    # If we have ADD followed by another ADD, we only keep the first one (earliest).
    # If we have REMOVE followed by another REMOVE, we only keep the first one.
    windowSpecDedup = Window.partitionBy("Ticker").orderBy("event_date", "event_type")
    events = events.withColumn("prev_event_type", F.lag("event_type").over(windowSpecDedup))
    events = events.filter((col("prev_event_type").isNull()) | (col("event_type") != col("prev_event_type")))
    
    # Restore window spec after filtering
    windowSpec = Window.partitionBy("Ticker").orderBy("event_date", "event_type")
    
    # Compute the end date by finding the next REMOVE event
    events_with_next = events.withColumn("next_date", lead("event_date").over(windowSpec)) \
                             .withColumn("next_event", lead("event_type").over(windowSpec))
    
    df_consolidated = events_with_next.filter(col("event_type") == "ADD").select(
        col("Ticker"),
        col("event_date").alias("Date_start"),
        when(col("next_event") == "REMOVE", col("next_date")).otherwise(None).alias("Date_end")
    )

    # Some events might result in duplicates if ADD dates are too close or identical overlapping chunks
    # We clean these up by removing nested duplicates: grouping by exact periods.
    # To keep it robust, any secondary cleanup or window functions could be applied here if needed.
    # Clean up duplicate Date_end for the same Ticker (overlapping presence)
    # We want to keep the OLDEST Date_start for each Ticker among instances that have the same Date_end
    # This deduplicates overlaps where a ticker appears added multiple times in short succession 
    df_cleaned = df_consolidated.groupBy("Ticker", "Date_end").agg(
        F.min("Date_start").alias("Date_start")
    )
    
    # Replace NULL Date_end with today's date
    # Also drop anomalous records where Date_start > Date_end
    df_final = df_cleaned.withColumn("Date_end", F.coalesce(col("Date_end"), F.current_date())) \
                         .filter(col("Date_start") <= col("Date_end")) \
                         .select("Ticker", "Date_start", "Date_end")

    tickers_to_exclude = ['EF', 'JBL', 'HP', 'TMUS', 'FMCC', 'FNMA', 'CTX', 'AET', 'MXIM', 'PARA']
    df_final = df_final.filter(~col("Ticker").isin(tickers_to_exclude))

    return df_final


def save_history_to_lake(df, output_path):
    """
    Saves the consolidated composition DataFrame to Delta Lake.
    """
    logger.info(f"💾 Saving Consolidated History to {output_path}...")
    
    try:
        # Write to Delta (Overwrite mode for full history refresh)
        df.write.format("delta") \
            .mode("overwrite") \
            .save(output_path)
            
        logger.success("✅ Success! Consolidated History saved.")
        
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
        # 1. Create Spark Session
        logger.info("🚀 Creating Spark Session...")
        spark = create_spark_session(app_name="SP500_Consolidated_History")

        # 2. Process Data
        df_consolidated = process_sp500_consolidated_history(spark)

        # Show a preview
        active_count = df_consolidated.filter(col("Date_end").isNull()).count()
        logger.info(f"📊 Processed {df_consolidated.count()} total periods. {active_count} active tickers without Date_end.")
        df_consolidated.show(10, truncate=False)

        # 3. Save Data
        save_history_to_lake(df_consolidated, Paths.SP500_CONSOLIDATED_HISTORY)

    except Exception as e:
        logger.critical(f"❌ Critical Error: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
