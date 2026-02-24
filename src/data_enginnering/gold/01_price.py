import os
import sys
from loguru import logger
from pyspark.sql.functions import (
    col, date_trunc, row_number, lag, stddev, sum, max as spark_max, 
    exp, log, covar_samp, var_samp, avg, abs as spark_abs, count
)
from pyspark.sql.window import Window

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def read_silver_table(spark, path, name):
    """Reads a cleaned table from Silver Delta Lake."""
    logger.info(f"Reading {name} from {path}...")
    df = spark.read.format("delta").load(path)
    logger.info(f"Found {df.count()} records in {name}.")
    return df

def main():
    setup_logging()
    logger.info("Starting Spark session for Gold Layer...")
    spark = create_spark_session()

    try:
        # 1. Load Data
        df_stocks = read_silver_table(spark, Paths.SP500_STOCK_PRICES_SILVER, "S&P 500 Prices")
        df_macro = read_silver_table(spark, Paths.MACRO_PRICES_SILVER, "Macro Indicators")
        
        logger.info("📈 Processing daily features before aggregating...")
        
        # 2. Add Monthly Truncation Date
        df_stocks = df_stocks.withColumn("year_month", date_trunc("month", col("date")))
        df_macro = df_macro.withColumn("year_month", date_trunc("month", col("date")))
        
        # 3. Process Macro Data (Market Return ^GSPC & Risk-Free Rate ^IRX)
        logger.info("Calculating valid macro metrics...")
        
        # Isolate GSPC and calculate daily market return
        w_macro_gspc = Window.partitionBy("symbol").orderBy("date")
        df_gspc = df_macro.filter(col("symbol") == "^GSPC") \
            .withColumn("prev_adjClose", lag("adjClose").over(w_macro_gspc)) \
            .withColumn("market_return", (col("adjClose") / col("prev_adjClose")) - 1) \
            .select(col("date").alias("macro_date"), col("market_return"))
            
        # Isolate IRX (which already has adjClose = daily yield)
        df_irx = df_macro.filter(col("symbol") == "^IRX") \
            .select(col("date").alias("irx_date"), col("adjClose").alias("risk_free_rate"))
            
        # Combine macro data on date
        df_macro_combined = df_gspc.join(df_irx, df_gspc.macro_date == df_irx.irx_date, "inner") \
            .drop("irx_date")
            
        # 4. Process Daily Stock Features
        logger.info("Calculating daily stock features (Returns, Dollar Volume)...")
        w_stock_daily = Window.partitionBy("symbol").orderBy("date")
        
        df_stocks = df_stocks \
            .withColumn("prev_adjClose", lag("adjClose").over(w_stock_daily)) \
            .withColumn("daily_return", (col("adjClose") / col("prev_adjClose")) - 1) \
            .withColumn("dollar_volume", col("adjClose") * col("volume"))
            
        # Drop rows where we couldn't calculate daily return (e.g., first day of the dataset)
        df_stocks = df_stocks.dropna(subset=["daily_return"])
        
        # Join stocks with daily macro data to compute beta later
        df_stocks = df_stocks.join(df_macro_combined, df_stocks.date == df_macro_combined.macro_date, "left") \
            .drop("macro_date")
            
        # Compute excess returns for Beta calculation
        # If risk_free_rate is missing for a day, fill with 0
        df_stocks = df_stocks.fillna({"risk_free_rate": 0.0, "market_return": 0.0}) \
            .withColumn("stock_excess_req", col("daily_return") - col("risk_free_rate")) \
            .withColumn("market_excess_req", col("market_return") - col("risk_free_rate"))

        # 5. Monthly Aggregations
        logger.info("🔄 Aggregating to monthly features...")
        
        # Define window to find the last trading day of the month for end-of-month price/date
        w_month_last_day = Window.partitionBy("symbol", "year_month").orderBy(col("date").desc())
        
        # Create end of month snapshot to keep the true month-end date and closing price
        df_eom_snapshot = df_stocks \
            .withColumn("rn", row_number().over(w_month_last_day)) \
            .filter(col("rn") == 1) \
            .select("symbol", "year_month", "date", "adjClose", "volume")
            
        # Calculate monthly aggregations
        # - retvol: stddev of daily returns
        # - maxret: max daily return
        # - ill: Amihud Illiquidity = avg(abs(daily_return) / dollar_volume)
        # - mom1m: Compounded daily total returns = exp(sum(log(1 + daily_return))) - 1
        # - beta: covariance(stock_excess, market_excess) / variance(market_excess)
        df_monthly_agg = df_stocks.groupBy("symbol", "year_month").agg(
            stddev("daily_return").alias("retvol"),
            spark_max("daily_return").alias("maxret"),
            avg(spark_abs(col("daily_return")) / col("dollar_volume")).alias("ill"),
            (exp(sum(log(1 + col("daily_return")))) - 1).alias("mom1m"),
            (covar_samp("stock_excess_req", "market_excess_req") / var_samp("market_excess_req")).alias("beta")
        )
        
        # Join snapshot with aggregations
        df_gold = df_eom_snapshot.join(df_monthly_agg, ["symbol", "year_month"], "inner")
        
        # 6. Rolling Window Features (mom6m, mom12m)
        logger.info("⏳ Calculating rolling momentum features (mom6m, mom12m)...")
        w_rolling_6m = Window.partitionBy("symbol").orderBy("year_month").rowsBetween(-5, 0)
        w_rolling_12m = Window.partitionBy("symbol").orderBy("year_month").rowsBetween(-11, 0)
        
        # We need to compute momentum by chaining the monthly compounded returns.
        # However, since mom1m is exactly the monthly compounded return, 
        # rolling momentum is simply exp(sum(log(1 + mom1m))) - 1 over the window.
        df_gold = df_gold \
            .withColumn("mom6m", exp(sum(log(1 + col("mom1m"))).over(w_rolling_6m)) - 1) \
            .withColumn("mom12m", exp(sum(log(1 + col("mom1m"))).over(w_rolling_12m)) - 1)

        # Drop the grouping column since we preserve the actual 'date' of the snapshot
        df_gold = df_gold.drop("year_month")

        logger.info(f"💾 Saving complete Monthly Indicators data to Gold: {Paths.SP500_STOCK_PRICES_GOLD}")
        
        # Save to Gold layer
        df_gold.repartition("symbol").write.format("delta") \
            .option("overwriteSchema", "true") \
            .partitionBy("symbol") \
            .mode("overwrite") \
            .save(Paths.SP500_STOCK_PRICES_GOLD)
        
        logger.info("✅ Gold layer (Monthly Features) created successfully!")

    except Exception as e:
        logger.error(f"❌ Error during Silver to Gold transition: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
