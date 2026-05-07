import os
import sys
from loguru import logger
from pyspark.sql.functions import (
    col, date_trunc, row_number, lag, stddev, sum, max as spark_max, 
    exp, log, covar_samp, var_samp, avg, abs as spark_abs, count
)
from pyspark.sql.window import Window
import pyspark.sql.functions as F

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

def diagnose_beta_nans(spark, df_stocks, df_macro_combined):
    """
    Script pour analyser pourquoi le Beta retourne des valeurs nulles.
    """
    print("🔍 Démarrage du diagnostic des NaNs sur le Beta (Weekly)...")

    # On utilise 'year_week' au lieu de 'year_month'
    diagnostic_df = df_stocks.groupBy("symbol", "year_week").agg(
        (F.covar_samp("stock_excess_req", "market_excess_req") / F.var_samp("market_excess_req")).alias("beta"),
        F.count("date").alias("nb_trading_days"),
        F.var_samp("market_excess_req").alias("market_variance"),
        F.sum(F.when(F.col("market_return") == 0.0, 1).otherwise(0)).alias("nb_market_returns_zero"),
        F.sum(F.when(F.col("risk_free_rate") == 0.0, 1).otherwise(0)).alias("nb_missing_irx")
    )

    failed_betas = diagnostic_df.filter(F.col("beta").isNull() | F.isnan("beta"))
    total_failures = failed_betas.count()
    print(f"\n⚠️ Trouvé {total_failures} semaines/actions avec un Beta Null/NaN.")

    if total_failures > 0:
        print("\nRésumé des causes probables d'erreur :")
        failed_betas.select(
            F.count(F.when(F.col("nb_trading_days") < 2, 1)).alias("Cause: Moins de 2 jours (Très fréquent en Hebdo)"),
            F.count(F.when(F.col("market_variance") == 0.0, 1)).alias("Cause: Variance marché = 0"),
            F.count(F.when(F.col("nb_market_returns_zero") == F.col("nb_trading_days"), 1)).alias("Cause: Join Macro échoué")
        ).show()

    return failed_betas


def main():
    setup_logging()
    logger.info("Starting Spark session for Gold Layer (WEEKLY FREQUENCY)...")
    spark = create_spark_session()

    try:
        # 1. Load Data
        df_stocks = read_silver_table(spark, Paths.SP500_STOCK_PRICES_SILVER, "S&P 500 Prices")
        df_GSPC = read_silver_table(spark, Paths.MACRO_PRICES_SILVER, "Macro Indicators")
        df_treasury_bond = read_silver_table(spark, Paths.TREASURY_BOND_GOLD, "Treasury Bond Prices")
        
        logger.info("📈 Processing daily features before aggregating...")
        
        # 2. Add WEEKLY Truncation Date (Changed from month to week)
        df_stocks = df_stocks.withColumn("year_week", date_trunc("week", col("date")))
        df_GSPC = df_GSPC.withColumn("year_week", date_trunc("week", col("date")))
        
        # 3. Process Macro Data
        logger.info("Calculating valid macro metrics...")
        
        w_macro_gspc = Window.partitionBy("symbol").orderBy("date")
        df_gspc = df_GSPC.filter(col("symbol") == "^GSPC") \
            .withColumn("prev_adjClose", lag("adjClose").over(w_macro_gspc)) \
            .withColumn("market_return", (col("adjClose") / col("prev_adjClose")) - 1) \
            .select(col("date").alias("macro_date"), col("market_return"))
            
        df_irx = df_treasury_bond.select(
            col("date").alias("irx_date"), 
            col("daily_risk_free_rate").alias("risk_free_rate")
        )
            
        df_macro_combined = df_gspc.join(df_irx, df_gspc.macro_date == df_irx.irx_date, "outer") \
            .withColumn("macro_date", F.coalesce(col("macro_date"), col("irx_date"))) \
            .drop("irx_date")
            
        # 4. Process Daily Stock Features
        logger.info("Calculating daily stock features (Returns, Dollar Volume)...")
        w_stock_daily = Window.partitionBy("symbol").orderBy("date")
        
        df_stocks = df_stocks \
            .withColumn("prev_adjClose", lag("adjClose").over(w_stock_daily)) \
            .withColumn("daily_return", (col("adjClose") / col("prev_adjClose")) - 1) \
            .withColumn("dollar_volume", col("adjClose") * col("volume"))
            
        df_stocks = df_stocks.dropna(subset=["daily_return"])

        df_stocks = df_stocks.withColumn("join_date", F.to_date(F.col("date")))
        df_macro_combined = df_macro_combined.withColumn("join_date", F.to_date(F.col("macro_date")))
        
        df_stocks = df_stocks.join(df_macro_combined, "join_date", "left") \
            .drop("macro_date", "join_date")
            
        df_stocks = df_stocks.fillna({"risk_free_rate": 0.0, "market_return": 0.0}) \
            .withColumn("stock_excess_req", col("daily_return") - col("risk_free_rate")) \
            .withColumn("market_excess_req", col("market_return") - col("risk_free_rate"))

        diagnostic_results = diagnose_beta_nans(spark, df_stocks, df_macro_combined)
 
        # 5. WEEKLY Aggregations
        logger.info("🔄 Aggregating to WEEKLY features...")
        
        # Window to find the last trading day of the WEEK
        w_week_last_day = Window.partitionBy("symbol", "year_week").orderBy(col("date").desc())
        
        df_eow_snapshot = df_stocks \
            .withColumn("rn", row_number().over(w_week_last_day)) \
            .filter(col("rn") == 1) \
            .select("symbol", "year_week", "date", "adjClose", "volume", "market_return", "risk_free_rate")
            
        # Aggregations based on year_week
        # mom1m becomes mom1w (1-week momentum)
        df_weekly_agg = df_stocks.groupBy("symbol", "year_week").agg(
            stddev("daily_return").alias("retvol"),
            spark_max("daily_return").alias("maxret"),
            avg(spark_abs(col("daily_return")) / col("dollar_volume")).alias("ill"),
            (exp(sum(log(1 + col("daily_return")))) - 1).alias("mom1w"), 
            (covar_samp("stock_excess_req", "market_excess_req") / var_samp("market_excess_req")).alias("beta")
        )
        
        df_gold = df_eow_snapshot.join(df_weekly_agg, ["symbol", "year_week"], "inner")
        
        # 6. Rolling Window Features (mom3m, mom6m, mom12m adapted for weeks)
        logger.info("⏳ Calculating rolling momentum features (13-week, 26-week and 52-week)...")
        
        # 3 months = ~13 weeks. 6 months = ~26 weeks. 12 months = ~52 weeks.
        # We roll over the past N weeks, up to the current week (0)
        w_rolling_4w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-4, 0)
        w_rolling_13w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-12, 0)
        w_rolling_26w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-25, 0)
        w_rolling_52w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-51, 0)
        w_rolling_156w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-155, 0)
        w_rolling_260w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-259, 0)
        
        df_gold = df_gold \
            .withColumn("mom1m", exp(sum(log(1 + col("mom1w"))).over(w_rolling_4w)) - 1) \
            .withColumn("mom3m", exp(sum(log(1 + col("mom1w"))).over(w_rolling_13w)) - 1) \
            .withColumn("mom6m", exp(sum(log(1 + col("mom1w"))).over(w_rolling_26w)) - 1) \
            .withColumn("mom12m", exp(sum(log(1 + col("mom1w"))).over(w_rolling_52w)) - 1) \
            .withColumn("mom36m", exp(sum(log(1 + col("mom1w"))).over(w_rolling_156w)) - 1) \
            .withColumn("mom60m", exp(sum(log(1 + col("mom1w"))).over(w_rolling_260w)) - 1) \

        # Drop the grouping column
        df_gold = df_gold.drop("year_week")

        # 8. Drop rows with NaN values
        df_gold = df_gold.dropna()

        logger.info(f"💾 Saving complete Weekly Indicators data to Gold: {Paths.SP500_STOCK_PRICES_WEEKLY_GOLD}")
        
        # Removed repartition("symbol") to avoid creating thousands of tiny files
        df_gold.write.format("delta") \
            .option("overwriteSchema", "true") \
            .partitionBy("symbol") \
            .mode("overwrite") \
            .save(Paths.SP500_STOCK_PRICES_WEEKLY_GOLD)
        
        logger.info("✅ Gold layer (Weekly Features) created successfully!")

    except Exception as e:
        logger.error(f"❌ Error during Silver to Gold transition: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()
