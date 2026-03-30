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
        
        # 2. Add WEEKLY Truncation Date
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

        # 5. WEEKLY Aggregations
        logger.info("🔄 Aggregating to WEEKLY features...")
        
        w_week_last_day = Window.partitionBy("symbol", "year_week").orderBy(col("date").desc())
        
        df_eow_snapshot = df_stocks \
            .withColumn("rn", row_number().over(w_week_last_day)) \
            .filter(col("rn") == 1) \
            .select("symbol", "year_week", "date", "adjClose", "volume", "market_return", "risk_free_rate")
            
        df_weekly_agg = df_stocks.groupBy("symbol", "year_week").agg(
            stddev("daily_return").alias("retvol"),
            spark_max("daily_return").alias("maxret"),
            avg(spark_abs(col("daily_return")) / col("dollar_volume")).alias("ill"),
            (exp(sum(log(1 + col("daily_return")))) - 1).alias("mom1w"), 
            (covar_samp("stock_excess_req", "market_excess_req") / var_samp("market_excess_req")).alias("beta")
        )
        
        df_gold = df_eow_snapshot.join(df_weekly_agg, ["symbol", "year_week"], "inner")
        
        # 6. Rolling Window Features (Momentum AND Moving Averages)
        logger.info("⏳ Calculating rolling momentum features and Moving Averages...")
        
        # Define windows (Note: -3 means current week + 3 previous weeks = 4 weeks total)
        w_rolling_4w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-3, 0)
        w_rolling_12w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-11, 0)
        w_rolling_13w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-12, 0) # Kept from your original for mom3m
        w_rolling_25w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-24, 0)
        w_rolling_26w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-25, 0) # Kept for mom6m
        w_rolling_50w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-49, 0)
        w_rolling_52w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-51, 0) # Kept for mom12m
        
        w_rolling_156w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-155, 0)
        w_rolling_260w = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-259, 0)
        w_rolling_past_6m = Window.partitionBy("symbol").orderBy("year_week").rowsBetween(-51, -27)
        
        df_gold = df_gold \
            .withColumn("mom1m", exp(sum(log(1 + col("mom1w"))).over(w_rolling_4w)) - 1) \
            .withColumn("mom3m", exp(sum(log(1 + col("mom1w"))).over(w_rolling_13w)) - 1) \
            .withColumn("mom6m", exp(sum(log(1 + col("mom1w"))).over(w_rolling_26w)) - 1) \
            .withColumn("mom12m", exp(sum(log(1 + col("mom1w"))).over(w_rolling_52w)) - 1) \
            .withColumn("mom36m", exp(sum(log(1 + col("mom1w"))).over(w_rolling_156w)) - 1) \
            .withColumn("mom60m", exp(sum(log(1 + col("mom1w"))).over(w_rolling_260w)) - 1) \
            .withColumn("rolling_max_52w", spark_max("adjClose").over(w_rolling_52w)) \
            .withColumn("high52", col("adjClose") / col("rolling_max_52w")) \
            .withColumn("past_6m_ret", exp(sum(log(1 + col("mom1w"))).over(w_rolling_past_6m)) - 1) \
            .withColumn("chmom", col("mom6m") - col("past_6m_ret")) \
            .withColumn("ma4_raw", avg("adjClose").over(w_rolling_4w)) \
            .withColumn("ma12_raw", avg("adjClose").over(w_rolling_12w)) \
            .withColumn("ma25_raw", avg("adjClose").over(w_rolling_25w)) \
            .withColumn("ma50_raw", avg("adjClose").over(w_rolling_50w))

        # --- NOUVEAU : Transformation des Moving Averages ---
        # Un réseau de neurones n'aime pas les prix bruts (ex: 150$). 
        # On calcule l'écart en % entre le prix actuel et la moyenne mobile.
        df_gold = df_gold \
            .withColumn("dist_ma4", (col("adjClose") / col("ma4_raw")) - 1) \
            .withColumn("dist_ma12", (col("adjClose") / col("ma12_raw")) - 1) \
            .withColumn("dist_ma25", (col("adjClose") / col("ma25_raw")) - 1) \
            .withColumn("dist_ma50", (col("adjClose") / col("ma50_raw")) - 1)

        # 8. Cross-Sectional Rank Normalization [-1, 1] & Zero-Filling NaNs
        logger.info("🧮 Applying Cross-Sectional Rank Normalization [-1, 1] and filling NaNs with 0...")
        
        # Ajout des nouvelles features de distance aux MAs dans la liste à normaliser
        features_to_normalize = [
            "retvol", "maxret", "ill", "beta", 
            "mom1w", "mom1m", "mom3m", "mom6m", "mom12m", "mom36m", "mom60m", 
            "chmom", "high52",
            "dist_ma4", "dist_ma12", "dist_ma25", "dist_ma50"
        ]
        
        w_cs = Window.partitionBy("date")
        
        exprs = []
        for c in df_gold.columns:
            if c not in features_to_normalize:
                exprs.append(F.col(c))
                
        for c in features_to_normalize:
            w_order = Window.partitionBy("date").orderBy(F.col(c).asc_nulls_last())
            valid_count = F.count(c).over(w_cs)
            exact_rank = F.rank().over(w_order)
            
            norm_expr = F.when(valid_count > 1, 
                           (2.0 * (exact_rank - 1) / (valid_count - 1)) - 1.0
                        ).when(valid_count == 1, 
                           0.0
                        ).otherwise(F.lit(None))
            
            exprs.append(
                F.when(F.col(c).isNotNull(), norm_expr).otherwise(0.0).alias(c)
            )
            
        df_gold = df_gold.select(*exprs)

        # Drop des colonnes temporaires (y compris les MA brutes)
        df_gold = df_gold.drop("year_week", "rolling_max_52w", "past_6m_ret", 
                               "ma4_raw", "ma12_raw", "ma25_raw", "ma50_raw")

        logger.info(f"💾 Saving complete Weekly Indicators data to Gold: {Paths.SP500_MOMENTUM_WEEKLY_GOLD}")
        
        df_gold.repartition("symbol") \
            .write.format("delta") \
            .option("overwriteSchema", "true") \
            .partitionBy("symbol") \
            .mode("overwrite") \
            .save(Paths.SP500_MOMENTUM_WEEKLY_GOLD)
        
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
