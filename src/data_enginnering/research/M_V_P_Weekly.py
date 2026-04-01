import os
import sys
from loguru import logger
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
    """Reads a cleaned table from Silver/Gold Delta Lake."""
    logger.info(f"Reading {name} from {path}...")
    df = spark.read.format("delta").load(path)
    logger.info(f"Found {df.count()} records in {name}.")
    return df

def main():
    setup_logging()
    logger.info("Starting Spark session for Momentum + Value + Profitable + Trend Layer...")
    spark = create_spark_session()

    try:
        # 1. Load Data
        df_momentum = read_silver_table(spark, Paths.SP500_MOMENTUM_WEEKLY_GOLD, "Weekly Momentum Gold")
        df_ratios = read_silver_table(spark, Paths.SP500_RATIOS, "Financial Ratios")
        df_metrics = read_silver_table(spark, Paths.SP500_KEY_METRICS, "Key Metrics")
        
        # 📈 NOUVEAU : Chargement des données Macro (S&P 500)
        df_gspc = read_silver_table(spark, Paths.MACRO_PRICES_SILVER, "Macro Indicators")
        
        logger.info("📊 Processing GSPC Market Trend (Bull/Bear Flags)...")
        
        # --- CALCUL DU RÉGIME DE MARCHÉ ---
        # On filtre le S&P 500 et on crée une clé de semaine pour la jointure
        df_gspc = df_gspc.filter(F.col("symbol") == "^GSPC") \
                         .withColumn("year_week", F.date_trunc("week", F.col("date")))
        
        # On garde uniquement le prix de la dernière journée de la semaine
        w_gspc = Window.partitionBy("year_week").orderBy(F.col("date").desc())
        df_trend = df_gspc.withColumn("rn", F.row_number().over(w_gspc)) \
                          .filter(F.col("rn") == 1) \
                          .select(F.col("year_week").alias("join_week"), F.col("adjClose").alias("gspc_price"))
                          
        # Calcul de la moyenne mobile sur 56 semaines (rowsBetween -55 et 0 = 56 semaines)
        w_56w = Window.orderBy("join_week").rowsBetween(-55, 0)
        df_trend = df_trend.withColumn("gspc_ma_56w", F.avg("gspc_price").over(w_56w))
        
        # Création des drapeaux Bull/Bear Market
        # Bull : Prix > MA 56
        # Bear : Prix <= MA 56
        df_trend = df_trend.withColumn(
            "bull_market", F.when(F.col("gspc_price") > F.col("gspc_ma_56w"), 1).otherwise(0)
        ).withColumn(
            "bear_market", F.when(F.col("gspc_price") <= F.col("gspc_ma_56w"), 1).otherwise(0)
        ).drop("gspc_price", "gspc_ma_56w") # Nettoyage

        # ==========================================
        
        logger.info("🔗 Preparing datasets and removing duplicate columns before join...")

        df_momentum = df_momentum.withColumn("is_momentum_date", F.lit(True))

        # 🛡️ PROTECTION ANTI-DOUBLONS DE COLONNES
        mom_cols = set(df_momentum.columns)
        
        ratios_unique_cols = [c for c in df_ratios.columns if c not in mom_cols and c not in ['symbol', 'date']]
        df_ratios = df_ratios.select(['symbol', 'date'] + ratios_unique_cols)
        
        combined_cols = mom_cols.union(set(ratios_unique_cols))
        
        metrics_unique_cols = [c for c in df_metrics.columns if c not in combined_cols and c not in ['symbol', 'date']]
        df_metrics = df_metrics.select(['symbol', 'date'] + metrics_unique_cols)

        # FULL OUTER JOIN
        df_combined = df_momentum.join(df_ratios, ["symbol", "date"], "outer") \
                                 .join(df_metrics, ["symbol", "date"], "outer")
        
        logger.info("⏳ Applying Forward Fill on fundamental data...")

        cols_to_ffill = ratios_unique_cols + metrics_unique_cols

        w_ffill = Window.partitionBy("symbol") \
                        .orderBy("date") \
                        .rowsBetween(Window.unboundedPreceding, Window.currentRow)

        exprs = []
        for c in df_combined.columns:
            if c in cols_to_ffill:
                exprs.append(F.last(F.col(c), ignorenulls=True).over(w_ffill).alias(c))
            else:
                exprs.append(F.col(c))
                
        df_combined = df_combined.select(*exprs)

        # Cleaning up and filtering back to Weekly frequency
        logger.info("🧹 Cleaning up and integrating Market Trend...")
        df_final = df_combined.filter(F.col("is_momentum_date") == True).drop("is_momentum_date")
        
        # 🔗 JOINTURE DU RÉGIME DE MARCHÉ
        # On crée la même clé `join_week` sur df_final pour faire correspondre proprement les données
        df_final = df_final.withColumn("join_week", F.date_trunc("week", F.col("date")))
        df_final = df_final.join(df_trend, "join_week", "left").drop("join_week")

        df_final = df_final.dropna()

        tickers_to_exclude = ['EP', 'JBL', 'HP', 'TMUS', 'FMCC', 'FNMA', 'CTX', 'AET', 'MXIM', 'PARA']
        df_final = df_final.filter(~F.col("symbol").isin(tickers_to_exclude))

        # NOUVELLE DESTINATION D'ENREGISTREMENT
        logger.info(f"💾 Saving complete Multi-Factor data to Gold: {Paths.SP500_MOMENTUM_VALUE_PROFITABLE_TREND_WEEKLY_GOLD}")
        
        df_final.repartition("symbol") \
            .write.format("delta") \
            .option("overwriteSchema", "true") \
            .partitionBy("symbol") \
            .mode("overwrite") \
            .save(Paths.SP500_MOMENTUM_VALUE_PROFITABLE_TREND_WEEKLY_GOLD)
        
        logger.info("✅ Gold layer (Multi-Factor + Trend Weekly) created successfully!")

    except Exception as e:
        logger.error(f"❌ Error during Multi-Factor + Trend Gold transition: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()