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
    logger.info("Starting Spark session for Momentum + Value + Profitable Layer...")
    spark = create_spark_session()

    try:
        # 1. Load Data
        df_momentum = read_silver_table(spark, Paths.SP500_MOMENTUM_WEEKLY_GOLD, "Weekly Momentum Gold")
        df_ratios = read_silver_table(spark, Paths.SP500_RATIOS, "Financial Ratios")
        df_metrics = read_silver_table(spark, Paths.SP500_KEY_METRICS, "Key Metrics")
        
        logger.info("🔗 Preparing datasets and removing duplicate columns before join...")

        # 2. Tag the base momentum rows so we know which dates to keep at the end
        df_momentum = df_momentum.withColumn("is_momentum_date", F.lit(True))

        # ==========================================
        # 🛡️ PROTECTION ANTI-DOUBLONS DE COLONNES
        # ==========================================
        # On récupère les colonnes actuelles pour comparer
        mom_cols = set(df_momentum.columns)
        
        # Pour les Ratios : On garde les clés (symbol, date) + les colonnes exclusives
        ratios_unique_cols = [c for c in df_ratios.columns if c not in mom_cols and c not in ['symbol', 'date']]
        df_ratios = df_ratios.select(['symbol', 'date'] + ratios_unique_cols)
        
        # On met à jour la liste des colonnes connues pour le prochain filtre
        combined_cols = mom_cols.union(set(ratios_unique_cols))
        
        # Pour les Metrics : On garde les clés + les colonnes exclusives
        metrics_unique_cols = [c for c in df_metrics.columns if c not in combined_cols and c not in ['symbol', 'date']]
        df_metrics = df_metrics.select(['symbol', 'date'] + metrics_unique_cols)

        # 3. FULL OUTER JOIN
        # Maintenant, la jointure est 100% propre, aucune collision possible.
        df_combined = df_momentum.join(df_ratios, ["symbol", "date"], "outer") \
                                 .join(df_metrics, ["symbol", "date"], "outer")
        
        logger.info("⏳ Applying Forward Fill on fundamental data...")

        # 4. Identify columns to forward fill (Seulement les nouvelles colonnes financières)
        cols_to_ffill = ratios_unique_cols + metrics_unique_cols

        # 5. Define the Forward Fill Window
        w_ffill = Window.partitionBy("symbol") \
                        .orderBy("date") \
                        .rowsBetween(Window.unboundedPreceding, Window.currentRow)

        # 6. Apply the Forward Fill (last non-null value)
        exprs = []
        for c in df_combined.columns:
            if c in cols_to_ffill:
                # Fill down fundamental data
                exprs.append(F.last(F.col(c), ignorenulls=True).over(w_ffill).alias(c))
            else:
                # Keep momentum data exactly as it is
                exprs.append(F.col(c))
                
        df_combined = df_combined.select(*exprs)

        # 7. Filter back to ONLY the weekly frequency
        logger.info("🧹 Cleaning up and filtering back to Weekly frequency...")
        df_final = df_combined.filter(F.col("is_momentum_date") == True).drop("is_momentum_date")

        df_final = df_final.dropna()

        tickers_to_exclude = ['EP', 'JBL', 'HP', 'TMUS', 'FMCC', 'FNMA', 'CTX', 'AET', 'MXIM', 'PARA']
        df_final = df_final.filter(~F.col("symbol").isin(tickers_to_exclude))

        # 8. Save the Final Combined Gold Table
        logger.info(f"💾 Saving complete Multi-Factor data to Gold: {Paths.SP500_MOMENTUM_VALUE_PROFITABLE_WEEKLY_GOLD}")
        
        df_final.repartition("symbol") \
            .write.format("delta") \
            .option("overwriteSchema", "true") \
            .partitionBy("symbol") \
            .mode("overwrite") \
            .save(Paths.SP500_MOMENTUM_VALUE_PROFITABLE_WEEKLY_GOLD)
        
        logger.info("✅ Gold layer (Multi-Factor Weekly) created successfully!")

    except Exception as e:
        logger.error(f"❌ Error during Multi-Factor Gold transition: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()