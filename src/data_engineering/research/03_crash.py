import os
import sys
import numpy as np
from loguru import logger
from pyspark.sql.functions import col, to_date, stddev, lag, lit
from pyspark.sql.window import Window

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def read_crash_bronze(spark):
    logger.info(f"📥 Reading crash data from Bronze: {Paths.SP500_CRASH_BRONZE}")
    df = spark.read.format("delta").load(Paths.SP500_CRASH_BRONZE)
    return df

def read_macro_silver(spark):
    logger.info(f"📥 Reading macro data from Silver: {Paths.MACRO_PRICES_SILVER}")
    df = spark.read.format("delta").load(Paths.MACRO_PRICES_SILVER)
    return df

def calculate_volatility(df_macro):
    logger.info("🧮 Calculating 3-Month Realized Volatility for S&P 500 (^GSPC) from Macro data...")
    
    # On isole l'indice de référence depuis la table Macro
    df_market = df_macro.filter(col("symbol") == "^GSPC")
    
    # On s'assure que la date est au bon format pour la future jointure
    df_market = df_market.withColumn("date", to_date(col("date")))
    
    window_spec = Window.partitionBy("symbol").orderBy("date")
    
    # Calcul du rendement quotidien (On utilise 'adjClose' si 'close' n'est pas fiable sur les indices longs)
    df_market = df_market.withColumn("prev_close", lag("adjClose").over(window_spec))
    df_market = df_market.withColumn("sp500_return", (col("adjClose") - col("prev_close")) / col("prev_close"))
    
    # Fenêtre glissante de 63 jours (environ 3 mois)
    rolling_window = Window.partitionBy("symbol").orderBy("date").rowsBetween(-62, 0)
    
    # Ecart-type glissant et annualisation
    df_market = df_market.withColumn("rolling_std", stddev("sp500_return").over(rolling_window))
    df_market = df_market.withColumn("realized_volatility_3m", col("rolling_std") * lit(np.sqrt(252)) * 100)
    
    return df_market.select("date", "realized_volatility_3m").filter(col("realized_volatility_3m").isNotNull())

def main():
    setup_logging()
    logger.info("🚀 Starting Job: S&P 500 Crash Bronze to Silver Transformation")
    spark = create_spark_session(app_name="Silver_Crash_Volatility")

    try:
        # 1. Chargement des deux sources de données
        df_bronze = read_crash_bronze(spark)
        df_macro = read_macro_silver(spark)
        
        logger.info("🧹 Cleaning & Formatting Bronze crash data...")
        df_silver = df_bronze \
            .withColumn("date", to_date(col("date"))) \
            .dropDuplicates(["symbol", "date"]) \
            .filter(col("close") > 0) \
            .dropna(subset=["close", "date", "symbol"])

        # 2. Calcul de la volatilité en utilisant UNIQUEMENT la donnée Macro
        df_vol = calculate_volatility(df_macro)
        
        # 3. Création du symbole synthétique ^VIX3M à partir des calculs macro
        df_vix3m = df_vol.select(
            col("date"),
            lit("^VIX3M").alias("symbol"),
            lit(0.0).alias("open"),
            lit(0.0).alias("high"),
            lit(0.0).alias("low"),
            col("realized_volatility_3m").alias("close"),
            col("realized_volatility_3m").alias("adjClose"),
            lit(0).cast("long").alias("volume")
        )

        # 4. Extraction des données originales du VIX (et autres)
        df_original = df_silver.select(
            "date", "symbol", "open", "high", "low", "close", "adjClose", "volume"
        )

        # 5. Calcul du ratio entre le vrai ^VIX et notre volatilité calculée ^VIX3M
        # Jointure sur la "date" qui est maintenant uniformisée grâce au to_date() des deux côtés
        df_ratio = df_original.filter(col("symbol") == "^VIX") \
            .join(df_vix3m.select("date", col("close").alias("vix3m_close")), "date", "inner") \
            .select("date", (col("close") / col("vix3m_close")).alias("vix_vix3m_ratio"))

        # 6. Assemblage final
        df_final = df_original.unionByName(df_vix3m)
        df_final = df_final.join(df_ratio, "date", "left")

        logger.info(f"💾 Saving processed data to Silver: {Paths.SP500_CRASH_SILVER}")
        
        df_final.repartition("symbol").write.format("delta") \
            .option("overwriteSchema", "true") \
            .partitionBy("symbol") \
            .mode("overwrite") \
            .save(Paths.SP500_CRASH_SILVER)
            
        logger.info("✅ Silver crash layer created successfully!")

    except Exception as e:
        logger.error(f"❌ Error during Bronze to Silver transition: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")


if __name__ == "__main__":
    main()