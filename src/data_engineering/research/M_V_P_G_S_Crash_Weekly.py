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
    logger.info("🚀 Starting Spark session for MASTER LAYER (Momentum + Value + Trend + Crash)...")
    spark = create_spark_session()

    try:
        # ==========================================
        # 1. LOAD ALL DATA
        # ==========================================
        df_momentum = read_silver_table(spark, Paths.SP500_MOMENTUM_WEEKLY_GOLD, "Weekly Momentum Gold")
        df_ratios = read_silver_table(spark, Paths.SP500_RATIOS, "Financial Ratios")
        df_metrics = read_silver_table(spark, Paths.SP500_KEY_METRICS, "Key Metrics")
        df_fsg = read_silver_table(spark, Paths.SP500_FINANCIAL_STATEMENT_GROWTH, "Financial Statement Growth")
        df_surprise = read_silver_table(spark, Paths.SP500_EARNINGS_SURPRISE_SILVER, "Earnings Surprise")
        df_gspc = read_silver_table(spark, Paths.MACRO_PRICES_SILVER, "Macro Indicators")
        df_crash = read_silver_table(spark, Paths.SP500_CRASH_SILVER, "Crash Indicators")
        
        # ==========================================
        # 2. PROCESS MACRO TREND (GSPC 56-Week MA)
        # ==========================================
        logger.info("📊 Processing GSPC Market Trend (Bull/Bear Flags)...")
        
        df_trend_base = df_gspc.filter(F.col("symbol") == "^GSPC") \
                         .withColumn("year_week", F.to_date(F.date_trunc("week", F.to_date(F.col("date")))))
        
        w_gspc = Window.partitionBy("year_week").orderBy(F.col("date").desc())
        df_trend = df_trend_base.withColumn("rn", F.row_number().over(w_gspc)) \
                          .filter(F.col("rn") == 1) \
                          .select(
                              F.col("year_week").alias("join_week_macro"),
                              F.col("adjClose").alias("gspc_price"),
                              F.col("volume").alias("volume_GSPC")
                          )
                          
        # Moyenne mobile sur 56 semaines
        w_56w = Window.orderBy("join_week_macro").rowsBetween(-55, 0)
        df_trend = df_trend.withColumn("gspc_ma_56w", F.avg("gspc_price").over(w_56w))
        
        # Drapeaux de marché sécurisés contre les Nul (Burn-in period)
        df_trend = df_trend.withColumn(
            "bull_market_ma", F.when(F.col("gspc_ma_56w").isNotNull() & (F.col("gspc_price") > F.col("gspc_ma_56w")), 1).otherwise(0)
        ).withColumn(
            "bear_market_ma", F.when(F.col("gspc_ma_56w").isNotNull() & (F.col("gspc_price") <= F.col("gspc_ma_56w")), 1).otherwise(0)
        ).withColumnRenamed("gspc_price", "adjClose_GSPC").drop("gspc_ma_56w")

        # ==========================================
        # 3. PROCESS CRASH INDICATORS (VIX & VIX3M)
        # ==========================================
        logger.info("💥 Processing Crash Indicators...")
        
        df_crash = df_crash.withColumn("year_week", F.to_date(F.date_trunc("week", F.to_date(F.col("date")))))
        w_crash = Window.partitionBy("symbol", "year_week").orderBy(F.col("date").desc())
        
        df_crash_weekly = df_crash.withColumn("rn", F.row_number().over(w_crash)) \
                                  .filter(F.col("rn") == 1)
                                  
        df_crash_pivot = df_crash_weekly.groupBy("year_week").pivot("symbol", ["^VIX", "^VIX3M"]).agg(F.first("close"))
        
        df_crash_pivoted = df_crash_pivot.select(
            F.col("year_week").alias("join_week_crash"),
            F.col("^VIX").alias("vix"),
            F.col("^VIX3M").alias("vix3m")
        ).withColumn("vix_vix3m_ratio", F.col("vix") / F.col("vix3m"))

        # ==========================================
        # 4. PREPARE FUNDAMENTALS (Avoid duplicate columns)
        # ==========================================
        logger.info("🔗 Preparing fundamental datasets and resolving overlapping columns...")

        df_momentum = df_momentum.withColumn("is_momentum_date", F.lit(True))
        
        mom_cols = set(df_momentum.columns)
        ratios_unique_cols = [c for c in df_ratios.columns if c not in mom_cols and c not in ['symbol', 'date']]
        df_ratios = df_ratios.select(['symbol', 'date'] + ratios_unique_cols)
        
        combined_cols = mom_cols.union(set(ratios_unique_cols))
        metrics_unique_cols = [c for c in df_metrics.columns if c not in combined_cols and c not in ['symbol', 'date']]
        df_metrics = df_metrics.select(['symbol', 'date'] + metrics_unique_cols)
        
        combined_cols_for_fsg = combined_cols.union(set(metrics_unique_cols))
        fsg_unique_cols = [c for c in df_fsg.columns if c not in combined_cols_for_fsg and c not in ['symbol', 'date']]
        df_fsg = df_fsg.select(['symbol', 'date'] + fsg_unique_cols)
        
        combined_cols_for_surprise = combined_cols_for_fsg.union(set(fsg_unique_cols))
        surprise_unique_cols = [c for c in df_surprise.columns if c not in combined_cols_for_surprise and c not in ['symbol', 'date']]
        df_surprise = df_surprise.select(['symbol', 'date'] + surprise_unique_cols)

        # ==========================================
        # 5. THE MASTER JOIN
        # ==========================================
        logger.info("🤝 Merging Fundamental data (Outer Join) onto Momentum base...")
        
        df_combined = df_momentum.join(df_ratios, ["symbol", "date"], "outer") \
                                 .join(df_metrics, ["symbol", "date"], "outer") \
                                 .join(df_fsg, ["symbol", "date"], "outer") \
                                 .join(df_surprise, ["symbol", "date"], "outer")
        
        # Filtre Hebdomadaire & Création de la clé de jointure pour la Macro
        # On s'assure d'utiliser to_date pour aligner parfaitement les semaines
        df_combined = df_combined.withColumn("join_week", F.to_date(F.date_trunc("week", F.to_date(F.col("date")))))
        
        logger.info("🤝 Merging Macro & Crash data (Left Join)...")
        df_combined = df_combined.join(df_trend, df_combined.join_week == df_trend.join_week_macro, "left") \
                                 .drop("join_week_macro")
                                 
        df_combined = df_combined.join(df_crash_pivoted, df_combined.join_week == df_crash_pivoted.join_week_crash, "left") \
                                 .drop("join_week_crash", "join_week")

        # ==========================================
        # 6. FORWARD FILL (Fundamentals + Macro)
        # ==========================================
        logger.info("⏳ Applying Forward Fill on all missing values...")

        # Toutes les colonnes qui viennent des jointures doivent être comblées si les dates ne s'alignent pas
        cols_to_ffill = ratios_unique_cols + metrics_unique_cols + fsg_unique_cols + surprise_unique_cols + \
                        ["adjClose_GSPC", "volume_GSPC", "bull_market_ma", "bear_market_ma", "vix", "vix3m", "vix_vix3m_ratio"]

        w_ffill = Window.partitionBy("symbol").orderBy("date").rowsBetween(Window.unboundedPreceding, Window.currentRow)

        exprs = []
        for c in df_combined.columns:
            if c in cols_to_ffill:
                exprs.append(F.last(F.col(c), ignorenulls=True).over(w_ffill).alias(c))
            else:
                exprs.append(F.col(c))
                
        df_combined = df_combined.select(*exprs)


        # ==========================================
        # 6.5. FEATURE ENGINEERING (ADVANCED RATIOS)
        # ==========================================
        logger.info("🧪 Calculating Advanced Valuation Ratios and Size Metrics...")

        # 1. priceToGrahamNumber (La juste valeur)
        # Protection : On ne divise que si le grahamNumber est strictement positif
        df_combined = df_combined.withColumn(
            "priceToGrahamNumber",
            F.when((F.col("grahamNumber").isNotNull()) & (F.col("grahamNumber") > 0),
                   F.col("adjClose") / F.col("grahamNumber")).otherwise(F.lit(None))
        )

        # 2. ncavToMarketCap (L'aubaine Net-Net)
        # Protection : On ne divise que si la Market Cap est > 0
        df_combined = df_combined.withColumn(
            "ncavToMarketCap",
            F.when((F.col("marketCap").isNotNull()) & (F.col("marketCap") > 0),
                   F.col("netCurrentAssetValue") / F.col("marketCap")).otherwise(F.lit(None))
        )

        # 3. cashYield (Le matelas de sécurité)
        # Protection : Le prix (adjClose) doit être > 0
        df_combined = df_combined.withColumn(
            "cashYield",
            F.when((F.col("adjClose").isNotNull()) & (F.col("adjClose") > 0),
                   F.col("cashPerShare") / F.col("adjClose")).otherwise(F.lit(None))
        )

        # 4. workingCapitalToMarketCap (L'efficacité du Capital Court Terme)
        df_combined = df_combined.withColumn(
            "workingCapitalToMarketCap",
            F.when((F.col("marketCap").isNotNull()) & (F.col("marketCap") > 0),
                   F.col("workingCapital") / F.col("marketCap")).otherwise(F.lit(None))
        )

        # 5. log_marketCap & log_enterpriseValue (Les Primes de Taille)
        # Protection : Le log(x) n'existe que si x > 0
        df_combined = df_combined.withColumn(
            "log_marketCap",
            F.when((F.col("marketCap").isNotNull()) & (F.col("marketCap") > 0),
                   F.log(F.col("marketCap"))).otherwise(F.lit(None))
        ).withColumn(
            "log_enterpriseValue",
            F.when((F.col("enterpriseValue").isNotNull()) & (F.col("enterpriseValue") > 0),
                   F.log(F.col("enterpriseValue"))).otherwise(F.lit(None))
        )

        # ==========================================
        # 7. FINAL CLEANUP
        # ==========================================
        logger.info("🧹 Filtering to weekly Momentum dates and cleaning NAs...")
        
        # Retour à la fréquence hebdomadaire stricte dictée par la table Momentum
        df_final = df_combined.filter(F.col("is_momentum_date") == True).drop("is_momentum_date")
        
        # Supprime la première année de données pour chaque action en raison de la MA 56w (Burn-in)
        df_final = df_final.dropna()

        # Suppression des "Per Share" et montants absolus toxiques
        cols_to_drop = [
            'revenuePerShare', 'netIncomePerShare', 'interestDebtPerShare', 
            'cashPerShare', 'bookValuePerShare', 'tangibleBookValuePerShare', 
            'shareholdersEquityPerShare', 'operatingCashFlowPerShare', 
            'capexPerShare', 'freeCashFlowPerShare', 'grahamNumber', 'grahamNetNet',
            'workingCapital', 'investedCapital', 'averageReceivables', 
            'averagePayables', 'averageInventory', 'freeCashFlowToEquity', 
            'freeCashFlowToFirm', 'tangibleAssetValue', 'netCurrentAssetValue',
            'dividendYieldPercentage' # Le doublon dont on avait parlé
        ]
        
        # Ne drop que les colonnes qui existent réellement dans le DataFrame
        existing_cols_to_drop = [c for c in cols_to_drop if c in df_final.columns]
        df_final = df_final.drop(*existing_cols_to_drop)

        tickers_to_exclude = ['EP', 'JBL', 'HP', 'TMUS', 'FMCC', 'FNMA', 'CTX', 'AET', 'MXIM', 'PARA', 'BHF', 'BCR', 'AN', 'CPGX', 'CMCSK', 'CMA', 'ILMN', 'RHI', 'PXD', 'XRAY', 'VFC', 'DXC', 'VNO', 'MBC', 'UA', 'DISCK', 'INFO', 'HFC', 'NBL', 'RTN', 'M', 'TEL', 'GM', 'GGP', 'UIS', 'COOP', 'SAF', 'LEHMQ', 'FMCC', 'FNMA', 'FDC', 'MERQ', 'G', 'RNB', 'COC-B', 'RAL', 'TOS', 'AFS-A', 'UMG', 'ASR']
        df_final = df_final.filter(~F.col("symbol").isin(tickers_to_exclude))

        # ==========================================
        # 8. SAVE TO DELTA
        # ==========================================
        MASTER_GOLD_PATH = Paths.SP500_MOMENTUM_VALUE_PROFITABLE_GROWTH_SURPRISE_CRASH_WEEKLY_GOLD
        
        logger.info(f"💾 Saving complete MASTER data to Gold: {MASTER_GOLD_PATH}")
        
        df_final.repartition("symbol") \
            .write.format("delta") \
            .option("overwriteSchema", "true") \
            .partitionBy("symbol") \
            .mode("overwrite") \
            .save(MASTER_GOLD_PATH)
        
        logger.info("✅ Master Gold layer created successfully!")

    except Exception as e:
        logger.error(f"❌ Error during Master Gold transition: {e}")
        sys.exit(1)

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark Session stopped.")

if __name__ == "__main__":
    main()