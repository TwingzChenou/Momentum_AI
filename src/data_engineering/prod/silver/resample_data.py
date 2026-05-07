import os
import sys
import argparse
import pandas as pd
from loguru import logger
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session

def process_resampling(df_daily, frequency):
    """
    Resamples daily data to the specified frequency (W-FRI or M).
    """
    start_resample = time.time()
    freq_label = "Hebdomadaire" if frequency == 'W-FRI' else "Mensuel"
    logger.info(f"🔧 Rééchantillonnage {freq_label} ({frequency}) de {len(df_daily)} points...")
    
    # Ensure Date is datetime and Ticker is present
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily = df_daily.sort_values(['Ticker', 'Date'])
    df_daily = df_daily.set_index('Date')
    
    # Aggregation rules
    agg_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    # Check if AdjClose exists (common in our pipeline)
    if 'AdjClose' in df_daily.columns:
        agg_rules['AdjClose'] = 'last'
    
    # Resample by Ticker
    resampled = df_daily.groupby('Ticker').resample(frequency).agg(agg_rules).reset_index()
    
    # Cleaning: Handle Close_Raw if needed (standard in our Gold layer)
    if 'AdjClose' in resampled.columns:
        # Fallback: Si AdjClose est NULL (cas des données historiques anciennes), 
        # on utilise le Close standard pour ne pas perdre la donnée
        resampled['AdjClose'] = resampled['AdjClose'].fillna(resampled['Close'])
        resampled = resampled.rename(columns={'Close': 'Close_Raw', 'AdjClose': 'Close'})
    
    # Final cleanup: Drop rows where Close is NaN
    resampled = resampled.dropna(subset=['Close'])
    
    duration = time.time() - start_resample
    logger.info(f"✨ Rééchantillonnage terminé en {duration:.2f}s. {len(resampled)} points générés.")
    return resampled

def main():
    parser = argparse.ArgumentParser(description="Generic Data Resampler for Silver Layer")
    parser.add_argument("--source", required=True, help="GCS path to source Daily Delta table")
    parser.add_argument("--target", required=True, help="GCS path to target Resampled Delta table")
    parser.add_argument("--freq", choices=['W-FRI', 'M'], default='W-FRI', help="Resampling frequency (W-FRI or M)")
    parser.add_argument("--name", default="Resampler", help="Spark Session Name")
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info(f"🎬 Démarrage du rééchantillonnage : {args.source} -> {args.target} ({args.freq})")
    
    spark = create_spark_session(f"Silver_Resample_{args.name}")
    
    try:
        # 1. Load Source (Détection automatique Parquet ou Delta)
        logger.info(f"📥 Chargement des données Daily depuis {args.source}")
        if args.source.endswith(".parquet"):
            sdf_daily = spark.read.parquet(args.source)
        else:
            sdf_daily = spark.read.format("delta").load(args.source)
        
        # 2. Conversion to Pandas (Efficient for our volume of cleaned data)
        df_daily = sdf_daily.toPandas()
        
        if df_daily.empty:
            logger.warning("⚠️ Aucune donnée source trouvée.")
            return

        # 3. Process
        df_resampled = process_resampling(df_daily, args.freq)
        
        # 4. Save to Delta
        save_start = time.time()
        sdf_resampled = spark.createDataFrame(df_resampled)
        
        logger.info(f"💾 Sauvegarde de {len(df_resampled)} lignes vers {args.target}")
        sdf_resampled.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(args.target)
        
        save_duration = time.time() - save_start
        logger.success(f"💾 Sauvegarde terminée en {save_duration:.2f}s")

    except Exception as e:
        logger.critical(f"❌ Erreur critique lors du rééchantillonnage : {e}")
        sys.exit(1)
    finally:
        if spark: spark.stop()

if __name__ == "__main__":
    main()
