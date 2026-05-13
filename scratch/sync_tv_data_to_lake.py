import os
import sys
import pandas as pd
from loguru import logger
from pyspark.sql.functions import col, to_date
from pyspark.sql.types import StringType, DoubleType, LongType

# Configuration environnement Spark
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

# Ajout du chemin du projet pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def sync_tickers_to_lake(spark, local_csv="scratch/tradingview_tickers_2B.csv"):
    """
    Remplace la liste des tickers dans le Lake par la nouvelle liste TradingView.
    """
    logger.info(f"🔄 Étape 1 : Remplacement de la liste des tickers ({Paths.LIST_TICKER_2B})...")
    
    if not os.path.exists(local_csv):
        logger.error(f"❌ Fichier local introuvable : {local_csv}")
        return

    # Chargement et préparation
    df_pd = pd.read_csv(local_csv)
    
    # Mapping des colonnes disponibles
    df_pd = df_pd.rename(columns={
        'name': 'symbol',
        'description': 'companyName',
        'market_cap_basic': 'marketCap',
        'exchange': 'exchangeShortName'
    })
    
    # Conversion Spark
    sdf = spark.createDataFrame(df_pd[['symbol', 'companyName', 'marketCap', 'exchangeShortName']])
    
    # Ajout des colonnes manquantes pour compatibilité Silver/Gold (Valeurs nulles)
    from pyspark.sql.functions import lit
    sdf = sdf.withColumn("sector", lit(None).cast("string")) \
             .withColumn("industry", lit(None).cast("string")) \
             .withColumn("beta", lit(None).cast("double")) \
             .withColumn("price", lit(None).cast("double")) \
             .withColumn("lastAnnualDividend", lit(None).cast("double")) \
             .withColumn("volume", lit(None).cast("double")) \
             .withColumn("exchange", col("exchangeShortName")) \
             .withColumn("country", lit(None).cast("string")) \
             .withColumn("isEtf", lit(False)) \
             .withColumn("isFund", lit(False)) \
             .withColumn("isActivelyTrading", lit(True))

    # Sauvegarde (Overwrite + OverwriteSchema pour la sécurité)
    sdf.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .save(Paths.LIST_TICKER_2B)
    logger.success("✅ Liste des tickers mise à jour (format compatible conservé).")

def sync_history_to_lake(spark, local_parquet="scratch/history_2y_2B.parquet"):
    """
    Remplace les données historiques par les nouvelles données téléchargées.
    Inclus le calcul des versions Weekly et Monthly pour la cohérence.
    """
    logger.info(f"🔄 Étape 2 : Remplacement des données historiques ({Paths.DATA_RAW_2B})...")
    
    if not os.path.exists(local_parquet):
        logger.error(f"❌ Fichier local introuvable : {local_parquet}")
        return

    # 1. Chargement Daily
    df_pd = pd.read_parquet(local_parquet)
    
    # Nettoyage des noms de colonnes (supprimer les espaces comme 'Adj Close')
    df_pd.columns = [c.replace(' ', '') for c in df_pd.columns]
    
    # Nettoyage et typage strict pour Spark
    df_pd['Date'] = pd.to_datetime(df_pd['Date']).dt.strftime('%Y-%m-%d')
    df_pd['Ticker'] = df_pd['Ticker'].astype(str)
    
    # Mapping colonnes (YFinance renvoie souvent Close et Adj Close)
    # Le script de téléchargement utilise auto_adjust=True, donc Close == Adj Close
    if 'Close' in df_pd.columns and 'AdjClose' not in df_pd.columns:
        df_pd['AdjClose'] = df_pd['Close']

    # Conversion Spark avec types corrects
    sdf_daily = spark.createDataFrame(df_pd)
    sdf_daily = sdf_daily.withColumn("Date", to_date(col("Date"))) \
                         .withColumn("Ticker", col("Ticker").cast(StringType())) \
                         .withColumn("Open", col("Open").cast(DoubleType())) \
                         .withColumn("High", col("High").cast(DoubleType())) \
                         .withColumn("Low", col("Low").cast(DoubleType())) \
                         .withColumn("Close", col("Close").cast(DoubleType())) \
                         .withColumn("AdjClose", col("AdjClose").cast(DoubleType())) \
                         .withColumn("Volume", col("Volume").cast(LongType()))

    # 2. Génération des versions Weekly et Monthly (Resampling)
    # On réutilise la logique de resampling sur le DataFrame Pandas car c'est plus simple pour un remplacement total
    logger.info("🔧 Génération des versions hebdomadaires et mensuelles...")
    df_pd['Date'] = pd.to_datetime(df_pd['Date'])
    df_pd_indexed = df_pd.set_index('Date')
    
    def resample_data(freq):
        resampled = df_pd_indexed.groupby('Ticker').resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'AdjClose': 'last',
            'Volume': 'sum'
        }).reset_index()
        resampled['Date'] = resampled['Date'].dt.strftime('%Y-%m-%d')
        sdf = spark.createDataFrame(resampled)
        return sdf.withColumn("Date", to_date(col("Date")))

    sdf_weekly = resample_data('W-FRI')
    sdf_monthly = resample_data('BM')

    # 3. Sauvegardes finales (Overwrite complet + OverwriteSchema)
    logger.info("💾 Écriture finale dans Delta Lake (Bronze Layer)...")
    sdf_daily.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(Paths.DATA_RAW_2B)
    sdf_weekly.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(Paths.DATA_RAW_2B_WEEKLY)
    sdf_monthly.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(Paths.DATA_RAW_2B_MONTHLY)
    
    logger.success("✅ Données historiques (Daily/Weekly/Monthly) remplacées avec succès !")

def main():
    logger.info("🚀 Démarrage du script de synchronisation vers le Data Lake...")
    
    spark = None
    try:
        spark = create_spark_session(app_name="Sync_TV_to_Lake")
        
        # 1. Sync Tickers
        sync_tickers_to_lake(spark)
        
        # 2. Sync History
        sync_history_to_lake(spark)
        
    except Exception as e:
        logger.error(f"❌ Erreur critique lors de la synchronisation : {e}")
    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Spark session arrêtée.")

if __name__ == "__main__":
    main()
