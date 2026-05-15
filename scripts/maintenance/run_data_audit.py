import os
import sys
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta

# Setup paths
sys.path.append(os.getcwd())
from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

def run_full_audit():
    logger.info("🚀 Lancement de l'Audit de Santé des Données (Bronze Layer)")
    
    spark = create_spark_session(app_name="Data_Health_Audit")
    
    try:
        # 1. Chargement des données Bronze
        logger.info(f"Lecture des données depuis {Paths.DATA_RAW_2B}")
        df = spark.read.format("delta").load(Paths.DATA_RAW_2B)
        
        # 2. Identification de la date de référence (Aujourd'hui ou max global)
        global_max_date = df.selectExpr("max(Date)").collect()[0][0]
        logger.info(f"📅 Date la plus récente en base : {global_max_date}")
        
        # 3. Calcul par ticker : Dernière date avec un prix VALIDE (non-null, non-nan)
        # On filtre les Close IS NOT NULL et on cherche le max(Date) par Ticker
        health_check = df.filter("Close IS NOT NULL") \
                         .groupBy("Ticker") \
                         .agg({"Date": "max"}) \
                         .withColumnRenamed("max(Date)", "LastValidDate")
        
        # 4. Conversion en Pandas pour analyse
        pdf = health_check.toPandas()
        pdf['LastValidDate'] = pd.to_datetime(pdf['LastValidDate'])
        global_max_dt = pd.to_datetime(global_max_date)
        
        # 5. Identification des tickers "Stuck" (Bloqués depuis plus de 7 jours)
        pdf['DaysBehind'] = (global_max_dt - pdf['LastValidDate']).dt.days
        stuck_tickers = pdf[pdf['DaysBehind'] > 7].sort_values('DaysBehind', ascending=False)
        
        # 6. Rapport Final
        print("\n" + "="*50)
        print("📊 RAPPORT D'AUDIT : TICKERS BLOQUÉS")
        print("="*50)
        
        if stuck_tickers.empty:
            logger.success("✅ Félicitations ! Tous les tickers sont à jour.")
        else:
            logger.warning(f"⚠️ {len(stuck_tickers)} tickers sont en retard de plus de 7 jours.")
            print(stuck_tickers[['Ticker', 'LastValidDate', 'DaysBehind']].to_string(index=False))
            
            # Focus sur AAOI pour confirmation
            if 'AAOI' in stuck_tickers['Ticker'].values:
                print(f"\n🚨 ALERTE : AAOI est bloqué depuis {stuck_tickers[stuck_tickers['Ticker']=='AAOI']['DaysBehind'].values[0]} jours.")
        
        print("="*50 + "\n")

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'audit : {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    run_full_audit()
