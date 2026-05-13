import os
import pandas as pd
from tradingview_screener import Query, Column
from loguru import logger

def download_and_verify_ticker(target_ticker="AXTI", output_file="scratch/tradingview_all_tickers.csv"):
    logger.info("🚀 Étape 1 : Téléchargement de TOUS les tickers depuis TradingView...")
    
    # On crée la requête pour un large éventail de pays
    # On augmente la limite pour être sûr de tout avoir (n_total était ~14.5k)
    query = (Query()
        .set_markets('america', 'france', 'germany', 'italy', 'uk', 'canada', 'japan', 'hongkong')
        .select('name', 'description', 'market_cap_basic', 'exchange', 'type', 'subtype')
        .where(
            Column('type') == 'stock',
            Column('is_primary') == True,
            Column('market_cap_basic') >= 2e9  
        )
        .limit(30000)) # Limite augmentée pour "tout" avoir

    try:
        n_total, df = query.get_scanner_data()
        logger.info(f"📊 {n_total} tickers identifiés. {len(df)} tickers téléchargés.")
        
        # Sauvegarde locale pour l'étape "téléchargement"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.success(f"💾 Fichier sauvegardé : {output_file}")
        
        # Étape 2 : Vérification à partir du fichier
        logger.info(f"🔍 Étape 2 : Vérification de la présence de '{target_ticker}' dans le fichier...")
        df_loaded = pd.read_csv(output_file)
        
        match = df_loaded[df_loaded['name'].str.upper() == target_ticker.upper()]
        
        if not match.empty:
            logger.success(f"🎯 '{target_ticker}' est bien PRÉSENT dans les données téléchargées !")
            print("\nDétails de l'actif trouvé :")
            print(match.to_string())
        else:
            logger.warning(f"❌ '{target_ticker}' est ABSENT des données téléchargées.")
            
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")

if __name__ == "__main__":
    download_and_verify_ticker("TCS")