import os
import pandas as pd
import yfinance as yf
from tradingview_screener import Query, Column
from loguru import logger
from datetime import datetime
import time

def download_tv_tickers_with_filters():
    """
    Télécharge les tickers depuis TradingView avec les filtres spécifiés par l'utilisateur.
    """
    logger.info("🚀 Étape 1 : Téléchargement des tickers depuis TradingView...")
    
    query = (Query()
        .set_markets('america', 'france', 'germany', 'italy', 'uk','japan', 'hongkong')
        .select('name', 'description', 'market_cap_basic', 'exchange', 'market', 'type', 'subtype')
        .where(
            Column('type') == 'stock',
            Column('is_primary') == True,
            Column('market_cap_basic') >= 2e9  
        )
        .limit(30000))

    try:
        n_total, df = query.get_scanner_data()
        logger.info(f"📊 {n_total} tickers identifiés. {len(df)} tickers téléchargés (Cap > 2B).")
        
        # Sauvegarde de la liste des tickers
        output_tickers = "scratch/tradingview_tickers_2B.csv"
        df.to_csv(output_tickers, index=False)
        logger.success(f"💾 Liste des tickers sauvegardée : {output_tickers}")
        
        return df
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement TradingView : {e}")
        return pd.DataFrame()

def map_tv_to_yf(row):
    """
    Mappe les symboles TradingView au format Yahoo Finance.
    """
    symbol = str(row['name'])
    exchange = str(row['exchange']).upper()
    
    # Correction pour le Canada (Format Yahoo: QSP-UN.TO au lieu de QSP.UN.TO)
    symbol = symbol.replace('.UN', '-UN')

    # Mapping des extensions Yahoo Finance
    if exchange in ['NASDAQ', 'NYSE', 'AMEX', 'OTC']:
        return symbol
    elif any(ex in exchange for ex in ['PAR', 'PAE', 'EURONEXT', 'ENX']): # France/Benelux
        return f"{symbol}.PA"
    elif any(ex in exchange for ex in ['XETR', 'FWB', 'GER', 'BER']): # Germany
        return f"{symbol}.DE"
    elif 'MIL' in exchange: # Italy
        return f"{symbol}.MI"
    elif 'LSE' in exchange: # UK
        return f"{symbol}L"
    elif 'TSX' in exchange: # Canada
        return f"{symbol}.TO"
    elif 'TSXV' in exchange: # Canada Venture
        return f"{symbol}.V"
    elif any(ex in exchange for ex in ['TSE', 'TYO', 'JPX']): # Japan
        return f"{symbol}.T"
    elif row.get('market') == 'hongkong' or any(ex in exchange for ex in ['HK', 'SEHK', 'HKG', 'HKSE', 'SZSE', 'SHG']): # Hong Kong / China
        # Yahoo Finance HK : Toujours 4 chiffres minimum (ex: 5 -> 0005) + .HK
        if symbol.isdigit():
            return f"{symbol.zfill(4)}.HK"
        return f"{symbol}.HK"
    
    # Fallback pour les indices si nécessaire
    return symbol

def download_historical_data(df_tickers, period="2y", chunk_size=40):
    """
    Télécharge l'historique de 2 ans pour les tickers fournis via yfinance.
    Ajout de mécanismes de robustesse pour éviter les blocages Yahoo Finance.
    """
    import random
    
    logger.info(f"⏳ Étape 2 : Téléchargement de l'historique ({period}) pour {len(df_tickers)} tickers...")
    
    # Transformation des symboles pour YF
    df_tickers['yf_symbol'] = df_tickers.apply(map_tv_to_yf, axis=1)
    yf_symbols = df_tickers['yf_symbol'].unique().tolist()
    
    output_history = "scratch/history_2y_2B.parquet"
    temp_dir = "scratch/temp_download"
    os.makedirs(temp_dir, exist_ok=True)
    
    total_chunks = (len(yf_symbols) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(yf_symbols), chunk_size):
        chunk = yf_symbols[i:i + chunk_size]
        current_chunk = (i // chunk_size) + 1
        chunk_file = os.path.join(temp_dir, f"chunk_{current_chunk}.parquet")
        
        # Sauter si déjà téléchargé (permet de reprendre après un crash)
        if os.path.exists(chunk_file):
            logger.info(f"⏩ Chunk {current_chunk} déjà présent, on passe.")
            continue
            
        logger.info(f"📦 Téléchargement chunk {current_chunk}/{total_chunks} ({len(chunk)} tickers)...")
        logger.debug(f"🔍 Exemples de symboles envoyés : {chunk[:5]}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    tickers=chunk,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=False, # Désactivé pour le débogage
                    timeout=30
                )
                
                if not data.empty:
                    # Log de succès partiel
                    valid_tickers = [t for t in chunk if t in data.columns.levels[0]] if len(chunk) > 1 else ([chunk[0]] if not data.empty else [])
                    logger.info(f"✅ Reçu des données pour {len(valid_tickers)}/{len(chunk)} tickers.")
                    # Transformation en format long
                    if len(chunk) == 1:
                        df_temp = data.reset_index()
                        df_temp['Ticker'] = chunk[0]
                    else:
                        df_temp = data.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
                    
                    # Sauvegarde immédiate du chunk
                    df_temp.to_parquet(chunk_file, index=False)
                    break # Succès
                else:
                    logger.warning(f"⚠️ Chunk {current_chunk} vide (essai {attempt+1}/{max_retries})")
            
            except Exception as e:
                logger.error(f"❌ Erreur essai {attempt+1} sur le chunk {current_chunk} : {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5 + random.uniform(2, 5)
                    logger.info(f"⏲️ Attente de {wait_time:.1f}s avant nouvel essai...")
                    time.sleep(wait_time)
        
        # Délai entre les chunks pour éviter le bannissement IP
        time.sleep(random.uniform(1, 3))

    # Assemblage final
    logger.info("聚合 Étape 3 : Assemblage des fichiers temporaires...")
    chunk_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.parquet')]
    
    if chunk_files:
        all_dfs = [pd.read_parquet(f) for f in chunk_files]
        final_history = pd.concat(all_dfs, ignore_index=True)
        final_history.to_parquet(output_history, index=False)
        logger.success(f"✅ Historique complet sauvegardé ({len(final_history)} lignes) : {output_history}")
        
        # Nettoyage optionnel des fichiers temporaires
        # import shutil
        # shutil.rmtree(temp_dir)
    else:
        logger.warning("⚠️ Aucune donnée n'a été téléchargée.")

if __name__ == "__main__":
    # 1. Obtenir les tickers filtrés
    df_tickers = download_tv_tickers_with_filters()
    
    # 2. Télécharger l'historique
    if not df_tickers.empty:
        download_historical_data(df_tickers, period="2y")
