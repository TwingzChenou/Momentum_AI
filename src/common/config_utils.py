import os
import mlflow
import pandas as pd
from loguru import logger

def get_champion_config(experiment_name="Strategy_Optimization_Champion"):
    """
    Récupère la configuration du 'Champion' depuis MLFlow.
    Si MLFlow est inaccessible ou vide, retourne une configuration par défaut.
    """
    try:
        # Configuration de l'URI (Adaptation automatique Docker vs Local)
        # host.docker.internal permet au container Streamlit de joindre le MLflow du Mac
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
        mlflow.set_tracking_uri(mlflow_uri)
        
        # 1. Accès aux expériences
        try:
            exps = mlflow.search_experiments()
            exp_ids = [e.experiment_id for e in exps]
        except Exception as e:
            logger.error(f"❌ Erreur de connexion MLflow ({mlflow_uri}) : {e}")
            return get_default_config()

        # 2. Recherche du dernier bilan d'optimisation ('Opt_...')
        try:
            runs = mlflow.search_runs(
                experiment_ids=exp_ids,
                filter_string="attributes.run_name LIKE 'Opt_%' OR tags.mlflow.runName LIKE 'Opt_%'",
                max_results=100,
                order_by=["start_time DESC"]
            )
        except:
            runs = mlflow.search_runs(
                experiment_ids=exp_ids,
                filter_string="tags.mlflow.runName LIKE 'Opt_%'",
                max_results=100,
                order_by=["start_time DESC"]
            )
        
        if not runs.empty:
            best_run = runs.iloc[0]
            logger.success("🏆 Configuration Champion chargée avec succès.")
        else:
            # Fallback sur le meilleur run absolu si aucun 'Opt_' n'est trouvé
            runs = mlflow.search_runs(experiment_ids=exp_ids, max_results=100, order_by=["metrics.calmar DESC"])
            if runs.empty: return get_default_config()
            best_run = runs.iloc[0]
        
        best_params = {k.replace("params.", ""): v for k, v in best_run.items() if k.startswith("params.")}
        
        # 4. Extraction sécurisée des paramètres
        def safe_get(key, default):
            val = best_params.get(key)
            return val if val is not None else default

        try:
            config = {
                'sp500_sma_slow': int(float(safe_get('sp500_sma_slow', 50))),
                'sp500_sma_fast': int(float(safe_get('sp500_sma_fast', 26))),
                'stock_sma_fast': int(float(safe_get('stock_sma_fast', 26))),
                'stock_sma_slow': int(float(safe_get('stock_sma_slow', 50))),
                'etf_sma_fast': int(float(safe_get('etf_sma_fast', 12))),
                'etf_sma_slow': int(float(safe_get('etf_sma_slow', 26))),
                'stock_atr_threshold': float(safe_get('stock_atr_threshold', 0.15)),
                'stock_adx_threshold': float(safe_get('stock_adx_threshold', 20.0)),
                'buffer_n': int(float(safe_get('buffer_n', 15))),
                'top_n': int(float(safe_get('top_n', 10))),
                'rebalance_freq': str(safe_get('rebalance_freq', '1M')),
                'stock_mom_period': int(float(safe_get('stock_mom_period', 13))),
                'etf_mom_period': int(float(safe_get('etf_mom_period', 13))),
                'cash_yield': float(safe_get('cash_yield', 0.04)),
                'margin_rate': float(safe_get('margin_rate', 0.06)),
                'fees': float(safe_get('fees', 0.001)),
                'use_pullback': str(safe_get('use_pullback', 'False')).lower() == 'true',
                'use_cond_1W': str(safe_get('use_cond_1W', 'False')).lower() == 'true',
                'run_name': best_run.get('tags.mlflow.runName', 'Champion'),
                'calmar': float(best_run.get('metrics.calmar', 0.0)),
                'mlflow_uri': mlflow_uri
            }
            return config
        except Exception as e:
            logger.error(f"❌ Erreur lors du mapping des paramètres : {e}")
            return get_default_config()

    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération de la config MLFlow : {e}")
        return get_default_config()

def get_default_config():
    """Paramètres de secours si MLflow est vide ou inaccessible"""
    return {
        'sp500_sma_slow': 50, 'sp500_sma_fast': 26, 'stock_sma_fast': 26, 'stock_sma_slow': 50,
        'etf_sma_fast': 12, 'etf_sma_slow': 26, 'stock_atr_threshold': 0.15, 'stock_adx_threshold': 20.0,
        'use_pullback': False, 'use_cond_1W': False, 'buffer_n': 15, 'top_n': 10, 'rebalance_freq': '1M',
        'stock_mom_period': 13, 'etf_mom_period': 13, 'cash_yield': 0.04, 'margin_rate': 0.06, 'fees': 0.001
    }
