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
        # Configuration de l'URI (Détection Docker vs Local)
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
        if os.path.exists("/.dockerenv"):
            final_uri = "http://momentum-mlflow-server:5000"
        else:
            final_uri = mlflow_uri
            
        mlflow.set_tracking_uri(final_uri)
        
        # 1. Identifier l'expérience
        exp = mlflow.get_experiment_by_name(experiment_name)
        
        if exp is None:
            logger.warning(f"⚠️ Expérience '{experiment_name}' introuvable dans MLFlow.")
            return get_default_config()

        # 2. Chercher le meilleur Run (celui avec le meilleur Calmar Ratio)
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=1,
            order_by=["metrics.calmar DESC"]
        )
        
        if runs.empty:
            logger.info("ℹ️ Aucun run trouvé dans MLflow pour cette expérience.")
            return get_default_config()
            
        # Extraction des paramètres du meilleur run
        best_run = runs.iloc[0]
        best_params = {k.replace("params.", ""): v for k, v in best_run.items() if k.startswith("params.")}
        
        logger.success(f"🏆 Meilleur Run chargé depuis MLFlow (Calmar: {best_run['metrics.calmar']:.2f})")
        
        # Mapping et conversion des types
        config = {
            'sp500_sma_slow': int(float(best_params.get('sp500_sma_slow', 50))),
            'sp500_sma_fast': int(float(best_params.get('sp500_sma_fast', 26))),
            'stock_sma_fast': int(float(best_params.get('stock_sma_fast', 26))),
            'stock_sma_slow': int(float(best_params.get('stock_sma_slow', 50))),
            'etf_sma_fast': int(float(best_params.get('etf_sma_fast', 12))),
            'etf_sma_slow': int(float(best_params.get('etf_sma_slow', 26))),
            'stock_atr_threshold': float(best_params.get('stock_atr_threshold', 0.15)),
            'stock_adx_threshold': float(best_params.get('stock_adx_threshold', 20.0)),
            'buffer_n': int(float(best_params.get('buffer_n', 15))),
            'top_n': int(float(best_params.get('top_n', 10))),
            'rebalance_freq': best_params.get('rebalance_freq', '1M'),
            'stock_mom_period': int(float(best_params.get('stock_mom_period', 13))),
            'etf_mom_period': int(float(best_params.get('etf_mom_period', 13))),
            'cash_yield': float(best_params.get('cash_yield', 0.04)),
            'margin_rate': float(best_params.get('margin_rate', 0.06)),
            'fees': float(best_params.get('fees', 0.001)),
            'use_pullback': best_params.get('use_pullback', 'False') == 'True',
            'use_cond_1W': best_params.get('use_cond_1W', 'False') == 'True'
        }
        return config

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
