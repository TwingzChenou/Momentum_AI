import os
import sys
import optuna
import mlflow
import pandas as pd
from datetime import datetime
from loguru import logger

# Project imports
sys.path.append(os.getcwd())
from src.common.setup_spark import create_spark_session
from src.strategy.backtest_engine import RegimeSwitchingMomentumBacktester
from config.config_spark import Paths

# --- CONFIGURATION MLFLOW ---
# Utilisation de l'adresse interne Docker si on tourne dans Airflow, sinon localhost
MLFLOW_TRACKING_URI = "http://momentum-mlflow-server:5000" if os.getenv("DOCKER_ENV") else "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Momentum_Strategy_Optimization")

def objective(trial, spark, df_sp500, df_etf, df_stocks):
    """
    Fonction objectif pour Optuna : définit les paramètres à tester
    """
    # 1. Définition de l'espace de recherche (Hyperparamètres)
    config = {
        # Paramètres S&P 500
        'sp500_sma_fast': trial.suggest_int('sp500_sma_fast', 5, 30),
        'sp500_sma_slow': trial.suggest_int('sp500_sma_slow', 35, 60),
        
        # Paramètres Actions (Basés sur l'historique dispo)
        'stock_sma_fast': trial.suggest_int('stock_sma_fast', 5, 15), # On réduit car historique court
        'stock_sma_slow': trial.suggest_int('stock_sma_slow', 20, 40),
        
        # Paramètres ETF
        'etf_sma_fast': trial.suggest_int('etf_sma_fast', 10, 30),
        'etf_sma_slow': trial.suggest_int('etf_sma_slow', 40, 60),
        
        # Filtres Qualité
        'stock_adx_threshold': trial.suggest_float('stock_adx_threshold', 10.0, 30.0),
        'stock_atr_threshold': trial.suggest_float('stock_atr_threshold', 0.10, 0.40),
        'stock_mom_period': trial.suggest_int('stock_mom_period', 10, 20),
        
        # Gestion Portefeuille
        'top_n': trial.suggest_int('top_n', 5, 20),
        'rebalance_freq': trial.suggest_categorical('rebalance_freq', ['1M', 'Q']),
        
        # Coûts (Fixes)
        'cash_yield': 0.04,
        'margin_rate': 0.06,
        'fees': 0.001
    }

    with mlflow.start_run(nested=True):
        # 2. Lancement du Backtest
        # On teste sur une période robuste
        engine = RegimeSwitchingMomentumBacktester(config, start_date="2025-01-01")
        
        # On utilise les données déjà chargées pour la vitesse
        allocations = engine.simulate_portfolio(df_sp500, df_etf, df_stocks)
        perf = engine.generate_performance(allocations, df_etf, df_stocks, df_sp500)
        
        if perf.empty:
            return -100.0 # Pénalité si aucune donnée

        # 3. Calcul des Métriques
        total_return = (perf['Portfolio_Equity'].iloc[-1] / 100) - 1
        max_drawdown = perf['Drawdown'].min()
        
        # Calmar Ratio (Annualized Return / Max Drawdown)
        # On simplifie pour l'optimisation
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 4. Logging MLflow
        mlflow.log_params(config)
        mlflow.log_metric("total_return", total_return)
        mlflow.log_metric("max_drawdown", max_drawdown)
        mlflow.log_metric("calmar_ratio", calmar)
        
        return calmar

def run_optimization(n_trials=50):
    logger.info("🚀 Démarrage de l'optimisation sur SP500_STOCK_PRICES (Historique Long)...")
    spark = create_spark_session('Strategy_Optimizer_Full')
    
    # Table avec beaucoup d'historique pour les Large Caps
    BQ_STOCKS_FULL = "finance-ml-project-486410.Dataset_Strategy_Momentum.SP500_STOCK_PRICES"
    
    try:
        # On teste sur une période plus large pour la robustesse (ex: 2023 à 2025)
        engine_base = RegimeSwitchingMomentumBacktester({}, start_date="2023-01-01")
        df_sp500 = engine_base.get_sp500_regime(spark)
        
        logger.info(f"📥 Chargement de la table complète : {BQ_STOCKS_FULL}")
        df_etf, df_stocks = engine_base.load_and_prep_data(spark, Paths.BQ_ETF_GOLD, BQ_STOCKS_FULL)
        
        # Création de l'étude Optuna
        study = optuna.create_study(direction='maximize')
        
        # Lancement de l'optimisation
        with mlflow.start_run(run_name=f"Optimization_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            study.optimize(lambda trial: objective(trial, spark, df_sp500, df_etf, df_stocks), n_trials=n_trials)
            
            # Sauvegarde du meilleur run
            logger.success(f"🏆 Meilleure stratégie trouvée : {study.best_value:.4f}")
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_calmar", study.best_value)
            
    finally:
        spark.stop()

if __name__ == "__main__":
    run_optimization(n_trials=30)
