import os
import sys
import optuna
import mlflow
import pandas as pd
import numpy as np
import gc
import time
from datetime import datetime
from loguru import logger

# Project imports
sys.path.append(os.getcwd())
from src.common.setup_spark import create_spark_session
from src.strategy.backtest_engine import RegimeSwitchingMomentumBacktester
from config.config_spark import Paths

from src.common.quality_manager import QualityManager

# --- CONFIGURATION MLFLOW ---
MLFLOW_TRACKING_URI = "http://momentum-mlflow-server:5000" if os.getenv("DOCKER_ENV") else "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Strategy_Optimization_Champion")

def optuna_callback(study, trial):
    if trial.number % 1 == 0:  # Log à chaque essai
        logger.info(f"🧪 Essai {trial.number}/{study.user_attrs.get('total_trials')} | Score: {trial.value:.4f} | Meilleur: {study.best_value:.4f}")

def run_optimization(n_trials=50):
    start_optim = time.time()
    logger.info(f"🎬 Démarrage de l'Optimisation de Stratégie ({n_trials} essais)")
    
    spark = create_spark_session('Strategy_Optimizer_Silver')
    
    try:
        # 1. Chargement et Validation des données
        logger.info("📥 Chargement des données Silver (S&P 500, ETFs, Index, 2B)...")
        
        # Indice et ETFs
        sdf_index = spark.read.format("delta").load(Paths.DATA_RAW_SP500_WEEKLY_SILVER)
        sdf_etf = spark.read.format("delta").load(Paths.DATA_RAW_ETF_WEEKLY_SILVER)
        
        # Fusion des Stocks (S&P 500 + Univers 2B)
        sdf_sp500_stocks = spark.read.format("delta").load(Paths.SP500_STOCK_PRICES_WEEKLY_SILVER)
        sdf_2b_stocks = spark.read.format("delta").load(Paths.DATA_RAW_2B_WEEKLY_SILVER)
        
        logger.info("🔗 Fusion des univers Stocks (S&P 500 + 2B)...")
        sdf_stocks = sdf_sp500_stocks.unionByName(sdf_2b_stocks, allowMissingColumns=True).dropDuplicates(['Ticker', 'Date'])
        
        # Validation Qualité GX avant conversion Pandas
        QualityManager.validate_silver_data(sdf_index, "Indice SP500")
        QualityManager.validate_silver_data(sdf_etf, "ETFs")
        QualityManager.validate_silver_data(sdf_stocks, "Mega-Universe Stocks")
        
        # Conversion Pandas
        sp500_raw = sdf_index.toPandas()
        df_etf = sdf_etf.toPandas()
        df_stocks = sdf_stocks.toPandas()

        # Statistiques
        for df, label in [(df_etf, "ETFs"), (df_stocks, "Stocks (Merged)")]:
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
                logger.info(f"📊 {label} : {len(df)} lignes, {df['Ticker'].nunique()} actifs, de {df['Date'].min().date()} à {df['Date'].max().date()}")

        if 'Date' in sp500_raw.columns:
            sp500_raw['Date'] = pd.to_datetime(sp500_raw['Date']).dt.normalize()
            sp500_raw = sp500_raw.set_index('Date').sort_index()
            
        sp500_raw = sp500_raw[~sp500_raw.index.duplicated(keep='last')]
        sp500_raw = sp500_raw[['Close']]
        
        # 2. Lancement Optuna avec Sampler Bayésien (TPE)
        sampler = optuna.samplers.TPESampler(n_startup_trials=10) # 10 trials aléatoires pour 'chauffer' le modèle bayésien
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        with mlflow.start_run(run_name=f"Opt_Silver_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            study.optimize(lambda trial: objective_silver(trial, sp500_raw, df_etf, df_stocks), n_trials=n_trials)
            
            logger.success(f"🏆 Meilleure stratégie trouvée : {study.best_value:.4f}")
            mlflow.log_params(study.best_params)
            mlflow.log_metric("calmar", study.best_value)

            # --- EXPORT POUR DBT ---
            import json
            config_path = os.path.join(os.getcwd(), "config/best_strategy_params.json")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(study.best_params, f, indent=4)
            logger.info(f"💾 Meilleurs paramètres exportés vers {config_path}")
            
    finally:
        spark.stop()

def objective_silver(trial, sp500_raw, df_etf_raw, df_stocks_raw):
    import ta
    
    config = {
        'sp500_sma_fast': trial.suggest_categorical('sp500_sma_fast', [7, 8, 9, 10, 12, 13, 14, 15, 20, 21, 25, 26]),
        'sp500_sma_slow': trial.suggest_categorical('sp500_sma_slow', [30, 35, 40, 45, 50, 55]),
        'stock_sma_fast': trial.suggest_categorical('stock_sma_fast', [7, 8, 9, 10, 12, 13, 14, 15, 20, 21, 25, 26, 30]),
        'stock_sma_slow': trial.suggest_categorical('stock_sma_slow', [30, 35, 40, 45, 50, 55]),
        'etf_sma_fast': trial.suggest_categorical('etf_sma_fast', [7, 8, 9, 10, 12, 13, 14, 15, 20, 21, 25, 26, 30]),
        'etf_sma_slow': trial.suggest_categorical('etf_sma_slow', [30, 35, 40, 45, 50, 55]),
        'stock_adx_threshold': trial.suggest_int('stock_adx_threshold', 10, 50, step=5),
        'stock_atr_threshold': trial.suggest_int('stock_atr_threshold', 10, 30, step=5),
        'stock_mom_period': trial.suggest_categorical('stock_mom_period', [4, 13, 26, 52]),
        'etf_mom_period': trial.suggest_categorical('etf_mom_period', [4, 13, 26, 52]),
        'top_n': trial.suggest_int('top_n', 5, 30, step=5),
        'rebalance_freq': trial.suggest_categorical('rebalance_freq', ['W', 'M', 'Q']),
        'buffer_n': trial.suggest_int('buffer_n', 5, 30, step=5),
        'leverage': 1.0, 'cash_yield': 0.04, 'margin_rate': 0.06, 'fees': 0.001
    }

    try:
        with mlflow.start_run(nested=True):
            # 1. Régime S&P 500
            sp500 = sp500_raw.copy()
            sp500['SMA_fast'] = ta.trend.sma_indicator(sp500['Close'], window=config['sp500_sma_fast'])
            sp500['SMA_slow'] = ta.trend.sma_indicator(sp500['Close'], window=config['sp500_sma_slow'])
            cond_bull = (sp500['SMA_fast'] > sp500['SMA_slow']) & (sp500['Close'] > sp500['SMA_slow'])
            sp500['Regime'] = np.where(cond_bull, 'Bull', 'Bear')

            # 2. ETFs (Données brutes)
            etfs = df_etf_raw[['Ticker', 'Date', 'Close']].copy().sort_values(['Ticker', 'Date'])

            # 3. Stocks (Données brutes)
            stocks = df_stocks_raw[['Ticker', 'Date', 'Close']].copy().sort_values(['Ticker', 'Date'])

            # 4. Simulation
            # Le moteur va lui-même calculer SMA, ADX, ATR et Eligible à chaque trial
            # en utilisant les paramètres suggérés par Optuna dans 'config'.
            engine = RegimeSwitchingMomentumBacktester(config=config, start_date="1980-01-01", leverage=config['leverage'])
            allocations = engine.simulate_portfolio(sp500, etfs, stocks)
            perf = engine.generate_performance(allocations, etfs, stocks, sp500)
            
            calmar = -1.0
            if not perf.empty:
                calmar = perf['Calmar_Ratio'].iloc[-1]
                mlflow.log_params(config)
                mlflow.log_metric("calmar", calmar)
                mlflow.log_metric("cagr", perf['CAGR'].iloc[-1])
                mlflow.log_metric("sharpe", perf['Sharpe_Ratio'].iloc[-1])
                mlflow.log_metric("max_drawdown", perf['Max_Drawdown'].iloc[-1])
                mlflow.log_metric("total_return", (perf['Portfolio_Equity'].iloc[-1]/100)-1)

            # NETTOYAGE MÉMOIRE
            del sp500, etfs, stocks, allocations, perf, engine
            gc.collect()
            
            return calmar if not np.isnan(calmar) else -1.0

    except Exception as e:
        import traceback
        logger.error(f"❌ Erreur Trial : {e}")
        logger.error(traceback.format_exc())
        gc.collect()
        return -1.0

if __name__ == "__main__":
    run_optimization(n_trials=5)
