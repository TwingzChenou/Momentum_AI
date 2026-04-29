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
        # 1. Chargement des données
        logger.info("📥 Chargement des données Silver depuis le Data Lake...")
        sp500_raw = spark.read.format("delta").load(Paths.DATA_RAW_SP500_WEEKLY_SILVER).toPandas()
        df_etf = spark.read.format("delta").load(Paths.DATA_RAW_ETF_WEEKLY_SILVER).toPandas()
        df_stocks = spark.read.format("delta").load(Paths.SP500_STOCK_PRICES_WEEKLY_SILVER).toPandas()

        # Statistiques
        for df, label in [(df_etf, "ETFs"), (df_stocks, "Stocks")]:
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
                logger.info(f"📊 {label} : {len(df)} lignes, {df['Ticker'].nunique()} actifs, de {df['Date'].min().date()} à {df['Date'].max().date()}")

        if 'Date' in sp500_raw.columns:
            sp500_raw['Date'] = pd.to_datetime(sp500_raw['Date']).dt.normalize()
            sp500_raw = sp500_raw.set_index('Date').sort_index()
            
        sp500_raw = sp500_raw[~sp500_raw.index.duplicated(keep='last')]
        sp500_raw = sp500_raw[['Close']]
        
        # 2. Données SILVER (Delta Lake)
        df_etf = spark.read.format("delta").load(Paths.DATA_RAW_ETF_WEEKLY_SILVER).toPandas()
        df_stocks = spark.read.format("delta").load(Paths.SP500_STOCK_PRICES_WEEKLY_SILVER).toPandas()

        # 3. Normalisation Robuste
        for df in [df_etf, df_stocks]:
            if df.empty: continue
            df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        
        # Lancement Optuna avec Sampler Bayésien (TPE)
        sampler = optuna.samplers.TPESampler(n_startup_trials=10) # 10 trials aléatoires pour 'chauffer' le modèle bayésien
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        with mlflow.start_run(run_name=f"Opt_Silver_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            study.optimize(lambda trial: objective_silver(trial, sp500_raw, df_etf, df_stocks), n_trials=n_trials)
            
            logger.success(f"🏆 Meilleure stratégie trouvée : {study.best_value:.4f}")
            mlflow.log_params(study.best_params)
            mlflow.log_metric("calmar", study.best_value)
            
    finally:
        spark.stop()

def objective_silver(trial, sp500_raw, df_etf_raw, df_stocks_raw):
    import ta
    
    config = {
        'sp500_sma_fast': trial.suggest_int('sp500_sma_fast', 5, 30),
        'sp500_sma_slow': trial.suggest_int('sp500_sma_slow', 35, 75),
        'stock_sma_fast': trial.suggest_int('stock_sma_fast', 5, 30),
        'stock_sma_slow': trial.suggest_int('stock_sma_slow', 35, 75),
        'etf_sma_fast': trial.suggest_int('etf_sma_fast', 5, 30),
        'etf_sma_slow': trial.suggest_int('etf_sma_slow', 35, 75),
        'stock_adx_threshold': trial.suggest_float('stock_adx_threshold', 10.0, 35.0),
        'stock_atr_threshold': trial.suggest_float('stock_atr_threshold', 0.10, 0.50),
        'stock_mom_period': trial.suggest_int('stock_mom_period', 5, 13),
        'etf_mom_period': trial.suggest_int('etf_mom_period', 4, 13),
        'top_n': trial.suggest_int('top_n', 5, 30),
        'rebalance_freq': trial.suggest_categorical('rebalance_freq', ['W', 'M', 'Q']),
        'buffer_n': 15, 'leverage': 1.0, 'cash_yield': 0.04, 'margin_rate': 0.06, 'fees': 0.001
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
            engine = RegimeSwitchingMomentumBacktester(config=config, start_date="1998-01-01", leverage=config['leverage'])
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
