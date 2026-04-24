import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

# Setup
sys.path.append(os.getcwd())
from src.common.setup_spark import create_spark_session
from src.strategy.backtest_engine import RegimeSwitchingMomentumBacktester
from config.config_spark import Paths

def run_diagnostic():
    spark = create_spark_session('Diagnostic_Graphique')
    
    # Paramètres identiques au Dashboard (ceux vus dans les logs)
    config = {
        'sp500_sma_fast': 10, 'sp500_sma_slow': 45,
        'stock_sma_fast': 25, 'stock_sma_slow': 70,
        'etf_sma_fast': 26, 'etf_sma_slow': 50,
        'stock_mom_period': 13, 'etf_mom_period': 13,
        'stock_adx_threshold': 11.0, 'stock_atr_threshold': 0.35,
        'buffer_n': 15, 'top_n': 10, 'rebalance_freq': 'Q',
        'cash_yield': 0.04, 'margin_rate': 0.06, 'fees': 0.001
    }

    try:
        engine = RegimeSwitchingMomentumBacktester(config, start_date='2025-01-01', end_date='2026-04-20')
        
        df_sp500 = engine.get_sp500_regime(spark)
        df_etf, df_stocks = engine.load_and_prep_data(spark, Paths.BQ_ETF_GOLD, Paths.BQ_STOCKS_GOLD)
        
        allocations = engine.simulate_portfolio(df_sp500, df_etf, df_stocks)
        perf = engine.generate_performance(allocations, df_etf, df_stocks, df_sp500)
        
        if perf.empty:
            print("🚨 Erreur : Performance vide.")
            return

        # Création du graphique
        plt.figure(figsize=(12, 6))
        plt.plot(perf.index, perf['Portfolio_Equity'], label='Stratégie Momentum (Réactive)', color='blue', linewidth=2)
        plt.plot(perf.index, perf['SP500_Equity'], label='S&P 500', color='red', linestyle='--', alpha=0.7)
        
        plt.title(f"Diagnostic : Stratégie vs S&P 500\n(SMA Actions {config['stock_sma_fast']}/{config['stock_sma_slow']})")
        plt.xlabel("Date")
        plt.ylabel("Valeur du Portefeuille (Base 100)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = 'diagnostic_performance.png'
        plt.savefig(output_path)
        print(f"✅ Graphique généré : {output_path}")
        
        # On affiche aussi les périodes d'achats d'actions
        stock_cols = [c for c in allocations.columns if c not in ['Date', 'Regime']]
        has_stocks = allocations[stock_cols].sum(axis=1) > 0
        print("\n--- PÉRIODES D'ACHATS D'ACTIONS DÉTECTÉES ---")
        print(has_stocks[has_stocks].index)

    finally:
        spark.stop()

if __name__ == "__main__":
    run_diagnostic()
