import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import warnings
from datetime import datetime
from loguru import logger

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.config_spark import Paths

warnings.filterwarnings('ignore')

class RegimeSwitchingMomentumBacktester:
    def __init__(self, start_date="2010-01-01", end_date=None, leverage=1.5):
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        self.leverage = leverage
        
        # Coûts fixes par défaut basés sur le notebook
        self.cash_yield_annual = 0.04
        self.margin_rate_annual = 0.06
        self.trading_fee_rate = 0.001

    def get_sp500_regime(self, spark_session) -> pd.DataFrame:
        logger.info(f"📈 Loading S&P 500 data from Silver Lake: {Paths.DATA_RAW_SP500_WEEKLY_SILVER}")
        try:
            sp500 = spark_session.read.format("delta").load(Paths.DATA_RAW_SP500_WEEKLY_SILVER).toPandas()
        except Exception as e:
            logger.error(f"🚨 Impossible de charger le SP500 depuis le Lake: {e}")
            return pd.DataFrame()
            
        if sp500.empty:
            logger.error("🚨 La table SP500 Silver est vide.")
            return pd.DataFrame()
            
        sp500 = sp500.rename(columns={'Date': 'date'}).set_index('date')
        sp500 = sp500.sort_index()
        sp500.index = pd.to_datetime(sp500.index)
        
        # Filtre sur les dates demandées
        sp500 = sp500.loc[self.start_date:self.end_date]
        
        if sp500.empty:
            logger.warning(f"⚠️ Aucune donnée SP500 entre {self.start_date} et {self.end_date}")
            return pd.DataFrame()
            
        # On ne garde que la colonne Close pour le calcul du régime
        sp500 = sp500[['Close']]
        
        sp500['SMA_26'] = ta.trend.sma_indicator(sp500['Close'], window=26)
        sp500['SMA_50'] = ta.trend.sma_indicator(sp500['Close'], window=50)
        
        cond_bull = ((sp500['SMA_26'] > sp500['SMA_50']) & (sp500['Close'] > sp500['SMA_50'])) | \
                    ((sp500['SMA_26'] < sp500['SMA_50']) & (sp500['Close'] > sp500['SMA_26']))
                    
        sp500['Regime'] = np.where(cond_bull, 'Bull', 'Bear')
        sp500.index = pd.to_datetime(sp500.index).tz_localize(None).normalize()
        
        return sp500[['Close', 'SMA_26', 'SMA_50', 'Regime']]

    def load_and_prep_data(self, spark_session, path_etf_gold, path_stocks_gold):
        try:
            logger.info(f"📡 Loading ETF Data from Gold: {path_etf_gold}")
            df_etf = spark_session.read.format("delta").load(path_etf_gold).toPandas()
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors du chargement de {path_etf_gold}: {e}")
            df_etf = pd.DataFrame()
            
        if not df_etf.empty:
            df_etf['Date'] = pd.to_datetime(df_etf['Date']).dt.normalize()
            # Règles d'éligibilité pour les ETFs fixées dans le notebook : (SMA_12 > SMA_26) AND (Price > SMA_26)
            df_etf['Eligible'] = (df_etf['SMA_12'] > df_etf['SMA_26']) & (df_etf['Close'] > df_etf['SMA_26'])
            df_etf = df_etf.rename(columns={'Date': 'date'})
            
        try:
            logger.info(f"📡 Loading Stock Data from Gold: {path_stocks_gold}")
            df_stocks = spark_session.read.format("delta").load(path_stocks_gold).toPandas()
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors du chargement de {path_stocks_gold}: {e}")
            df_stocks = pd.DataFrame()
        
        if not df_stocks.empty:
            df_stocks['Date'] = pd.to_datetime(df_stocks['Date']).dt.normalize()
            # Règles d'éligibilité strictes provenant du Notebook (Filtre Stop 50, Force 20, Volatilité)
            cond_trend = (df_stocks['SMA_26'] > df_stocks['SMA_50']) & (df_stocks['AdjClose'] > df_stocks['SMA_26'])
            cond_strength = df_stocks['ADX_20'] > 20
            cond_volatility = df_stocks['ATR_pct'] < 0.1
            cond_1W = df_stocks['Momentum_1W'] < 0.0
            
            df_stocks['Eligible'] = cond_trend & cond_strength & cond_volatility & cond_1W
            df_stocks = df_stocks.rename(columns={'Date': 'date', 'AdjClose': 'adjClose'})
            
        return df_etf, df_stocks

    def simulate_portfolio(self, sp500, etfs, stocks) -> pd.DataFrame:
        logger.info(f"⚙️ Lancement de la Simulation Vectorisée (Levier {self.leverage}x + Top 2 ETFs en Bear)...")
        
        dates = sp500.index
        if not etfs.empty: dates = dates.intersection(etfs['date'].unique())
        if not stocks.empty: dates = dates.intersection(stocks['date'].unique())
            
        dates = sorted(dates)
        if not dates: return pd.DataFrame()
        
        df_dates = pd.DataFrame({'date': dates})
        rebalance_dates_series = df_dates.groupby(df_dates['date'].dt.to_period('M'))['date'].max()
        rebalance_dates_str = set(date.strftime('%Y-%m-%d') for date in rebalance_dates_series)
        
        portfolio_allocations = {}
        current_portfolio = [] 
        trades_count = 0
        
        for d in dates:
            d_str = d.strftime('%Y-%m-%d')
            regime = sp500.loc[d, 'Regime']
            if isinstance(regime, pd.Series): regime = regime.iloc[0]
            
            # --- 1. FILTRE HEBDOMADAIRE (Stop-Loss Maintien) ---
            surviving_portfolio = []
            for pos in current_portfolio:
                ticker, ptype = pos['Ticker'], pos['Type']
                kept = True 
                
                if ptype == 'Stock':
                    asset_mask = (stocks['date'] == d) & (stocks['Ticker'] == ticker)
                    if asset_mask.any():
                        price = stocks[asset_mask].iloc[0]['adjClose']
                        ma_stop = stocks[asset_mask].iloc[0]['SMA_26'] 
                        if price < ma_stop: kept = False 
                else: 
                    asset_mask = (etfs['date'] == d) & (etfs['Ticker'] == ticker)
                    if asset_mask.any():
                        price = etfs[asset_mask].iloc[0]['Close']
                        ma_stop = etfs[asset_mask].iloc[0]['SMA_50'] 
                        if price < ma_stop: kept = False 
                        
                if kept: surviving_portfolio.append(pos)
                    
            current_portfolio = surviving_portfolio

            # --- 2. REBALANCEMENT MENSUEL ---
            if d_str in rebalance_dates_str:
                
                # 🟢 REGIME BULL : Achats Dynamiques Top 10 Actions
                if regime == 'Bull' and not stocks.empty:
                    current_portfolio = [p for p in current_portfolio if p['Type'] == 'Stock']
                    current_tickers = [p['Ticker'] for p in current_portfolio]
                    daily_stocks = stocks[stocks['date'] == d].copy()
                    
                    if not daily_stocks.empty:
                        daily_stocks['Rank'] = daily_stocks['Momentum_3M'].rank(ascending=False, method='first')
                        
                        kept_tickers = []
                        for ticker in current_tickers:
                            ticker_data = daily_stocks[daily_stocks['Ticker'] == ticker]
                            if not ticker_data.empty:
                                rank = ticker_data.iloc[0]['Rank']
                                if rank <= 15: # Maintien dans le Top 15 (Buffer)
                                    kept_tickers.append(ticker)
                                    
                        new_portfolio = [{'Ticker': t, 'Weight': 0.1, 'Type': 'Stock'} for t in kept_tickers]
                        places_libres = 10 - len(kept_tickers)
                        
                        if places_libres > 0:
                            eligible_stocks = daily_stocks[daily_stocks['Eligible']]
                            candidates = eligible_stocks[~eligible_stocks['Ticker'].isin(kept_tickers)]
                            top_new = candidates.nsmallest(places_libres, 'Rank') 
                            
                            for _, row in top_new.iterrows():
                                new_portfolio.append({'Ticker': row['Ticker'], 'Weight': 0.1, 'Type': 'Stock'})
                                trades_count += 1
                                
                        current_portfolio = new_portfolio
                        
                        # Fix Levied specific weights
                        n_assets = len(current_portfolio)
                        if n_assets > 0:
                            dynamic_weight = self.leverage / n_assets
                            for pos in current_portfolio: pos['Weight'] = dynamic_weight
                        
                # 🔴 REGIME BEAR : Répartition sur les 2 meilleurs ETFs (50/50)
                elif regime == 'Bear' and not etfs.empty:
                    current_portfolio = [p for p in current_portfolio if p['Type'] == 'ETF']
                    current_tickers = [p['Ticker'] for p in current_portfolio]
                    
                    # On cherche à avoir 2 ETFs maximum
                    places_libres = 2 - len(current_portfolio)
                    
                    if places_libres > 0:
                        daily_etfs = etfs[etfs['date'] == d].copy()
                        daily_etfs = daily_etfs[~daily_etfs['Ticker'].isin(current_tickers)]
                        eligible_etfs = daily_etfs[daily_etfs['Eligible']]
                        
                        # Sélection des nouveaux ETFs pour combler le vide
                        top_new = eligible_etfs.nlargest(places_libres, 'Momentum_3M')
                        for _, row in top_new.iterrows():
                            current_portfolio.append({'Ticker': row['Ticker'], 'Weight': 0.5, 'Type': 'ETF'})
                            trades_count += 1
                            
                    # ⚖️ Ajustement dynamique (ex: 1.0 / 2 = 50% chacun, SANS LEVIER)
                    n_assets = len(current_portfolio)
                    if n_assets > 0:
                        dynamic_weight = 1.0 / n_assets
                        for pos in current_portfolio: pos['Weight'] = dynamic_weight
            
            # --- 3. ENREGISTREMENT ---
            current_target = {pos['Ticker']: pos['Weight'] for pos in current_portfolio}
            portfolio_allocations[d] = current_target
            
        logger.info(f"✅ Backtest terminé. Total trades initiés : {trades_count}")
        return pd.DataFrame(portfolio_allocations).T.fillna(0)

    def generate_performance(self, allocations_df, etfs, stocks, sp500):
        logger.info("🧪 Calcul de l'Equity Curve et du rendement du Portefeuille vs SP500...")
        
        if allocations_df.empty or sp500.empty:
            logger.warning("⚠️ Impossible de générer les performances : Allocations ou S&P500 vides.")
            return pd.DataFrame()
            
        prices_etf = etfs.pivot(index='date', columns='Ticker', values='Close') if not etfs.empty else pd.DataFrame()
        prices_stocks = stocks.pivot(index='date', columns='Ticker', values='adjClose') if not stocks.empty else pd.DataFrame()
        all_prices = pd.concat([prices_etf, prices_stocks], axis=1)
        
        weekly_asset_returns = all_prices.pct_change()
        
        # L'allocation se fait à T et est subie de T à T+1. On décale d'une période pour croiser T avec Render(T+1)
        allocations_shifted = allocations_df.shift(1)
        common_dates = allocations_shifted.index.intersection(weekly_asset_returns.index)
        
        allocations_aligned = allocations_shifted.loc[common_dates]
        returns_aligned = weekly_asset_returns.loc[common_dates, allocations_aligned.columns].fillna(0)
        
        asset_return = (allocations_aligned * returns_aligned).sum(axis=1)
        
        total_invested = allocations_aligned.sum(axis=1)
        cash_weight = 1.0 - total_invested
        CASH_YIELD_WEEKLY = (1 + self.cash_yield_annual)**(1/52) - 1
        cash_return = cash_weight * CASH_YIELD_WEEKLY
        
        MARGIN_RATE = (1 + self.margin_rate_annual)**(1/52) - 1
        borrowed_weight = total_invested.clip(lower=1.0) - 1.0
        margin_cost = borrowed_weight * MARGIN_RATE
        
        weight_changes = allocations_shifted.diff().abs().sum(axis=1)
        transaction_costs = weight_changes * self.trading_fee_rate
        
        portfolio_return_weekly_net = asset_return + cash_return - margin_cost - transaction_costs
        
        equity_curve = (1 + portfolio_return_weekly_net).cumprod() * 100
        
        sp500_returns = sp500['Close'].pct_change().loc[common_dates]
        sp500_equity = (1 + sp500_returns).cumprod() * 100
        
        perf_df = pd.DataFrame({
            'Portfolio_Return': portfolio_return_weekly_net,
            'Portfolio_Equity': equity_curve,
            'SP500_Return': sp500_returns,
            'SP500_Equity': sp500_equity
        })
        
        # Le premier jour de la courbe doit démarrer propre à 100
        if not perf_df.empty:
            perf_df.iloc[0, perf_df.columns.get_loc('Portfolio_Equity')] = 100.0
            perf_df.iloc[0, perf_df.columns.get_loc('SP500_Equity')] = 100.0
        
        return perf_df
