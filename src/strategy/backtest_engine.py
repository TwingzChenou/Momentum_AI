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
    def __init__(self, config, start_date="2010-01-01", end_date=None, leverage=1.5):
        self.config = config
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        self.leverage = leverage
        
        # Coûts fixes par défaut basés sur la config MLFlow
        self.cash_yield_annual = self.config.get('cash_yield', 0.04)
        self.margin_rate_annual = self.config.get('margin_rate', 0.06)
        self.trading_fee_rate = self.config.get('fees', 0.001)

    def get_sp500_regime(self, spark_session) -> pd.DataFrame:
        logger.info(f"📈 Loading S&P 500 data from BigQuery: {Paths.BQ_SP500_GOLD}")
        try:
            sp500 = spark_session.read.format("bigquery") \
                .option("table", Paths.BQ_SP500_GOLD) \
                .load().toPandas()
        except Exception as e:
            logger.error(f"🚨 Impossible de charger le SP500 depuis BigQuery: {e}")
            return pd.DataFrame()
            
        if sp500.empty:
            return pd.DataFrame()
            
        sp500 = sp500.rename(columns={'Date': 'date'}).set_index('date')
        sp500 = sp500.sort_index()
        sp500.index = pd.to_datetime(sp500.index)
        sp500 = sp500.loc[self.start_date:self.end_date]
        
        if sp500.empty: return pd.DataFrame()
            
        # Priorité aux SMAs pré-calculées dans BigQuery
        if 'SMA_26' in sp500.columns:
            sp500['SMA_fast'] = sp500['SMA_26']
        else:
            sp500['SMA_fast'] = ta.trend.sma_indicator(sp500['Close'], window=self.config.get('sp500_sma_fast', 26))
            
        if 'SMA_50' in sp500.columns:
            sp500['SMA_slow'] = sp500['SMA_50']
        else:
            sp500['SMA_slow'] = ta.trend.sma_indicator(sp500['Close'], window=self.config.get('sp500_sma_slow', 50))
        
        cond_bull = ((sp500['SMA_fast'] > sp500['SMA_slow']) & (sp500['Close'] > sp500['SMA_slow'])) | \
                    ((sp500['SMA_fast'] < sp500['SMA_slow']) & (sp500['Close'] > sp500['SMA_fast']))
                    
        sp500['Regime'] = np.where(cond_bull, 'Bull', 'Bear')
        sp500.index = pd.to_datetime(sp500.index).tz_localize(None).normalize()
        
        return sp500[['Close', 'SMA_fast', 'SMA_slow', 'Regime']]

    def load_and_prep_data(self, spark_session, bq_etf_table, bq_stocks_table):
        try:
            df_etf = spark_session.read.format("bigquery").option("table", bq_etf_table).load().toPandas()
        except: df_etf = pd.DataFrame()
            
        if not df_etf.empty:
            df_etf['Date'] = pd.to_datetime(df_etf['Date']).dt.normalize()
            df_etf = df_etf.sort_values(['Ticker', 'Date'])
            df_etf['SMA_fast'] = df_etf.groupby('Ticker')['Close'].transform(lambda x: ta.trend.sma_indicator(x, window=self.config.get('etf_sma_fast', 26), fillna=True))
            df_etf['SMA_slow'] = df_etf.groupby('Ticker')['Close'].transform(lambda x: ta.trend.sma_indicator(x, window=self.config.get('etf_sma_slow', 50), fillna=True))
            df_etf['Momentum_XM'] = df_etf.groupby('Ticker')['Close'].transform(lambda x: x.pct_change(self.config.get('etf_mom_period', 13)))
            df_etf['Eligible'] = (df_etf['SMA_fast'] > df_etf['SMA_slow']) & (df_etf['Close'] > df_etf['SMA_slow'])
            df_etf = df_etf.rename(columns={'Date': 'date'})
            
        try:
            df_stocks = spark_session.read.format("bigquery").option("table", bq_stocks_table).load().toPandas()
        except: df_stocks = pd.DataFrame()
        
        if not df_stocks.empty:
            df_stocks['Date'] = pd.to_datetime(df_stocks['Date']).dt.normalize()
            df_stocks = df_stocks.sort_values(['Ticker', 'Date'])
            
            # --- RÉCUPÉRATION DES INDICATEURS PRÉ-CALCULÉS (GOLD) ---
            s_fast = self.config.get('stock_sma_fast', 26)
            s_slow = self.config.get('stock_sma_slow', 50)
            logger.info(f"⚙️ Paramètres Actions : SMA Fast={s_fast}, SMA Slow={s_slow}")
            
            # Si les colonnes correspondent exactement aux paramètres demandés, on les utilise
            # Sinon on tente le calcul local (si assez d'historique)
            if s_fast == 26 and 'SMA_26' in df_stocks.columns:
                df_stocks['SMA_fast'] = df_stocks['SMA_26']
            else:
                df_stocks['SMA_fast'] = df_stocks.groupby('Ticker')['AdjClose'].transform(lambda x: ta.trend.sma_indicator(x, window=s_fast, fillna=True))
                
            if s_slow == 50 and 'SMA_50' in df_stocks.columns:
                df_stocks['SMA_slow'] = df_stocks['SMA_50']
            else:
                df_stocks['SMA_slow'] = df_stocks.groupby('Ticker')['AdjClose'].transform(lambda x: ta.trend.sma_indicator(x, window=s_slow, fillna=True))
            
            df_stocks['Momentum_XM'] = df_stocks.groupby('Ticker')['AdjClose'].transform(lambda x: x.pct_change(self.config.get('stock_mom_period', 13)))
            
            # --- RÈGLES D'ÉLIGIBILITÉ ---
            cond_trend = (df_stocks['SMA_fast'] > df_stocks['SMA_slow']) & (df_stocks['AdjClose'] > df_stocks['SMA_slow'])
            
            cond_strength = True
            if 'ADX_20' in df_stocks.columns:
                cond_strength = df_stocks['ADX_20'] > self.config.get('stock_adx_threshold', 20.0)
                
            cond_volatility = True
            if 'ATR_pct' in df_stocks.columns:
                cond_volatility = df_stocks['ATR_pct'] < self.config.get('stock_atr_threshold', 0.15)
                
            df_stocks['Eligible'] = cond_trend & cond_strength & cond_volatility
            df_stocks = df_stocks.rename(columns={'Date': 'date', 'AdjClose': 'adjClose'})
            
        return df_etf, df_stocks

    def simulate_portfolio(self, sp500, etfs, stocks) -> pd.DataFrame:
        logger.info(f"⚙️ Lancement de la Simulation Vectorisée (Levier {self.leverage}x + Top 2 ETFs en Bear)...")
        
        # On utilise les dates de l'indice S&P 500 comme timeline de référence
        dates = sorted(sp500.index)
        if not dates: 
            logger.warning("🚨 Aucune donnée S&P 500 trouvée pour la timeline.")
            return pd.DataFrame()
        
        df_dates = pd.DataFrame({'date': dates})
        rebalance_dates_series = df_dates.groupby(df_dates['date'].dt.to_period(self.config.get('rebalance_freq', '1M')))['date'].max()
        rebalance_dates_str = set(date.strftime('%Y-%m-%d') for date in rebalance_dates_series)
        
        portfolio_allocations = {}
        current_portfolio = [] 
        trades_count = 0
        prev_regime = None
        
        # --- 0. AJUSTEMENT DE LA DATE DE DÉBUT (Warming up) ---
        s_fast = self.config.get('stock_sma_fast', 26)
        min_data_date = pd.to_datetime(stocks['date'].min())
        effective_start = min_data_date + pd.Timedelta(weeks=s_fast)
        
        sim_start_date = max(pd.to_datetime(self.start_date), effective_start)
        logger.info(f"📅 Début théorique : {self.start_date} | Début effectif (post-warmup {s_fast}w) : {sim_start_date.strftime('%Y-%m-%d')}")
        
        for d in dates:
            if d < sim_start_date: continue
            
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
                        ma_stop = stocks[asset_mask].iloc[0]['SMA_slow'] 
                        if price < ma_stop: kept = False 
                else: 
                    asset_mask = (etfs['date'] == d) & (etfs['Ticker'] == ticker)
                    if asset_mask.any():
                        price = etfs[asset_mask].iloc[0]['Close']
                        ma_stop = etfs[asset_mask].iloc[0]['SMA_slow'] 
                        if price < ma_stop: kept = False 
                        
                if kept: surviving_portfolio.append(pos)
            current_portfolio = surviving_portfolio

            # --- 2. REBALANCEMENT (Calendrier, Breakout ou Démarrage) ---
            # On rebalance si :
            # - C'est le premier jour de la simulation (pour ne pas attendre 3 mois)
            # - OU c'est la date prévue par le calendrier
            # - OU si on vient de passer de Bear à Bull
            is_regime_breakout = (prev_regime == 'Bear' and regime == 'Bull')
            is_first_day = (prev_regime is None)
            
            if d_str in rebalance_dates_str or is_regime_breakout or is_first_day:
                if is_regime_breakout:
                    logger.info(f"⚡ Régime Breakout détecté le {d_str} ! Rebalancement forcé.")
                if is_first_day:
                    logger.info(f"🚀 Initialisation du portefeuille le {d_str}.")
                
                if regime == 'Bull':
                    current_portfolio = [p for p in current_portfolio if p['Type'] == 'Stock']
                    current_tickers = [p['Ticker'] for p in current_portfolio]
                    daily_stocks = stocks[stocks['date'] == d].copy()
                    
                    if not daily_stocks.empty:
                        daily_stocks['Rank'] = daily_stocks['Momentum_XM'].rank(ascending=False, method='first')
                        
                        # LOG DIAGNOSTIC
                        eligible_stocks = daily_stocks[daily_stocks['Eligible']]
                        logger.info(f"🔍 Diagnostic {d_str} | Régime: {regime} | Total: {len(daily_stocks)}")
                        logger.info(f"   - Eligible: {len(eligible_stocks)} | ADX Thresh: {self.config.get('stock_adx_threshold')}")
                        
                        kept_tickers = []
                        for ticker in current_tickers:
                            ticker_data = daily_stocks[daily_stocks['Ticker'] == ticker]
                            if not ticker_data.empty:
                                rank = ticker_data.iloc[0]['Rank']
                                if rank <= self.config.get('buffer_n', 15):
                                    kept_tickers.append(ticker)
                                    
                        new_portfolio = [{'Ticker': t, 'Weight': 0.1, 'Type': 'Stock'} for t in kept_tickers]
                        places_libres = self.config.get('top_n', 10) - len(kept_tickers)
                        
                        if places_libres > 0:
                            candidates = eligible_stocks[~eligible_stocks['Ticker'].isin(kept_tickers)]
                            top_new = candidates.nsmallest(places_libres, 'Rank') 
                            for _, row in top_new.iterrows():
                                new_portfolio.append({'Ticker': row['Ticker'], 'Weight': 0.1, 'Type': 'Stock'})
                                trades_count += 1
                                
                        current_portfolio = new_portfolio
                        n_assets = len(current_portfolio)
                        if n_assets > 0:
                            dynamic_weight = self.leverage / n_assets
                            for pos in current_portfolio: pos['Weight'] = dynamic_weight
                        
                else:
                    current_portfolio = [p for p in current_portfolio if p['Type'] == 'ETF']
                    current_tickers = [p['Ticker'] for p in current_portfolio]
                    places_libres = 2 - len(current_portfolio)
                    if places_libres > 0:
                        daily_etfs = etfs[etfs['date'] == d].copy()
                        daily_etfs = daily_etfs[~daily_etfs['Ticker'].isin(current_tickers)]
                        eligible_etfs = daily_etfs[daily_etfs['Eligible']]
                        top_new = eligible_etfs.nlargest(places_libres, 'Momentum_XM')
                        for _, row in top_new.iterrows():
                            current_portfolio.append({'Ticker': row['Ticker'], 'Weight': 0.5, 'Type': 'ETF'})
                            trades_count += 1
                    n_assets = len(current_portfolio)
                    if n_assets > 0:
                        dynamic_weight = 1.0 / n_assets
                        for pos in current_portfolio: pos['Weight'] = dynamic_weight
            
            current_target = {pos['Ticker']: pos['Weight'] for pos in current_portfolio}
            portfolio_allocations[d] = current_target
            prev_regime = regime
            
        logger.info(f"✅ Backtest terminé. Total trades initiés : {trades_count}")
        return pd.DataFrame(portfolio_allocations).T.fillna(0)

    def generate_performance(self, allocations_df, etfs, stocks, sp500):
        if allocations_df.empty or sp500.empty: return pd.DataFrame()
            
        prices_etf = etfs.pivot(index='date', columns='Ticker', values='Close') if not etfs.empty else pd.DataFrame()
        prices_stocks = stocks.pivot(index='date', columns='Ticker', values='adjClose') if not stocks.empty else pd.DataFrame()
        all_prices = pd.concat([prices_etf, prices_stocks], axis=1)
        
        weekly_asset_returns = all_prices.pct_change()
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
        
        if not perf_df.empty:
            perf_df.iloc[0, perf_df.columns.get_loc('Portfolio_Equity')] = 100.0
            perf_df.iloc[0, perf_df.columns.get_loc('SP500_Equity')] = 100.0
        
        return perf_df
