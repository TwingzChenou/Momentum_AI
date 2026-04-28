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

    def get_sp500_regime_from_df(self, df_sp500: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le régime sur un DataFrame S&P 500 déjà chargé.
        """
        import ta
        sp500 = df_sp500.copy()
        
        # Normalisation si index non temporel
        if not isinstance(sp500.index, pd.DatetimeIndex):
             if 'Date' in sp500.columns:
                 sp500['Date'] = pd.to_datetime(sp500['Date'])
                 sp500 = sp500.set_index('Date')
        
        sp500 = sp500.sort_index()
        
        # On utilise les colonnes déjà présentes si possible (Gold)
        if 'Regime' in sp500.columns and 'SMA_fast' in sp500.columns and 'SMA_slow' in sp500.columns:
            logger.info("✅ Utilisation du Régime S&P 500 déjà présent dans le DataFrame.")
        else:
            # Calcul local de secours
            logger.info(f"⚙️ Calcul local du Régime S&P 500 ({self.config.get('sp500_sma_fast', 26)}/{self.config.get('sp500_sma_slow', 50)})")
            sp500['SMA_fast'] = ta.trend.sma_indicator(sp500['Close'], window=self.config.get('sp500_sma_fast', 26))
            sp500['SMA_slow'] = ta.trend.sma_indicator(sp500['Close'], window=self.config.get('sp500_sma_slow', 50))
            
            cond_bull = ((sp500['SMA_fast'] > sp500['SMA_slow']) & (sp500['Close'] > sp500['SMA_slow'])) | \
                        ((sp500['SMA_fast'] < sp500['SMA_slow']) & (sp500['Close'] > sp500['SMA_fast']))
                        
            sp500['Regime'] = np.where(cond_bull, 'Bull', 'Bear')
        
        return sp500[['Close', 'SMA_fast', 'SMA_slow', 'Regime']]

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
            
        sp500 = sp500.set_index('Date')
        sp500 = sp500.sort_index()
        sp500.index = pd.to_datetime(sp500.index)
        sp500 = sp500.loc[self.start_date:self.end_date]
        
        if sp500.empty: return pd.DataFrame()
            
        # On utilise les colonnes déjà présentes si possible (Gold)
        if 'Regime' in sp500.columns and 'SMA_fast' in sp500.columns and 'SMA_slow' in sp500.columns:
            logger.info("✅ Utilisation des indicateurs S&P 500 déjà présents dans BigQuery.")
        else:
            # Calcul local de secours
            logger.info(f"⚙️ Calcul des indicateurs S&P 500 locaux ({self.config.get('sp500_sma_fast', 26)}/{self.config.get('sp500_sma_slow', 50)})")
            sp500['SMA_fast'] = ta.trend.sma_indicator(sp500['Close'], window=self.config.get('sp500_sma_fast', 26))
            sp500['SMA_slow'] = ta.trend.sma_indicator(sp500['Close'], window=self.config.get('sp500_sma_slow', 50))
            
            cond_bull = ((sp500['SMA_fast'] > sp500['SMA_slow']) & (sp500['Close'] > sp500['SMA_slow'])) | \
                        ((sp500['SMA_fast'] < sp500['SMA_slow']) & (sp500['Close'] > sp500['SMA_fast']))
                        
            sp500['Regime'] = np.where(cond_bull, 'Bull', 'Bear')
        
        sp500.index = pd.to_datetime(sp500.index).tz_localize(None).normalize()
        return sp500[['Close', 'SMA_fast', 'SMA_slow', 'Regime']]

    def load_and_prep_data(self, spark_session, bq_etf_table, bq_stocks_table):
        # 1. LOAD ETFS
        try:
            df_etf = spark_session.read.format("bigquery").option("table", bq_etf_table).load().toPandas()
        except: df_etf = pd.DataFrame()
            
        if not df_etf.empty:
            df_etf['Date'] = pd.to_datetime(df_etf['Date']).dt.normalize()
            df_etf = df_etf.sort_values(['Ticker', 'Date'])
            
            # Priorité aux colonnes BigQuery
            if 'SMA_fast' in df_etf.columns and 'SMA_slow' in df_etf.columns and 'Momentum_XM' in df_etf.columns:
                logger.info("✨ Utilisation des indicateurs pré-calculés pour les ETFs")
            else:
                logger.info("⚙️ Calcul local des indicateurs pour les ETFs")
                df_etf['SMA_fast'] = df_etf.groupby('Ticker')['Close'].transform(lambda x: ta.trend.sma_indicator(x, window=self.config.get('etf_sma_fast', 26), fillna=True))
                df_etf['SMA_slow'] = df_etf.groupby('Ticker')['Close'].transform(lambda x: ta.trend.sma_indicator(x, window=self.config.get('etf_sma_slow', 50), fillna=True))
                df_etf['Momentum_XM'] = df_etf.groupby('Ticker')['Close'].transform(lambda x: x.pct_change(self.config.get('etf_mom_period', 13)))
            
            df_etf['Eligible'] = (df_etf['SMA_fast'] > df_etf['SMA_slow']) & (df_etf['Close'] > df_etf['SMA_slow'])
            
            
        # 2. LOAD STOCKS
        try:
            df_stocks = spark_session.read.format("bigquery").option("table", bq_stocks_table).load().toPandas()
        except: df_stocks = pd.DataFrame()
        
        if not df_stocks.empty:
            df_stocks['Date'] = pd.to_datetime(df_stocks['Date']).dt.normalize()
            df_stocks = df_stocks.sort_values(['Ticker', 'Date'])
            
            # Priorité aux colonnes BigQuery
            if 'SMA_fast' in df_stocks.columns and 'SMA_slow' in df_stocks.columns and 'Momentum_XM' in df_stocks.columns:
                logger.info("✨ Utilisation des indicateurs pré-calculés pour les Actions")
            else:
                logger.info("⚙️ Calcul local des indicateurs pour les Actions")
                df_stocks['SMA_fast'] = df_stocks.groupby('Ticker')['Close'].transform(lambda x: ta.trend.sma_indicator(x, window=self.config.get('stock_sma_fast', 26), fillna=True))
                df_stocks['SMA_slow'] = df_stocks.groupby('Ticker')['Close'].transform(lambda x: ta.trend.sma_indicator(x, window=self.config.get('stock_sma_slow', 50), fillna=True))
                df_stocks['Momentum_XM'] = df_stocks.groupby('Ticker')['Close'].transform(lambda x: x.pct_change(self.config.get('stock_mom_period', 13)))
            
            # --- RÈGLES D'ÉLIGIBILITÉ ---
            cond_trend = (df_stocks['SMA_fast'] > df_stocks['SMA_slow']) & (df_stocks['Close'] > df_stocks['SMA_slow'])
            
            cond_strength = True
            if 'ADX_20' in df_stocks.columns:
                cond_strength = df_stocks['ADX_20'] > self.config.get('stock_adx_threshold', 20.0)
                
            cond_volatility = True
            if 'ATR_pct' in df_stocks.columns:
                cond_volatility = df_stocks['ATR_pct'] < self.config.get('stock_atr_threshold', 0.15)
                
            df_stocks['Eligible'] = cond_trend & cond_strength & cond_volatility
            
        return df_etf, df_stocks

    def load_and_prep_data_silver(self, df_etf, df_stocks):
        """
        Prépare les données ETF et Stocks en utilisant les indicateurs GOLD si présents.
        """
        import ta
        
        # 1. ETFs
        if not df_etf.empty:
            df_etf['Date'] = pd.to_datetime(df_etf['Date']).dt.normalize()
            df_etf = df_etf.sort_values(['Ticker', 'Date'])
            
            # On ne recalcule que si les colonnes manquent
            if 'SMA_fast' not in df_etf.columns or 'Momentum_XM' not in df_etf.columns:
                logger.info("⚙️ Calcul local des indicateurs ETFs...")
                df_etf['SMA_fast'] = df_etf.groupby('Ticker')['Close'].transform(lambda x: ta.trend.sma_indicator(x, window=self.config.get('etf_sma_fast', 26), fillna=True))
                df_etf['SMA_slow'] = df_etf.groupby('Ticker')['Close'].transform(lambda x: ta.trend.sma_indicator(x, window=self.config.get('etf_sma_slow', 50), fillna=True))
                df_etf['Momentum_XM'] = df_etf.groupby('Ticker')['Close'].transform(lambda x: x.pct_change(self.config.get('etf_mom_period', 13)))
                
            df_etf['Eligible'] = (df_etf['SMA_fast'] > df_etf['SMA_slow']) & (df_etf['Close'] > df_etf['SMA_slow'])

        # 2. STOCKS
        if not df_stocks.empty:
            df_stocks['Date'] = pd.to_datetime(df_stocks['Date']).dt.normalize()
            df_stocks = df_stocks.sort_values(['Ticker', 'Date'])
            
            # On ne recalcule que si les colonnes manquent
            if 'SMA_fast' not in df_stocks.columns or 'Momentum_XM' not in df_stocks.columns:
                logger.info("⚙️ Calcul local des indicateurs Stocks...")
                df_stocks['SMA_fast'] = df_stocks.groupby('Ticker')['Close'].transform(lambda x: ta.trend.sma_indicator(x, window=self.config.get('stock_sma_fast', 26), fillna=True))
                df_stocks['SMA_slow'] = df_stocks.groupby('Ticker')['Close'].transform(lambda x: ta.trend.sma_indicator(x, window=self.config.get('stock_sma_slow', 50), fillna=True))
                df_stocks['Momentum_XM'] = df_stocks.groupby('Ticker')['Close'].transform(lambda x: x.pct_change(self.config.get('stock_mom_period', 13)))
            
            # Filtres techniques (On utilise ADX et ATR de BigQuery si possible)
            cond_trend = (df_stocks['SMA_fast'] > df_stocks['SMA_slow']) & (df_stocks['Close'] > df_stocks['SMA_slow'])
            
            cond_strength = True
            if 'ADX' in df_stocks.columns:
                cond_strength = df_stocks['ADX'] > self.config.get('stock_adx_threshold', 20.0)
            
            cond_volatility = True
            if 'ATR' in df_stocks.columns:
                cond_volatility = (df_stocks['ATR'] / df_stocks['Close']) < self.config.get('stock_atr_threshold', 0.15)
                
            df_stocks['Eligible'] = cond_trend & cond_strength & cond_volatility
            
        return df_etf, df_stocks

    def simulate_portfolio(self, sp500, etfs, stocks) -> pd.DataFrame:
        logger.info(f"⚙️ Lancement de la Simulation (Levier {self.leverage}x)...")
        
        # --- PRÉPARATION INTERNE ---
        # Si les données ne sont pas préparées (ex: chargement direct BigQuery), on le fait ici
        if not etfs.empty and 'Eligible' not in etfs.columns:
            etfs, stocks = self.load_and_prep_data_silver(etfs, stocks)
            
        # --- ALIGNEMENT CALENDRIER ---
        # On force le S&P 500 en hebdomadaire (Vendredi) pour matcher les actions/ETFs
        sp500 = sp500.resample('W-FRI').last().ffill()
        dates = sorted(sp500.index)
        
        if not dates: return pd.DataFrame()
        
        # 0. PRE-CALCULS
        rebalance_dates = set(pd.DataFrame({'Date': dates}).groupby(pd.to_datetime(dates).to_period(self.config.get('rebalance_freq', '1M')))['Date'].max().dt.normalize())
        
        # Dictionnaires pour accès rapide O(1)
        logger.info("📦 Indexation des données en mémoire...")
        stocks_indexed = stocks.set_index(['Date', 'Ticker'])
        s_data = stocks_indexed[['Close', 'SMA_slow', 'Momentum_XM', 'Eligible']].to_dict('index')
        del stocks_indexed
        
        etfs_indexed = etfs.set_index(['Date', 'Ticker'])
        e_data = etfs_indexed[['Close', 'SMA_slow', 'Eligible']].to_dict('index')
        del etfs_indexed
        
        # Timeline
        portfolio_allocations = {}
        current_portfolio = [] 
        prev_regime = None
        
        s_fast = self.config.get('stock_sma_fast', 26)
        min_date = pd.to_datetime(stocks['Date'].min()) if not stocks.empty else pd.to_datetime(dates[0])
        sim_start_date = max(pd.to_datetime(self.start_date), min_date + pd.Timedelta(weeks=s_fast))

        for d in dates:
            if d < sim_start_date: continue
            
            regime = sp500.loc[d, 'Regime']
            if isinstance(regime, pd.Series): regime = regime.iloc[0]
            
            # --- 1. MAINTENANCE QUOTIDIENNE (Stop-Loss) ---
            surviving = []
            for pos in current_portfolio:
                t, ptype = pos['Ticker'], pos['Type']
                if ptype == 'Stock':
                    row = s_data.get((d, t), {})
                    price = row.get('Close')
                    sma = row.get('SMA_slow')
                    if price and sma and price > sma: 
                        surviving.append(pos)
                    else:
                        logger.debug(f"📉 {d.date()} | Sortie Stop-Loss: {t} (Price {price} <= SMA {sma})")
                else:
                    row = e_data.get((d, t), {})
                    price = row.get('Close')
                    sma = row.get('SMA_slow')
                    if price and sma and price > sma: surviving.append(pos)
            current_portfolio = surviving

            # --- 2. REBALANCEMENT ---
            is_breakout = (prev_regime == 'Bear' and regime == 'Bull')
            if d in rebalance_dates or is_breakout or prev_regime is None:
                logger.info(f"🔄 {d.date()} | Rebalancement ({regime}) | Portfolio actuel: {len(current_portfolio)} actifs")
                
                if regime == 'Bull':
                    # On garde les stocks actuels qui sont encore bons
                    current_portfolio = [p for p in current_portfolio if p['Type'] == 'Stock']
                    cur_tickers = [p['Ticker'] for p in current_portfolio]
                    
                    # Sélection des nouveaux
                    day_stocks = stocks[stocks['Date'] == d].copy()
                    if day_stocks.empty:
                        logger.warning(f"⚠️ {d.date()} | Aucune donnée Action disponible ce jour.")
                    else:
                        # Diagnostic d'éligibilité
                        n_total = len(day_stocks)
                        n_eligible = day_stocks['Eligible'].sum()
                        logger.info(f"🔎 {d.date()} | Éligibilité Stocks: {n_eligible} / {n_total}")
                        
                        day_stocks['Rank'] = day_stocks['Momentum_XM'].rank(ascending=False)
                        # On garde ceux qui sont déjà en portefeuille s'ils sont dans le top buffer
                        kept = []
                        for p in current_portfolio:
                            row = day_stocks[day_stocks['Ticker'] == p['Ticker']]
                            if not row.empty and row.iloc[0]['Rank'] <= self.config.get('buffer_n', 15):
                                kept.append(p)
                            else:
                                logger.debug(f"♻️ {d.date()} | {p['Ticker']} sorti (Rank > Buffer)")
                        
                        # On complète jusqu'à top_n
                        needed = self.config.get('top_n', 10) - len(kept)
                        logger.info(f"🎯 {d.date()} | Besoin de {needed} nouveaux stocks (Déjà {len(kept)} gardés)")
                        
                        if needed > 0:
                            candidates = day_stocks[day_stocks['Eligible'] & (~day_stocks['Ticker'].isin([p['Ticker'] for p in kept]))]
                            logger.info(f"💡 {d.date()} | Candidats éligibles et nouveaux: {len(candidates)}")
                            
                            top_new = candidates.nlargest(needed, 'Momentum_XM')
                            for _, row in top_new.iterrows():
                                kept.append({'Ticker': row['Ticker'], 'Weight': 0, 'Type': 'Stock'})
                        
                        current_portfolio = kept
                        if not current_portfolio:
                             logger.warning(f"❌ {d.date()} | Aucun stock sélectionné!")
                        
                        if current_portfolio:
                            w = self.leverage / len(current_portfolio)
                            for p in current_portfolio: p['Weight'] = w
                else:
                    # Bear Regime : ETFs
                    current_portfolio = [p for p in current_portfolio if p['Type'] == 'ETF']
                    cur_tickers = [p['Ticker'] for p in current_portfolio]
                    needed = 2 - len(current_portfolio)
                    
                    logger.info(f"🛡️ {d.date()} | Mode Bear: Recherche de {needed} ETFs...")
                    
                    if needed > 0:
                        day_etfs = etfs[(etfs['Date'] == d) & (etfs['Eligible']) & (~etfs['Ticker'].isin(cur_tickers))]
                        logger.info(f"🔎 {d.date()} | ETFs éligibles: {len(day_etfs)}")
                        
                        top_etfs = day_etfs.nlargest(needed, 'Momentum_XM')
                        for _, row in top_etfs.iterrows():
                            current_portfolio.append({'Ticker': row['Ticker'], 'Weight': 0, 'Type': 'ETF'})
                    
                    if current_portfolio:
                        w = 1.0 / len(current_portfolio)
                        for p in current_portfolio: p['Weight'] = w
                    else:
                        logger.warning(f"💸 {d.date()} | 100% Cash (Pas d'ETF éligible)")

            # Diagnostic journalier
            w_sum = sum(p['Weight'] for p in current_portfolio) if current_portfolio else 0.0
            portfolio_allocations[d] = {p['Ticker']: p['Weight'] for p in current_portfolio}
            
            if d.weekday() == 4: # On logge seulement les vendredis pour ne pas polluer
                 logger.info(f"📅 {d.date()} | Poids total investi: {w_sum:.2f} | Actifs: {len(current_portfolio)}")
                 
            prev_regime = regime
        
        # LOG FINAL ALLOCATION FOR ANALYSIS
        if current_portfolio:
            final_tickers = [p['Ticker'] for p in current_portfolio]
            logger.info(f"🎯 Dernière Sélection ({d.date()}): {', '.join(final_tickers)}")
        
        return pd.DataFrame(portfolio_allocations).T.fillna(0)

    def generate_performance(self, allocations_df, etfs, stocks, sp500):
        if allocations_df.empty or sp500.empty: return pd.DataFrame()
            
        prices_etf = etfs.pivot(index='Date', columns='Ticker', values='Close') if not etfs.empty else pd.DataFrame()
        prices_stocks = stocks.pivot(index='Date', columns='Ticker', values='Close') if not stocks.empty else pd.DataFrame()
        all_prices = pd.concat([prices_etf, prices_stocks], axis=1)
        
        # --- DIAGNOSTIC ---
        logger.info(f"📊 Diagnostic Performance: ETF {prices_etf.shape}, Stocks {prices_stocks.shape}")
        
        # Harmonisation et Resampling Hebdomadaire (Vendredi) pour matcher les allocations
        all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None).normalize()
        weekly_prices = all_prices.resample('W-FRI').last().ffill()
        weekly_asset_returns = weekly_prices.pct_change()
        
        allocations_shifted = allocations_df.shift(1)
        
        # --- ALIGNEMENT DES TICKERS (Case-Insensitive) ---
        allocations_shifted.columns = [str(c).upper() for c in allocations_shifted.columns]
        weekly_asset_returns.columns = [str(c).upper() for c in weekly_asset_returns.columns]
        
        common_dates = allocations_shifted.index.intersection(weekly_asset_returns.index)
        
        if common_dates.empty:
            logger.warning(f"⚠️ Aucune date commune trouvée ! Allocations: {len(allocations_shifted)}, Prix: {len(weekly_asset_returns)}")
            return pd.DataFrame()
        
        allocations_aligned = allocations_shifted.loc[common_dates]
        
        # Intersection des Tickers pour éviter les colonnes manquantes
        common_tickers = allocations_aligned.columns.intersection(weekly_asset_returns.columns)
        logger.info(f"🧬 Tickers correspondants entre Alloc et Prix: {len(common_tickers)} / {len(allocations_aligned.columns)}")
        
        returns_aligned = weekly_asset_returns.loc[common_dates, common_tickers].fillna(0)
        alloc_aligned_final = allocations_aligned[common_tickers]
        
        # --- DIAGNOSTIC VALEURS ---
        logger.info(f"⚖️ Somme totale des poids investis: {alloc_aligned_final.sum().sum():.2f}")
        total_ret_sum = returns_aligned.sum().sum()
        logger.info(f"📊 Somme totale des rendements actifs: {total_ret_sum:.6f}")
        
        asset_return = (alloc_aligned_final * returns_aligned).sum(axis=1)
        total_invested = alloc_aligned_final.sum(axis=1)
        cash_weight = 1.0 - total_invested
        CASH_YIELD_WEEKLY = (1 + self.cash_yield_annual)**(1/52) - 1
        cash_return = cash_weight * CASH_YIELD_WEEKLY
        
        MARGIN_RATE = (1 + self.margin_rate_annual)**(1/52) - 1
        borrowed_weight = total_invested.clip(lower=1.0) - 1.0
        margin_cost = borrowed_weight * MARGIN_RATE
        
        weight_changes = allocations_shifted.diff().abs().sum(axis=1).fillna(0)
        transaction_costs = weight_changes * self.trading_fee_rate
        
        portfolio_return_weekly_net = asset_return + cash_return - margin_cost - transaction_costs
        equity_curve = (1 + portfolio_return_weekly_net.fillna(0)).cumprod() * 100
        
        # Accès à l'indice S&P 500 (Case-Insensitive)
        close_col_sp500 = 'Close' if 'Close' in sp500.columns else 'close'
        sp500_weekly = sp500[close_col_sp500].resample('W-FRI').last().ffill()
        sp500_returns = sp500_weekly.pct_change().loc[common_dates]
        sp500_equity = (1 + sp500_returns.fillna(0)).cumprod() * 100
        
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        
        perf_df = pd.DataFrame({
            'Portfolio_Return': portfolio_return_weekly_net,
            'Portfolio_Equity': equity_curve,
            'Drawdown': drawdown,
            'SP500_Return': sp500_returns,
            'SP500_Equity': sp500_equity
        })
        
        if not perf_df.empty:
            perf_df.iloc[0, perf_df.columns.get_loc('Portfolio_Equity')] = 100.0
            perf_df.iloc[0, perf_df.columns.get_loc('SP500_Equity')] = 100.0
            
            # --- CALCUL DES MÉTRIQUES ---
            n_years = len(perf_df) / 52
            total_return = (perf_df['Portfolio_Equity'].iloc[-1] / 100) - 1
            
            # CAGR
            cagr = (1 + total_return)**(1/n_years) - 1 if n_years > 0 and total_return > -1 else -1
            perf_df['CAGR'] = cagr
            
            # Max Drawdown
            max_dd = abs(perf_df['Drawdown'].min())
            perf_df['Max_Drawdown'] = max_dd
            
            # Sharpe Ratio (Volatilité annualisée)
            vol_weekly = perf_df['Portfolio_Return'].std()
            vol_ann = vol_weekly * np.sqrt(52)
            # On considère un taux sans risque de 0% pour le Sharpe simplifié
            perf_df['Sharpe_Ratio'] = (cagr / vol_ann) if vol_ann > 0 else 0
            
            # Calmar Ratio
            perf_df['Calmar_Ratio'] = cagr / max_dd if max_dd > 0 else 0
        
        if not perf_df.empty:
            total_ret = (perf_df['Portfolio_Equity'].iloc[-1] / 100) - 1
            logger.success(f"✅ Backtest terminé sur {len(perf_df)} semaines. Rendement total: {total_ret*100:.2f}%")
        
        return perf_df
