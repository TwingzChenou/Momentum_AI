import pandas as pd
import numpy as np
import yfinance as yf
import ta
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    from config.config_spark import Paths
except ImportError:
    class Paths:
        SP500_STOCK_PRICES = "path/to/stock_prices"

class RegimeSwitchingMomentumBacktester:
    def __init__(self, start_date="2010-01-01", end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        
        self.etf_tickers = [
            'XLP', 'XLV', 'XLU', 'XLE', 'XLK', 'XLC', 'XLI', 'XLY', 
            'XLB', 'XLRE', 'TLT', 'IEF', 'HYG', 'DBC', 'GLD', 'VNQ'
        ]
        
    def get_sp500_regime(self) -> pd.DataFrame:
        """
        Market Regime Filter (S&P 500)
        Bull Market: MA(12 weeks) > MA(26 weeks) AND Prix > MA(12)
        """
        print("Fetching S&P 500 data for Regime Filter...")
        sp500 = yf.download('^GSPC', start=self.start_date, end=self.end_date, progress=False)
        sp500 = pd.DataFrame(sp500['Close'].resample('W-FRI').last())
        sp500.columns = ['Close']
        sp500['SMA_12'] = ta.trend.sma_indicator(sp500['Close'], window=12)
        sp500['SMA_26'] = ta.trend.sma_indicator(sp500['Close'], window=26)
        
        cond_bull = (sp500['SMA_12'] > sp500['SMA_26']) & (sp500['Close'] > sp500['SMA_12'])
        sp500['Regime'] = np.where(cond_bull, 'Bull', 'Bear')
        return sp500[['Close', 'SMA_12', 'SMA_26', 'Regime']]
        
    def get_etf_data(self) -> pd.DataFrame:
        """
        ETF Data for Bear Market Strategy
        """
        print("Fetching ETF data...")
        etf_data = yf.download(self.etf_tickers, start=self.start_date, end=self.end_date, progress=False)
        etf_close = etf_data['Close'].resample('W-FRI').last()
        
        df_list = []
        for ticker in self.etf_tickers:
            df = etf_close[[ticker]].dropna().rename(columns={ticker: 'Close'})
            df['Ticker'] = ticker
            
            df['SMA_12'] = ta.trend.sma_indicator(df['Close'], window=12)
            df['SMA_26'] = ta.trend.sma_indicator(df['Close'], window=26)
            
            # Momentum 3 Mois (hors dernier mois) = (Prix(t-1 mois) / Prix(t-3 mois)) - 1
            # 1 mois = 4 semaines, 3 mois = 13 semaines (approx 12) => on utilise 4 et 13 semaines
            df['Momentum_3M'] = (df['Close'].shift(4) / df['Close'].shift(13)) - 1
            
            df['Eligible'] = (df['SMA_12'] > df['SMA_26']) & (df['Close'] > df['SMA_12'])
            df_list.append(df.reset_index())
            
        res = pd.concat(df_list, ignore_index=True)
        res = res.rename(columns={'Date': 'date'})
        return res

    def load_and_prep_stock_data(self, spark_session) -> pd.DataFrame:
        """
        Load Daily Stock Prices from SP500_STOCK_PRICES and convert to weekly.
        Calculates ADX(20 days) and ATR(14 days) before resampling.
        """
        print("Loading Daily S&P 500 Constituent Data from Gold Table...")
        df_bronze = spark_session.read.format("delta").load(Paths.SP500_STOCK_PRICES)
        pd_df = df_bronze.select('date', 'symbol', 'adjHigh', 'adjLow', 'adjClose').toPandas()
        
        pd_df['date'] = pd.to_datetime(pd_df['date'])
        print(f"Loaded {len(pd_df)} daily price records. Processing indicators and resampling to weekly...")

        results = []
        for symbol, df in pd_df.groupby('symbol'):
            df = df.sort_values('date').set_index('date')
            
            # Daily Indicators
            df['ADX_20'] = ta.trend.adx(df['adjHigh'], df['adjLow'], df['adjClose'], window=20)
            df['ATR_14'] = ta.volatility.average_true_range(df['adjHigh'], df['adjLow'], df['adjClose'], window=14)
            
            # Resample Weekly (Last of the week)
            weekly_df = df.resample('W-FRI').agg({
                'adjClose': 'last',
                'ADX_20': 'last',
                'ATR_14': 'last'
            }).dropna(subset=['adjClose'])
            
            # Weekly Indicators (MA12, MA26)
            weekly_df['SMA_12'] = ta.trend.sma_indicator(weekly_df['adjClose'], window=12)
            weekly_df['SMA_26'] = ta.trend.sma_indicator(weekly_df['adjClose'], window=26)
            weekly_df['ATR_pct'] = weekly_df['ATR_14'] / weekly_df['adjClose']
            
            # Momentum 6 Mois (hors dernier mois) = 26 weeks vs 4 weeks
            weekly_df['Momentum_6M'] = (weekly_df['adjClose'].shift(4) / weekly_df['adjClose'].shift(26)) - 1
            
            # Eligibility
            cond_trend = (weekly_df['SMA_12'] > weekly_df['SMA_26']) & (weekly_df['adjClose'] > weekly_df['SMA_12'])
            cond_strength = weekly_df['ADX_20'] > 25
            cond_volatility = weekly_df['ATR_pct'] < 0.04
            
            weekly_df['Eligible'] = cond_trend & cond_strength & cond_volatility
            weekly_df['Ticker'] = symbol
            
            results.append(weekly_df.reset_index())
            
        stocks_weekly = pd.concat(results, ignore_index=True)
        return stocks_weekly

    def simulate_portfolio(self, sp500: pd.DataFrame, etfs: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
        """
        Simuler le portefeuille (Rebalancement Mensuel, Signal de vente Hebdomadaire).
        Si Prix < MA(12), l'actif est vendu en Cash.
        """
        print("Running Vectorized Portfolio Simulation...")
        
        # Trouver les dates communes
        dates = sp500.index.intersection(etfs['date'].unique())
        if not stocks.empty:
            dates = dates.intersection(stocks['date'].unique())
        dates = sorted(dates)
        
        if not dates:
            return pd.DataFrame()
        
        # Identifier les dates de rebalancement (la derniere date dispo de chaque mois)
        df_dates = pd.DataFrame({'date': dates})
        df_dates['year_month'] = df_dates['date'].dt.to_period('M')
        rebalance_dates = set(df_dates.groupby('year_month')['date'].max().values)
        
        portfolio_allocations = {}
        current_portfolio = [] # Stocke les positions actutelles ex: {'Ticker': 'AAPL', 'Weight': 0.1, 'Type': 'Stock'}
        
        for d in dates:
            regime = sp500.loc[d, 'Regime']
            
            # --- 1. Rebalancement Mensuel ---
            if d in rebalance_dates:
                # On vide le portefeuille pour tout ré-allouer
                current_portfolio = []
                
                if regime == 'Bull' and not stocks.empty:
                    daily_stocks = stocks[stocks['date'] == d]
                    eligible = daily_stocks[daily_stocks['Eligible']]
                    top_10 = eligible.nlargest(10, 'Momentum_6M')
                    
                    for _, row in top_10.iterrows():
                        current_portfolio.append({'Ticker': row['Ticker'], 'Weight': 0.1, 'Type': 'Stock'})
                
                elif regime == 'Bear' and not etfs.empty:
                    daily_etfs = etfs[etfs['date'] == d]
                    eligible = daily_etfs[daily_etfs['Eligible']]
                    top_2 = eligible.nlargest(2, 'Momentum_3M')
                    
                    for _, row in top_2.iterrows():
                        current_portfolio.append({'Ticker': row['Ticker'], 'Weight': 0.5, 'Type': 'ETF'})
            
            # --- 2. Vérification Hebdomadaire (Signal de vente vers Cash) ---
            # Even if it's the rebalance week, we loop through the target to compute actual final weights for the week
            current_target = {}
            for pos in current_portfolio:
                ticker = pos['Ticker']
                ptype = pos['Type']
                
                # Récupérer les données de la semaine en cours
                if ptype == 'Stock':
                    asset_mask = (stocks['date'] == d) & (stocks['Ticker'] == ticker)
                    if asset_mask.any():
                        asset_data = stocks[asset_mask].iloc[0]
                        price = asset_data['adjClose']
                        ma12 = asset_data['SMA_12']
                        if price < ma12:
                            pos['Weight'] = 0.0 # Vendu en Cash !
                else: 
                    asset_mask = (etfs['date'] == d) & (etfs['Ticker'] == ticker)
                    if asset_mask.any():
                        asset_data = etfs[asset_mask].iloc[0]
                        price = asset_data['Close']
                        ma12 = asset_data['SMA_12']
                        if price < ma12:
                            pos['Weight'] = 0.0 # Vendu en Cash !
                
                current_target[ticker] = pos['Weight']
                
            portfolio_allocations[d] = current_target
            
        weights_df = pd.DataFrame(portfolio_allocations).T.fillna(0)
        print("Backtest processing complete.")
        return weights_df

# Example Execution Block
if __name__ == "__main__":
    backtester = RegimeSwitchingMomentumBacktester(start_date="2015-01-01")
    
    # 1. Get Regime
    sp500 = backtester.get_sp500_regime()
    
    # 2. Get ETFs
    etfs = backtester.get_etf_data()
    
    # 3. Get Stocks
    from src.common.setup_spark import create_spark_session
    spark = create_spark_session("MomentumBacktest")
    stocks = backtester.load_and_prep_stock_data(spark)
    
    # 4. Simulate
    allocations = backtester.simulate_portfolio(sp500, etfs, stocks)
    print("Affichage des dernières allocations du portefeuille (Les poids manquants à 1.0 représentent le Cash)")
    print(allocations.tail(10))
