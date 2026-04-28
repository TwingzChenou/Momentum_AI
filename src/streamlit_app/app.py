import os
import sys
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import datetime
import mlflow
from loguru import logger

# --- SETUP PATHS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.common.setup_spark import create_spark_session
import importlib
import src.strategy.backtest_engine as backtest_engine
importlib.reload(backtest_engine)
from src.common.config_utils import get_champion_config
from src.strategy.backtest_engine import RegimeSwitchingMomentumBacktester
from config.config_spark import Paths

# --- PAGE CONFIG ---
st.set_page_config(page_title="Momentum AI - Backtest Engine", page_icon="🚀", layout="wide")

# --- CUSTOM CSS (PREMIUM DESIGN) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&display=swap');

/* Styles globaux (Dark Glassmorphism) */
.stApp {
    background-color: #111827;
    color: #e2e8f0;
}
/* Réduire l'espace au sommet de la page */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
}
/* Supprimer le bandeau blanc au sommet */
header {
    background-color: rgba(0,0,0,0) !important;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0) !important;
}
/* Style du bandeau latéral (Sidebar) */
[data-testid="stSidebar"] {
    background-color: #0b0f19 !important;
    background-image: none !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}
[data-testid="stSidebarNav"] {
    background-color: transparent !important;
}
/* Ajustement des textes dans la sidebar pour le mode sombre */
[data-testid="stSidebar"] .stMarkdown h1, 
[data-testid="stSidebar"] .stMarkdown h2, 
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}
/* Cartes de KPIs (Gros Nombres) */
.metric-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.2);
    border: 1px solid rgba(0, 242, 254, 0.4);
}
.metric-title {
    font-size: 1rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
    margin-bottom: 10px;
}
.big-font {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin: 0;
}
.main-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 6.5rem !important;
    font-weight: 700;
    text-align: center;
    color: #ffffff !important;
    margin-top: -3.5rem;
    margin-bottom: 2rem;
    letter-spacing: -4px;
    line-height: 1;
}
/* Style des boutons (Sombre avec écriture blanche) */
.stButton > button {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background-color: #334155 !important;
    border: 1px solid #00f2fe !important;
    box-shadow: 0 0 15px rgba(0, 242, 254, 0.2) !important;
    color: #00f2fe !important;
}
/* Style du sélecteur de date */
div[data-testid="stDateInput"] > div {
    background-color: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}
div[data-testid="stDateInput"] input {
    color: #ffffff !important;
}
div[data-testid="stDateInput"] svg {
    fill: #ffffff !important;
}
.big-font.sp500 {
    background: linear-gradient(135deg, #cbd5e1 0%, #64748b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.big-font.drawdown {
    background: linear-gradient(135deg, #f87171 0%, #b91c1c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
/* Style pour forcer le texte en blanc dans l'expander de la sidebar */
[data-testid="stSidebar"] [data-testid="stExpander"] p,
[data-testid="stSidebar"] [data-testid="stExpander"] li {
    color: #ffffff !important;
}
/* Masquer le menu hamburger par défaut */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- CACHING & DATA LOADERS ---

@st.cache_resource(show_spinner="🔌 Initialisation de Spark... (Peut prendre 5s)")
def init_spark():
    return create_spark_session(app_name="Streamlit_Backtester", log_level="ERROR")

@st.cache_data(show_spinner="📥 Chargement des données GOLD (BigQuery)...", ttl=3600)
def load_and_prep_all_data(start_date_str, config_dict):
    spark = init_spark()
    
    # 1. Indice S&P 500 via BigQuery GOLD
    # 1. S&P 500 via BigQuery GOLD
    df_sp500 = spark.read.format("bigquery").option("table", Paths.BQ_SP500_GOLD).load().toPandas()
    
    # Normalisation pour l'indexation temporelle
    if 'Date' in df_sp500.columns:
        df_sp500['Date'] = pd.to_datetime(df_sp500['Date']).dt.normalize()
        df_sp500 = df_sp500.set_index('Date').sort_index()
    
    # 2. ETFs via BigQuery GOLD
    df_etf = spark.read.format("bigquery").option("table", Paths.BQ_ETF_GOLD).load().toPandas()
    df_etf['Date'] = pd.to_datetime(df_etf['Date']).dt.tz_localize(None).dt.normalize()
    
    # 3. Actions via BigQuery GOLD
    df_stocks = spark.read.format("bigquery").option("table", Paths.BQ_STOCKS_GOLD).load().toPandas()
    df_stocks['Date'] = pd.to_datetime(df_stocks['Date']).dt.tz_localize(None).dt.normalize()

    # Initialisation du moteur
    engine = RegimeSwitchingMomentumBacktester(config=config_dict, start_date=start_date_str)
    
    # Note: On laisse le moteur gérer la préparation interne lors du simulate_portfolio.
    # On s'assure juste que le S&P 500 a son régime pour l'affichage des graphiques.
    df_sp500 = engine.get_sp500_regime_from_df(df_sp500)
    
    return df_sp500, df_etf, df_stocks

@st.cache_data(show_spinner="🔍 Recherche du meilleur modèle dans MLFlow...", ttl=10)
def load_champion_config():
    config = get_champion_config()
    if 'calmar' in config: # Optionnel: log d'info si présent
         st.sidebar.success("🏆 Meilleur Run chargé depuis MLFlow")
    return config

@st.cache_data(show_spinner="🧠 Calcul de l'Intelligence Financière en cours...", ttl=300)
def compute_backtest_cached(start_date_str, leverage, config_dict):
    # On charge les données (cachées)
    df_sp500, df_etf, df_stocks = load_and_prep_all_data(start_date_str, config_dict)
    
    if df_stocks.empty and df_etf.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    engine = RegimeSwitchingMomentumBacktester(
        config=config_dict,
        start_date=start_date_str, 
        leverage=leverage
    )
    
    # Exécution du backtest (rapide grâce aux index/dictionnaires)
    allocations = engine.simulate_portfolio(df_sp500, df_etf, df_stocks)
    perf_df = engine.generate_performance(allocations, df_etf, df_stocks, df_sp500)
    
    return perf_df, allocations, df_etf, df_stocks

# --- SIDEBAR CONFIG ---
st.sidebar.title("⚙️ Paramètres de Backtest")

# Dates limites
min_date = datetime.date(2000, 1, 1)
max_date = datetime.date.today()
default_start = datetime.date(2010, 1, 1)

# Gestion de la session state pour la date
if 'start_date_input' not in st.session_state:
    st.session_state.start_date_input = default_start

def set_start_date(period_days):
    if period_days == "since":
        st.session_state.start_date_input = min_date
    else:
        st.session_state.start_date_input = datetime.date.today() - datetime.timedelta(days=period_days)
    st.rerun()

st.sidebar.markdown("### 🕒 Quick Select")
# Ligne 1: 1M, 3M, 6M
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("1M", use_container_width=True): set_start_date(30)
if c2.button("3M", use_container_width=True): set_start_date(90)
if c3.button("6M", use_container_width=True): set_start_date(180)

# Ligne 2: 1Y, 5Y, 10Y
c4, c5, c6 = st.sidebar.columns(3)
if c4.button("1Y", use_container_width=True): set_start_date(365)
if c5.button("5Y", use_container_width=True): set_start_date(1825)
if c6.button("10Y", use_container_width=True): set_start_date(3650)

# Ligne 3: Since Everything
if st.sidebar.button("Since 2000 (Max)", use_container_width=True): set_start_date("since")

st.sidebar.write("---")

capital_initial = st.sidebar.number_input("Capital Initial ($)", value=100000, step=5000, min_value=1000, help="Capital au début de la période de backtest.")
capital_actuel = st.sidebar.number_input("Capital Actuel ($)", value=100000, step=5000, min_value=1000, help="Votre capital aujourd'hui pour calculer les ordres à passer.")

# Affichage du widget avec synchronisation automatique via 'key'
start_date = st.sidebar.date_input(
    "Date de début", 
    value=st.session_state.start_date_input, 
    min_value=min_date, 
    max_value=max_date, 
    key="start_date_input"
)

leverage = st.sidebar.slider("🚀 Effet de Levier", 1.0, 3.0, 1.0, 0.1)

run_button = st.sidebar.button("🚀 Lancer la Simulation", use_container_width=True)


# --- MAIN PAGE CONFIG ---
st.markdown('<h1 class="main-title">Momentum AI</h1>', unsafe_allow_html=True)

if run_button or 'perf_df' in st.session_state:
    try:
        config = load_champion_config()
            
        st.sidebar.success(f"✅ Modèle MLFlow 'Champion' chargé")
        
        with st.sidebar.expander("🧠 Détails de la Stratégie"):
            st.markdown(f"**Sélection :** Top {config['top_n']} actifs")
            st.markdown(f"**Fréquence :** {config['rebalance_freq']}")
            st.markdown("---")
            st.markdown("**Cœur Momentum :**")
            st.write(f"- Période Actions : {config['stock_mom_period']} sem.")
            st.write(f"- Période ETFs : {config['etf_mom_period']} sem.")
            st.markdown("**Filtres de Régime :**")
            st.write(f"- S&P 500 : SMA {config['sp500_sma_fast']}/{config['sp500_sma_slow']}")
            st.write(f"- Actions : SMA {config['stock_sma_fast']}/{config['stock_sma_slow']}")
            st.markdown("**Seuils de Risque :**")
            st.write(f"- ATR : {config['stock_atr_threshold']:.2f}")
            st.write(f"- ADX : {config['stock_adx_threshold']:.1f}")




        
        perf_df, allocations, df_etf, df_stocks = compute_backtest_cached(start_date.strftime("%Y-%m-%d"), leverage, config)
        
        # Sauver dans le state pour éviter le rechargement forcé
        st.session_state['perf_df'] = perf_df
        st.session_state['allocations'] = allocations
        
        if perf_df.empty:
            st.error("Aucune donnée générée. Vérifiez si les tables GOLD sont remplies ou si les dates sont cohérentes.")
            st.stop()
            
        # Affichage de la fraîcheur des données Airflow (Date maximale disponible)
        last_data_date = perf_df.index[-1].strftime('%d/%m/%Y')
        st.info(f"🔄 **Base de Données AirFlow** : Fraîcheur des cotations garantie jusqu'au **{last_data_date}**")

        # --- RÉCUPÉRATION DES KPIs (Calculés par l'Engine) ---
        total_ret_strat = perf_df['Portfolio_Equity'].iloc[-1] - 100.0
        cagr = perf_df['CAGR'].iloc[-1] if 'CAGR' in perf_df.columns else 0
        sharpe = perf_df['Sharpe_Ratio'].iloc[-1] if 'Sharpe_Ratio' in perf_df.columns else 0
        max_drawdown = -perf_df['Max_Drawdown'].iloc[-1] * 100.0 if 'Max_Drawdown' in perf_df.columns else 0
        
        # --- DISPLAY KPIs (GROS NOMBRES) ---
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Return</div>
                <div class="big-font\">{total_ret_strat:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">CAGR (Annuel)</div>
                <div class="big-font\">{cagr*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Ratio de Sharpe</div>
                <div class="big-font sp500">{sharpe:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Max Drawdown</div>
                <div class="big-font drawdown">{max_drawdown:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("---")

        # --- 📈 PLOTLY EQUITY CURVE (PREMIUM DESIGN) ---
        st.subheader("Évolution de la Valeur du Portefeuille ($)")
        
        # Calcul de la valeur en dollars basée sur le capital initial
        perf_df['Portfolio_Value'] = (perf_df['Portfolio_Equity'] / 100.0) * capital_initial
        perf_df['SP500_Value'] = (perf_df['SP500_Equity'] / 100.0) * capital_initial
        
        fig_equity = go.Figure()
        
        # Courbe principale (Momentum)
        fig_equity.add_trace(go.Scatter(
            x=perf_df.index, y=perf_df['Portfolio_Value'],
            mode='lines',
            name='Portefeuille Momentum',
            line=dict(color='#4facfe', width=3),
            fill='tozeroy',
            fillcolor='rgba(79, 172, 254, 0.1)',
            hovertemplate='<b>Date</b>: %{x}<br><b>Valeur</b>: $ %{y:,.0f}<extra></extra>'
        ))
        
        # Benchmark (S&P 500)
        fig_equity.add_trace(go.Scatter(
            x=perf_df.index, y=perf_df['SP500_Value'],
            mode='lines',
            name='S&P 500 (Benchmark)',
            line=dict(color='rgba(203, 213, 225, 0.4)', width=2, dash='dot'),
            hovertemplate='<b>Date</b>: %{x}<br><b>S&P 500</b>: $ %{y:,.0f}<extra></extra>'
        ))
        
        fig_equity.update_layout(
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Grotesk, sans-serif', color='#94a3b8'),
            hovermode="x unified",
            xaxis=dict(
                showgrid=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="MAX")
                    ]),
                    bgcolor="#1e293b",
                    activecolor="#00f2fe",
                    font=dict(color="#ffffff", size=11)
                ),
                type="date"
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='rgba(255,255,255,0.05)', 
                type='log',
                title="Valeur ($) - Échelle Log",
                side="right"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        st.plotly_chart(fig_equity, use_container_width=True, config={'displayModeBar': False})
        
        if not allocations.empty:
            # --- 🔄 DERNIERS MOUVEMENTS (TRADES) ---
            if len(allocations) >= 2:
                st.write("---")
                st.subheader("🔄 Derniers Mouvements du Portefeuille")
                
                last_date = allocations.index[-1]
                prev_date = allocations.index[-2]
                
                curr_alloc = allocations.loc[last_date]
                prev_alloc = allocations.loc[prev_date]
                
                # Entrées : Présent aujourd'hui, absent hier
                to_enter = curr_alloc[(curr_alloc > 0) & (prev_alloc == 0)]
                # Sorties : Absent aujourd'hui, présent hier
                to_exit = prev_alloc[(prev_alloc > 0) & (curr_alloc == 0)]
                
                if not to_enter.empty or not to_exit.empty:
                    trade_data = []
                    for t, w in to_enter.items():
                        trade_data.append({
                            'Ticker': t, 
                            'Action': 'ACHAT', 
                            'Poids (%)': w * 100, 
                            'Montant ($)': w * capital_actuel
                        })
                    for t, w in to_exit.items():
                        trade_data.append({
                            'Ticker': t, 
                            'Action': 'VENTE', 
                            'Poids (%)': w * 100, 
                            'Montant ($)': w * capital_actuel
                        })
                    
                    df_trades = pd.DataFrame(trade_data)
                    
                    def style_trades(row):
                        color = 'rgba(34, 197, 94, 0.15)' if row['Action'] == 'ACHAT' else 'rgba(239, 68, 68, 0.15)'
                        return [f'background-color: {color}'] * len(row)

                    st.dataframe(
                        df_trades.style.apply(style_trades, axis=1),
                        column_config={
                            "Poids (%)": st.column_config.NumberColumn("Poids (%)", format="%.1f%%"),
                            "Montant ($)": st.column_config.NumberColumn("Estimation ($)", format="$ %.0f"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.success("✨ Aucun changement de position lors du dernier rebalancement.")

            # --- 📊 HISTORIQUE DE LA COMPOSITION (MODERNE) ---
            st.write("---")
            st.subheader("📜 Historique Détaillé du Portefeuille")
            
            with st.spinner("Calcul des performances historiques..."):
                # On décale les allocations pour afficher ce qu'on détenait RÉELLEMENT pendant la période
                # La performance affichée à T est celle des actifs choisis à T-1
                alloc_hist = allocations.shift(1).fillna(0)
                
                # 1. Identifier les blocs de détention continus pour chaque ticker
                is_held = alloc_hist > 0
                # On crée des IDs de blocs uniques par ticker
                blocks = (is_held != is_held.shift()).cumsum()
                # On filtre pour ne garder que les périodes de détention réelle
                blocks_filtered = blocks.where(is_held)
                
                # 2. Transformation au format LONG
                df_long = blocks_filtered.reset_index().melt(id_vars='index', var_name='Ticker', value_name='BlockID')
                df_long = df_long.dropna(subset=['BlockID'])
                df_long.rename(columns={'index': 'Date'}, inplace=True)
                
                # 3. Récupération des poids (Weight)
                weights_melted = alloc_hist.reset_index().melt(id_vars='index', var_name='Ticker', value_name='Weight')
                weights_melted.rename(columns={'index': 'Date'}, inplace=True)
                df_final = pd.merge(df_long, weights_melted, on=['Date', 'Ticker'])
                
                # 4. Identification de la date d'entrée par bloc
                df_final['EntryDate'] = df_final.groupby(['Ticker', 'BlockID'])['Date'].transform('min')
                
                # 5. Mapping des prix pour calcul de performance (Optimisé)
                price_lookup = {}
                stocks_tickers_set = set(df_stocks['Ticker'].unique())
                unique_tickers = df_final['Ticker'].unique()
                
                for t in unique_tickers:
                    if t == 'Cash': continue
                    if t in stocks_tickers_set:
                        p_df = df_stocks[df_stocks['Ticker'] == t]
                        p_col = 'Close'
                    else:
                        p_df = df_etf[df_etf['Ticker'] == t]
                        p_col = 'Close'
                    
                    if not p_df.empty:
                        # Les dates sont déjà normalisées en pd.Timestamp par load_and_prep_all_data
                        price_lookup[t] = dict(zip(pd.to_datetime(p_df['Date']), p_df[p_col]))

                def compute_perfs(row):
                    ticker = row['Ticker']
                    if ticker == 'Cash': return pd.Series([0.0, 0.0])
                    
                    # Les dates d'entrée et actuelles sont déjà des Timestamps cohérents
                    d_now = pd.to_datetime(row['Date'])
                    d_entry = pd.to_datetime(row['EntryDate'])
                    
                    try:
                        idx = allocations.index.get_loc(row['Date'])
                        if idx > 0:
                            d_prev = pd.to_datetime(allocations.index[idx-1])
                        else:
                            d_prev = d_now
                    except:
                        d_prev = d_now
                    
                    p_now = price_lookup.get(ticker, {}).get(d_now)
                    
                    # Correction du prix d'entrée : On prend le prix de la veille (T-1) du début du bloc
                    try:
                        idx_entry = allocations.index.get_loc(d_entry)
                        if idx_entry > 0:
                            d_true_entry = allocations.index[idx_entry - 1]
                        else:
                            d_true_entry = d_entry
                    except:
                        d_true_entry = d_entry
                        
                    p_entry = price_lookup.get(ticker, {}).get(d_true_entry)
                    p_prev = price_lookup.get(ticker, {}).get(d_prev)
                    
                    perf_total = (p_now / p_entry - 1) * 100 if p_now and p_entry else 0.0
                    perf_weekly = (p_now / p_prev - 1) * 100 if p_now and p_prev else 0.0
                    
                    return pd.Series([perf_total, perf_weekly])

                df_final[['Perf. Totale (%)', 'Perf. Hebdo (%)']] = df_final.apply(compute_perfs, axis=1)
                
                # Récupération de l'equity historique (Base 100) pour chaque date
                equity_series = (perf_df['Portfolio_Equity'] / 100.0).to_frame(name='Portfolio_Equity_Ratio')
                df_final = pd.merge(df_final, equity_series, left_on='Date', right_index=True)
                
                # Tri : Plus récent en premier, puis plus gros poids
                df_final = df_final.sort_values(by=['Date', 'Weight'], ascending=[False, False])
                df_display = df_final[['Date', 'Ticker', 'Weight', 'Perf. Totale (%)', 'Perf. Hebdo (%)', 'Portfolio_Equity_Ratio']].copy()
                
                # Calcul de la valeur dynamique : (Capital Initial * Ratio de croissance) * Poids
                df_display['Valeur ($)'] = capital_initial * df_display['Portfolio_Equity_Ratio'] * df_display['Weight']
                # Multiplication par 100 pour l'affichage en %
                df_display['Weight'] = df_display['Weight'] * 100
            
            # Style du tableau : Couleur adaptée à la performance
            def style_perf(val):
                if val > 0:
                    return 'background-color: rgba(34, 197, 94, 0.15); color: #4ade80; font-weight: bold;'
                elif val < 0:
                    return 'background-color: rgba(239, 68, 68, 0.15); color: #f87171; font-weight: bold;'
                return ''

            df_styled = df_display.style.applymap(style_perf, subset=['Perf. Totale (%)', 'Perf. Hebdo (%)'])

            # Affichage moderne via Streamlit Dataframe
            st.dataframe(
                df_styled,
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                    "Ticker": st.column_config.TextColumn("Actif"),
                    "Weight": st.column_config.NumberColumn(
                        "Poids (%)",
                        help="Allocation dans le portefeuille",
                        format="%.1f%%",
                    ),
                    "Valeur ($)": st.column_config.NumberColumn(
                        "Position ($)",
                        help="Valeur estimée de la position basée sur le capital initial",
                        format="$ %.0f",
                    ),
                    "Perf. Totale (%)": st.column_config.NumberColumn(
                        "Total (%)",
                        help="Performance du ticker depuis son entrée en portefeuille",
                        format="%.2f%%",
                    ),
                    "Perf. Hebdo (%)": st.column_config.NumberColumn(
                        "Hebdo (%)",
                        help="Performance du ticker sur la semaine (entre deux rebalancements)",
                        format="%.2f%%",
                    )
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )
        else:
            st.warning("⚠️ L'algorithme est actuellement 100% Cash / Aucun Trade actif.")
    except Exception as e:
        st.error(f"❌ Une erreur interne est survenue lors du calcul : {e}")
else:
    # --- WELCOME SCREEN ---
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Bienvenue sur Momentum AI 🚀
        
        Cet outil utilise l'intelligence artificielle pour optimiser une stratégie de **Regime-Switching Momentum**.
        
        **Comment ça marche ?**
        1. Les données sont extraites en temps réel de **BigQuery GOLD**.
        2. Le modèle récupère la meilleure configuration ("Champion") depuis **MLFlow**.
        3. Une simulation complète est effectuée sur l'historique choisi.
        
        **Prêt à commencer ?**
        Cliquez sur le bouton **🚀 Lancer la Simulation** dans la barre latérale pour générer les analyses.
        """)
    with col2:
        st.info("💡 **Astuce** : Vous pouvez modifier le levier et la date de début pour tester différents scénarios de risque.")
