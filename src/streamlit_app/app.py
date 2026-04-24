import os
import sys
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import datetime
import mlflow

# --- SETUP PATHS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.common.setup_spark import create_spark_session
import importlib
import src.strategy.backtest_engine as backtest_engine
importlib.reload(backtest_engine)
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
# FORCE CLEAR CACHING ON RELOAD (Temporary for validation)
st.cache_data.clear()
st.cache_resource.clear()

@st.cache_resource(show_spinner="🔌 Initialisation de Spark... (Peut prendre 5s)")
def init_spark():
    return create_spark_session(app_name="Streamlit_Backtester", log_level="ERROR")

@st.cache_data(show_spinner="🔍 Recherche du meilleur modèle dans MLFlow...", ttl=10)
def get_mlflow_config_from_docker():
    try:
        # Configuration de l'URI
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
        if os.path.exists("/.dockerenv"):
            final_uri = "http://mlflow:5000"
        else:
            final_uri = mlflow_uri
            
        mlflow.set_tracking_uri(final_uri)
        
        # 1. Identifier l'expérience
        experiment_name = "Momentum_Bayesian_Optimization_v2_Calmar"
        exp = mlflow.get_experiment_by_name(experiment_name)
        
        if exp is None:
            st.sidebar.warning(f"⚠️ Expérience '{experiment_name}' introuvable. Utilisation des paramètres par défaut.")
            return get_default_config()

        # 2. Chercher le meilleur Run (celui avec le meilleur Calmar Ratio)
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=1,
            order_by=["metrics.calmar DESC"]
        )
        
        if runs.empty:
            st.sidebar.info("ℹ️ Aucun run trouvé dans MLflow. Lancez le notebook d'optimisation !")
            return get_default_config()
            
        # Extraction des paramètres du meilleur run
        best_run = runs.iloc[0]
        best_params = {k.replace("params.", ""): v for k, v in best_run.items() if k.startswith("params.")}
        
        st.sidebar.success(f"🏆 Meilleur Run chargé (Calmar: {best_run['metrics.calmar']:.2f})")
        
        # Mapping et conversion des types
        config = {
            'sp500_sma_slow': int(float(best_params.get('sp500_sma_slow', 40))),
            'sp500_sma_fast': int(float(best_params.get('sp500_sma_fast', 20))),
            'stock_sma_fast': int(float(best_params.get('stock_sma_fast', 30))),
            'stock_sma_slow': int(float(best_params.get('stock_sma_slow', 40))),
            'etf_sma_fast': int(float(best_params.get('etf_sma_fast', 29))),
            'etf_sma_slow': int(float(best_params.get('etf_sma_slow', 50))),
            'stock_atr_threshold': float(best_params.get('stock_atr_threshold', 0.1)),
            'stock_adx_threshold': float(best_params.get('stock_adx_threshold', 20.0)),
            'buffer_n': int(float(best_params.get('buffer_n', 15))),
            'top_n': int(float(best_params.get('top_n', 10))),
            'rebalance_freq': best_params.get('rebalance_freq', '1M'),
            'stock_mom_period': int(float(best_params.get('stock_mom_period', 13))),
            'etf_mom_period': int(float(best_params.get('etf_mom_period', 13))),
            'cash_yield': float(best_params.get('cash_yield', 0.04)),
            'margin_rate': float(best_params.get('margin_rate', 0.06)),
            'fees': float(best_params.get('fees', 0.001)),
            # Ces conditions sont désormais désactivées par défaut (stratégie pure)
            'use_pullback': False,
            'use_cond_1W': False
        }
        return config

    except Exception as e:
        st.error(f"❌ Erreur MLFlow : {e}")
        return get_default_config()

def get_default_config():
    """Paramètres de secours si MLflow est vide ou inaccessible"""
    return {
        'sp500_sma_slow': 50, 'sp500_sma_fast': 26, 'stock_sma_fast': 26, 'stock_sma_slow': 50,
        'etf_sma_fast': 12, 'etf_sma_slow': 26, 'stock_atr_threshold': 0.1, 'stock_adx_threshold': 20.0,
        'use_pullback': False, 'use_cond_1W': False, 'buffer_n': 15, 'top_n': 10, 'rebalance_freq': '1M',
        'stock_mom_period': 13, 'etf_mom_period': 13, 'cash_yield': 0.04, 'margin_rate': 0.06, 'fees': 0.001
    }

# @st.cache_data(show_spinner="🧠 Calcul de l'Intelligence Financière en cours...", ttl=10)
def compute_backtest(start_date, leverage, config):
    spark = init_spark()
    engine = RegimeSwitchingMomentumBacktester(
        config=config,
        start_date=start_date.strftime("%Y-%m-%d"), 
        leverage=leverage
    )
    
    # Exécution complète de la stratégie
    df_sp500 = engine.get_sp500_regime(spark)
    df_etf, df_stocks = engine.load_and_prep_data(spark, Paths.BQ_ETF_GOLD, Paths.BQ_STOCKS_GOLD)
    
    if df_stocks.empty and df_etf.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    # --- DEBUG INFO ---
    latest_date = df_stocks['date'].max() if not df_stocks.empty else "N/A"
    st.sidebar.info(f"📅 Dernières données chargées : {latest_date}")
        
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

leverage = st.sidebar.slider("Niveau de Levier (x)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

run_button = st.sidebar.button("🚀 Lancer la Simulation", use_container_width=True)

# --- MAIN PAGE CONFIG ---
st.markdown('<h1 class="main-title">Momentum AI</h1>', unsafe_allow_html=True)

if run_button or 'perf_df' in st.session_state:
    try:
        config = get_mlflow_config_from_docker()
            
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
            st.write(f"- ATR : {config['stock_atr_threshold']}")
            st.write(f"- ADX : {config['stock_adx_threshold']}")
            st.markdown("**Options :**")
            st.write(f"- Pullback : {'Activé' if config['use_pullback'] else 'Désactivé'}")
            st.write(f"- Cond. 1W : {'Activé' if config['use_cond_1W'] else 'Désactivé'}")




        
        perf_df, allocations, df_etf, df_stocks = compute_backtest(start_date, leverage, config)
        
        # Sauver dans le state pour éviter le rechargement forcé
        st.session_state['perf_df'] = perf_df
        st.session_state['allocations'] = allocations
        
        if perf_df.empty:
            st.error("Aucune donnée générée. Vérifiez si les tables GOLD sont remplies ou si les dates sont cohérentes.")
            st.stop()
            
        # Affichage de la fraîcheur des données Airflow (Date maximale disponible)
        last_data_date = perf_df.index[-1].strftime('%d/%m/%Y')
        st.info(f"🔄 **Base de Données AirFlow** : Fraîcheur des cotations garantie jusqu'au **{last_data_date}**")
            
        # --- CALCUL DES KPIs ---
        # Rendement total
        total_ret_strat = perf_df['Portfolio_Equity'].iloc[-1] - 100.0
        
        # CAGR (Compound Annual Growth Rate)
        n_days = (perf_df.index[-1] - perf_df.index[0]).days
        n_years = max(n_days / 365.25, 0.1) # Sécurité division par zéro
        cagr = ((perf_df['Portfolio_Equity'].iloc[-1] / 100.0) ** (1.0 / n_years)) - 1.0
        
        # Sharpe Ratio (Annuélisé)
        # On assume un taux sans risque de 0% pour le calcul rapide
        weekly_returns = perf_df['Portfolio_Return']
        vol_ann = weekly_returns.std() * np.sqrt(52)
        sharpe = (weekly_returns.mean() * 52) / vol_ann if vol_ann > 0 else 0
        
        # Max Drawdown
        roll_max = perf_df['Portfolio_Equity'].cummax()
        drawdown = (perf_df['Portfolio_Equity'] / roll_max) - 1.0
        max_drawdown = drawdown.min() * 100.0
        
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
        
        # Courbe principale (Momentum) avec effet de lueur
        fig_equity.add_trace(go.Scatter(
            x=perf_df.index, y=perf_df['Portfolio_Value'],
            mode='lines',
            name='Portefeuille Momentum',
            line=dict(color='#00f2fe', width=4, shape='spline', smoothing=1.3),
            fill='tozeroy',
            fillcolor='rgba(0, 242, 254, 0.05)',
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
                # 1. Identifier les blocs de détention continus pour chaque ticker
                is_held = allocations > 0
                # On crée des IDs de blocs uniques par ticker
                blocks = (is_held != is_held.shift()).cumsum()
                # On filtre pour ne garder que les périodes de détention réelle
                blocks_filtered = blocks.where(is_held)
                
                # 2. Transformation au format LONG
                df_long = blocks_filtered.reset_index().melt(id_vars='index', var_name='Ticker', value_name='BlockID')
                df_long = df_long.dropna(subset=['BlockID'])
                df_long.rename(columns={'index': 'Date'}, inplace=True)
                
                # 3. Récupération des poids (Weight)
                weights_melted = allocations.reset_index().melt(id_vars='index', var_name='Ticker', value_name='Weight')
                weights_melted.rename(columns={'index': 'Date'}, inplace=True)
                df_final = pd.merge(df_long, weights_melted, on=['Date', 'Ticker'])
                
                # 4. Identification de la date d'entrée par bloc
                df_final['EntryDate'] = df_final.groupby(['Ticker', 'BlockID'])['Date'].transform('min')
                
                # 5. Mapping des prix pour calcul de performance (Optimisé)
                price_lookup = {}
                unique_tickers = df_final['Ticker'].unique()
                for t in unique_tickers:
                    if t == 'Cash': continue
                    # Choix de la source de données (Stocks vs ETF)
                    if t in df_stocks['Ticker'].values:
                        p_df = df_stocks[df_stocks['Ticker'] == t]
                        p_col = 'adjClose'
                    else:
                        p_df = df_etf[df_etf['Ticker'] == t]
                        p_col = 'Close'
                    
                    if not p_df.empty:
                        p_df = p_df.copy()
                        p_df['d_naive'] = pd.to_datetime(p_df['date']).dt.tz_localize(None)
                        price_lookup[t] = dict(zip(p_df['d_naive'], p_df[p_col]))

                def compute_perfs(row):
                    ticker = row['Ticker']
                    if ticker == 'Cash': return 0.0, 0.0
                    d_now = pd.Timestamp(row['Date']).replace(tzinfo=None)
                    d_entry = pd.Timestamp(row['EntryDate']).replace(tzinfo=None)
                    
                    # Performance Hebdo : Gain depuis la période précédente dans l'historique
                    try:
                        idx = allocations.index.get_loc(row['Date'])
                        if idx > 0:
                            d_prev = pd.Timestamp(allocations.index[idx-1]).replace(tzinfo=None)
                        else:
                            d_prev = d_now
                    except:
                        d_prev = d_now
                    
                    p_now = price_lookup.get(ticker, {}).get(d_now)
                    p_entry = price_lookup.get(ticker, {}).get(d_entry)
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
    st.info("👈 Réglez vos paramètres (Date, Levier) dans le menu de gauche et cliquez sur **Lancer la Simulation** !")
