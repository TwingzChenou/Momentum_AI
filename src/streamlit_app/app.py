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
from src.strategy.backtest_engine import RegimeSwitchingMomentumBacktester
from config.config_spark import Paths

# --- PAGE CONFIG ---
st.set_page_config(page_title="Momentum AI - Backtest Engine", page_icon="🚀", layout="wide")

# --- CUSTOM CSS (PREMIUM DESIGN) ---
st.markdown("""
<style>
/* Styles globaux (Dark Glassmorphism) */
.stApp {
    background-color: #0b0f19;
    color: #e2e8f0;
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
    font-size: 2.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin: 0;
}
.main-title {
    font-size: 4.5rem !important;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #ffffff 0%, #a5f3fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 1rem;
    margin-bottom: 2.5rem;
    letter-spacing: -1px;
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
/* Masquer le menu hamburger par défaut */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- CACHING & DATA LOADERS ---
@st.cache_resource(show_spinner="🔌 Initialisation de Spark... (Peut prendre 5s)")
def init_spark():
    return create_spark_session(app_name="Streamlit_Backtester", log_level="ERROR")

@st.cache_data(show_spinner="🔍 Connexion au conteneur MLFlow interne...", ttl=3600)
def get_mlflow_config_from_docker():
    try:
        # Port par défaut si l'on n'est pas dans Docker (usage Mac natif)
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        
        # Mode Conteneurisé : Streamlit tape directement sur le service interne 'mlflow' de docker-compose
        if os.path.exists("/.dockerenv"):
            final_uri = "http://mlflow:5000"
        else:
            final_uri = mlflow_uri
            
        mlflow.set_tracking_uri(final_uri)
        
        # Passage EXPLICITE de l'URI au client pour ne PAS utiliser de potentielles configurations SQLite en cache
        client = mlflow.tracking.MlflowClient(tracking_uri=final_uri, registry_uri=final_uri)
        model_version = client.get_model_version_by_alias("Momentum_Strategy", "champion")
        best = client.get_run(model_version.run_id).data.params
        
        config = {
            'sp500_sma_slow': int(best.get('sp500_sma_slow', 40)),
            'sp500_sma_fast': int(best.get('sp500_sma_fast', 20)),
            'stock_sma_fast': int(best.get('stock_sma_fast', 30)),
            'stock_sma_slow': int(best.get('stock_sma_slow', 40)),
            'etf_sma_fast': int(best.get('etf_sma_fast', 29)),
            'etf_sma_slow': int(best.get('etf_sma_slow', 50)),
            'stock_atr_threshold': float(best.get('stock_atr_threshold', 0.1)),
            'stock_adx_threshold': float(best.get('stock_adx_threshold', 20.0)),
            'use_pullback': str(best.get('use_pullback', 'False')).lower() == 'true',
            'use_cond_1W': str(best.get('use_cond_1W', 'True')).lower() == 'true',
            'buffer_n': int(best.get('buffer_n', 15)),
            'top_n': int(best.get('top_n', 10)),
            'rebalance_freq': best.get('rebalance_freq', '1M'),
            'stock_mom_period': int(best.get('stock_mom_period', 13)),
            'etf_mom_period': int(best.get('etf_mom_period', 13)),
            'cash_yield': float(best.get('cash_yield', 0.04)),
            'margin_rate': float(best.get('margin_rate', 0.06)),
            'fees': float(best.get('fees', 0.001))
        }
        return config
    except Exception as e:
        import traceback
        st.error(f"❌ Erreur de connexion au Registre MLFlow : {e}")
        st.error(traceback.format_exc())
        
    # Default fallback
    return {
        'sp500_sma_slow': 50, 'sp500_sma_fast': 26, 'stock_sma_fast': 26, 'stock_sma_slow': 50,
        'etf_sma_fast': 12, 'etf_sma_slow': 26, 'stock_atr_threshold': 0.1, 'stock_adx_threshold': 20.0,
        'use_pullback': False, 'use_cond_1W': True, 'buffer_n': 15, 'top_n': 10, 'rebalance_freq': '1M',
        'stock_mom_period': 13, 'etf_mom_period': 13, 'cash_yield': 0.04, 'margin_rate': 0.06, 'fees': 0.001
    }

@st.cache_data(show_spinner="🧠 Calcul de l'Intelligence Financière en cours...", ttl=3600)
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
        return pd.DataFrame(), pd.DataFrame()
        
    allocations = engine.simulate_portfolio(df_sp500, df_etf, df_stocks)
    perf_df = engine.generate_performance(allocations, df_etf, df_stocks, df_sp500)
    
    return perf_df, allocations

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
#st.markdown("Dashboard.")

if run_button or 'perf_df' in st.session_state:
    try:
        config = get_mlflow_config_from_docker()
        st.sidebar.success(f"✅ Modèle MLFlow : Top {config['top_n']} actions | {config['rebalance_freq']}")
        
        perf_df, allocations = compute_backtest(start_date, leverage, config)
        
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
        
        # --- 📈 PLOTLY EQUITY CURVE ---
        st.subheader("Performance Cumulée (Base 100)")
        fig_equity = go.Figure()
        
        fig_equity.add_trace(go.Scatter(
            x=perf_df.index, y=perf_df['Portfolio_Equity'],
            mode='lines',
            name='Portefeuille Momentum',
            line=dict(color='#00f2fe', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 242, 254, 0.1)'
        ))
        
        fig_equity.add_trace(go.Scatter(
            x=perf_df.index, y=perf_df['SP500_Equity'],
            mode='lines',
            name='S&P 500 (Benchmark)',
            line=dict(color='#cbd5e1', width=2, dash='dot')
        ))
        
        fig_equity.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', type='log'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # --- 🥧 CURRENT ALLOCATION ---
        st.write("---")
        st.subheader("Composition actuelle de l'algorithme")
        
        if not allocations.empty:
            last_date = allocations.index[-1]
            last_alloc = allocations.loc[last_date]
            last_alloc = last_alloc[last_alloc > 0.0]
            
            if not last_alloc.empty:
                alloc_df = pd.DataFrame({'Ticker': last_alloc.index, 'Weight': last_alloc.values})
                
                fig_pie = px.pie(
                    alloc_df, values='Weight', names='Ticker',
                    hole=0.6,
                    color_discrete_sequence=px.colors.sequential.Tealgrn
                )
                
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    annotations=[dict(text='Alloc', x=0.5, y=0.5, font_size=24, showarrow=False, font_color='#00f2fe')],
                    showlegend=True,
                    margin=dict(l=0, r=0, b=0, t=0)
                )
                
                st.info(f"Portefeuille en date du : **{last_date.strftime('%d %B %Y')}**")
                colA, colB, colC = st.columns([1, 2, 1])
                with colB:
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("⚠️ L'algorithme est actuellement 100% Cash / Aucun Trade actif.")
                
    except Exception as e:
        st.error(f"❌ Une erreur interne est survenue lors du calcul : {e}")

else:
    st.info("👈 Réglez vos paramètres (Date, Levier) dans le menu de gauche et cliquez sur **Lancer la Simulation** !")
