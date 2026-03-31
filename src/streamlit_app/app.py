import os
import sys
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import datetime

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
    font-size: 1.2rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
    margin-bottom: 10px;
}
.big-font {
    font-size: 4rem !important;
    font-weight: 900;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin: 0;
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

@st.cache_data(show_spinner="🧠 Calcul de l'Intelligence Financière en cours...", ttl=3600)
def compute_backtest(start_date, leverage):
    spark = init_spark()
    engine = RegimeSwitchingMomentumBacktester(
        start_date=start_date.strftime("%Y-%m-%d"), 
        leverage=leverage
    )
    
    # Exécution complète de la stratégie
    df_sp500 = engine.get_sp500_regime()
    df_etf, df_stocks = engine.load_and_prep_data(spark, Paths.DATA_RAW_ETF_WEEKLY_GOLD, Paths.DATA_RAW_2B_WEEKLY_GOLD)
    
    if df_stocks.empty and df_etf.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    allocations = engine.simulate_portfolio(df_sp500, df_etf, df_stocks)
    perf_df = engine.generate_performance(allocations, df_etf, df_stocks, df_sp500)
    
    return perf_df, allocations

# --- SIDEBAR CONFIG ---
st.sidebar.title("⚙️ Paramètres de Backtest")

default_start = datetime.date(2010, 1, 1)
min_date = datetime.date(2000, 1, 1)
max_date = datetime.date.today()

start_date = st.sidebar.date_input("Date de début", value=default_start, min_value=min_date, max_value=max_date)
leverage = st.sidebar.slider("Niveau de Levier (x)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

run_button = st.sidebar.button("🚀 Lancer la Simulation", use_container_width=True)

# --- MAIN PAGE CONFIG ---
st.title("✨ Momentum AI : Tableau de Bord Stratégique")
st.markdown("Comparatif de la performance du Portefeuille Vectorisé contre le **S&P 500**.")

if run_button or 'perf_df' in st.session_state:
    try:
        perf_df, allocations = compute_backtest(start_date, leverage)
        
        # Sauver dans le state pour éviter le rechargement forcé
        st.session_state['perf_df'] = perf_df
        st.session_state['allocations'] = allocations
        
        if perf_df.empty:
            st.error("Aucune donnée générée. Vérifiez si les tables GOLD sont remplies ou si les dates sont cohérentes.")
            st.stop()
            
        # --- CALCUL DES KPIs ---
        # Rendement total
        total_ret_strat = perf_df['Portfolio_Equity'].iloc[-1] - 100.0
        total_ret_sp500 = perf_df['SP500_Equity'].iloc[-1] - 100.0
        
        # Max Drawdown
        roll_max = perf_df['Portfolio_Equity'].cummax()
        drawdown = (perf_df['Portfolio_Equity'] / roll_max) - 1.0
        max_drawdown = drawdown.min() * 100.0
        
        # --- DISPLAY KPIs (GROS NOMBRES) ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Performance AI ({leverage}x)</div>
                <div class="big-font\">{total_ret_strat:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Performance S&P 500</div>
                <div class="big-font sp500">{total_ret_sp500:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Max Drawdown Stratégie</div>
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
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
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
