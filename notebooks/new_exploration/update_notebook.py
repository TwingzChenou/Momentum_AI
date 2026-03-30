import nbformat

def update_notebook():
    nb_path = "/Users/forget/Desktop/Project_Momentum_AI/notebooks/new_exploration/explorating_Regression.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Find the imports cell
    for cell in nb.cells:
        if cell.cell_type == "code" and "from sklearn.ensemble import RandomForestRegressor" in cell.source:
            # We will insert new imports before this line if not present
            if "import xgboost" not in cell.source:
                cell.source = cell.source.replace("from sklearn.ensemble import RandomForestRegressor", 
"""from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor""")

        # Find the training loop cell
        if cell.cell_type == "code" and "def train_dnn_ensemble" in cell.source:
            # First, add the training functions for ML models right before `train_dnn_ensemble`
            if "def train_ols" not in cell.source:
                ml_funcs = """def train_ols(X_train_scaled, y_train):
    with mlflow.start_run(run_name="OLS", nested=True):
        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        return ols

def train_lasso(X_train_scaled, y_train):
    with mlflow.start_run(run_name="Lasso", nested=True):
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_train_scaled, y_train)
        return lasso

def train_ridge(X_train_scaled, y_train):
    with mlflow.start_run(run_name="Ridge", nested=True):
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train_scaled, y_train)
        return ridge

def train_rf(X_train_scaled, y_train):
    with mlflow.start_run(run_name="RandomForest", nested=True):
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(X_train_scaled, y_train)
        return rf

def train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val):
    with mlflow.start_run(run_name="XGBoost", nested=True):
        xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42)
        xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
        return xgb_model

def train_lightgbm(X_train_scaled, y_train, X_val_scaled, y_val):
    with mlflow.start_run(run_name="LightGBM", nested=True):
        lgbm = lgb.LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42)
        lgbm.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)])
        return lgbm

def train_catboost(X_train_scaled, y_train, X_val_scaled, y_val):
    with mlflow.start_run(run_name="CatBoost", nested=True):
        cb = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, random_seed=42, verbose=False, thread_count=-1)
        cb.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val), use_best_model=True)
        return cb

def predict_ml_models(models, scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(X_test_scaled)
    return preds
"""
                cell.source = cell.source.replace("def train_dnn_ensemble", ml_funcs + "\ndef train_dnn_ensemble")
        
        # Then, modify the backtesting cell loop
        if cell.cell_type == "code" and "STRATEGY_FREQ =" in cell.source and "DNN_MODELS = 5" in cell.source:
            if "ml_models =" not in cell.source:
                # Add training and logging
                predict_block = """            # Entraînement ML Models
            print("  -> Training ML Models (OLS, Lasso, Ridge, RF, XGB, LGBM, CatBoost)...")
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            ml_models = {}
            ml_models['OLS'] = train_ols(X_train_scaled, y_train)
            ml_models['Lasso'] = train_lasso(X_train_scaled, y_train)
            ml_models['Ridge'] = train_ridge(X_train_scaled, y_train)
            ml_models['RF'] = train_rf(X_train_scaled, y_train)
            ml_models['XGB'] = train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
            ml_models['LGBM'] = train_lightgbm(X_train_scaled, y_train, X_val_scaled, y_val)
            ml_models['CatBoost'] = train_catboost(X_train_scaled, y_train, X_val_scaled, y_val)
            
            ml_preds = predict_ml_models(ml_models, scaler, X_test)
            
            # --- SUIVI ANNÉE PAR ANNÉE (ML) ---
            for name, preds in ml_preds.items():
                r2 = float(calculate_spearman_ic(y_test.values, preds))
                mlflow.log_metric(f"Yearly_{name}_R2_OOS", r2, step=int(test_year))
                print(f"  -> {name} R2 {test_year}: {r2:.4f}")
                
            test_df_year = df.loc[test_idx].copy()
            test_df_year['dnn_pred'] = dnn_preds
            for name, preds in ml_preds.items():
                test_df_year[f'{name.lower()}_pred'] = preds"""
                
                cell.source = cell.source.replace("""            # Stockage des prédictions pour cette année
            test_df_year = df.loc[test_idx].copy()
            test_df_year['dnn_pred'] = dnn_preds""", predict_block)

                # Now, modify the bottom block (Running Backtest for DNN Strategy...)
                backtest_block = """        # Passage au simulateur de portefeuille pour TOUS LES MODÈLES
        print("Running Backtest for all Models Strategy...")
        model_names = ['dnn'] + [name.lower() for name in ml_models.keys()]
        
        all_metrics = []
        
        for name in model_names:
            print(f"Backtesting {name.upper()}...")
            bt, metrics = backtest_portfolio(final_test_df, f'{name}_pred', transaction_cost=TRANS_COST, top_n=TOP)
            
            if not bt.empty:
                # Log metrics with prefix
                mlflow.log_metrics({f"{name.upper()}_{k}": float(v) for k, v in metrics.items()})
                
                global_r2 = calculate_spearman_ic(final_test_df['target_y'].values, final_test_df[f'{name}_pred'].values)
                mlflow.log_metric(f"Global_{name.upper()}_R2_OOS", float(global_r2))
                
                all_metrics.append({
                    "Model": name.upper(),
                    "Total Return": f"{metrics['Total_Return']*100:.2f}%", 
                    "CAGR (Annualisé)": f"{metrics['CAGR']*100:.2f}%", 
                    "Sharpe Ratio": f"{metrics['Sharpe_Ratio']:.2f}", 
                    "Max Drawdown": f"{metrics['Max_Drawdown']*100:.2f}%"
                })
        
        if all_metrics:
            metrics_table = pd.DataFrame(all_metrics)
            print("\n" + "="*55)
            print("📊 TABLEAU DES PERFORMANCES FINALES (AVEC FRAIS)")
            print("="*55)
            print(metrics_table.to_string(index=False))
            print("="*55 + "\n")
            
            with open("performance_metrics.txt", "w") as f:
                f.write(metrics_table.to_string(index=False))
            mlflow.log_artifact("performance_metrics.txt")"""
            
                # We replace the bottom block that only ran DNN
                import re
                cell.source = re.sub(r'# Passage au simulateur de portefeuille.*?print\("Backtest yielded empty results \(perhaps not enough data to form deciles\)\."\)', backtest_block, cell.source, flags=re.DOTALL)

        # Finally, visualization
        if cell.cell_type == "code" and "bt_dnn, metrics_dnn = backtest_portfolio" in cell.source:
             if "bt_rf" not in cell.source:
                 viz_block = """print("Running Backtest for all Models Strategies Visualization...")

models_to_test = ['dnn', 'rf', 'xgb', 'lgbm', 'catboost']
bt_results = {}
metrics_results = {}

for name in models_to_test:
    bt, metrics = backtest_portfolio(final_test_df, f'{name}_pred', transaction_cost=0.001, top_n=5)
    if not bt.empty:
        bt_results[name] = bt
        metrics_results[name] = metrics

if bt_results:
    # 1. TABLEAU DES PERFORMANCES
    all_metrics = []
    for name, metrics in metrics_results.items():
         all_metrics.append({
            "Model": name.upper(),
            "Total Return": f"{metrics['Total_Return']*100:.2f}%", 
            "CAGR (Annualisé)": f"{metrics['CAGR']*100:.2f}%", 
            "Sharpe Ratio": f"{metrics['Sharpe_Ratio']:.2f}", 
            "Max Drawdown": f"{metrics['Max_Drawdown']*100:.2f}%"
        })
    metrics_table = pd.DataFrame(all_metrics)
    
    print("\n" + "="*55)
    print("📊 TABLEAU DES PERFORMANCES FINALES (HEBDOMADAIRE - AVEC FRAIS 0.1%)")
    print("="*55)
    print(metrics_table.to_string(index=False))
    print("="*55 + "\n")

    # 2. GRAPHIQUES
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    colors = ['green', 'blue', 'orange', 'purple', 'brown']
    
    for i, name in enumerate(models_to_test):
        if name in bt_results:
            ax1.plot(bt_results[name]['date'], bt_results[name]['cum_return'], 
                     label=f"{name.upper()} (Sharpe: {metrics_results[name]['Sharpe_Ratio']:.2f})", 
                     color=colors[i % len(colors)], linewidth=2, alpha=0.8)
    
    # Benchmark is same for all, just take from first
    first_name = list(bt_results.keys())[0]
    ax1.plot(bt_results[first_name]['date'], bt_results[first_name]['cum_benchmark'], 
             label=f"Benchmark S&P 500 (Sharpe: {metrics_results[first_name]['Bench_Sharpe']:.2f})", 
             color='black', linestyle='--', linewidth=2)
             
    ax1.axhline(y=1.0, color='red', linestyle=':', alpha=0.5)
    ax1.set_yscale('log')
    ax1.set_title('Walk-Forward Cumulative Returns vs Benchmark S&P 500 (WEEKLY - LOG SCALE)')
    ax1.set_ylabel('Cumulative Return (Log Scale, 1.0 = Initial Capital)')
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    
    # Base spread on DNN
    if 'dnn' in bt_results:
        ax2.fill_between(bt_results['dnn']['date'], bt_results['dnn']['outperformance'], 0, 
                         where=(bt_results['dnn']['outperformance'] >= 0), color='green', alpha=0.3, label='DNN Outperformance')
        ax2.fill_between(bt_results['dnn']['date'], bt_results['dnn']['outperformance'], 0, 
                         where=(bt_results['dnn']['outperformance'] < 0), color='red', alpha=0.3, label='DNN Underperformance')
    
    ax2.axhline(y=0.0, color='black', linewidth=1)
    ax2.set_title('Outperformance Spread (DNN - Benchmark)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Spread (Linear)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "global_cumulative_returns_vs_benchmark_all.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.show()"""
                 cell.source = viz_block

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        print("Successfully updated notebook with ML models.")

if __name__ == "__main__":
    update_notebook()
