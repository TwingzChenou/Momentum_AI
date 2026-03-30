import json
import re

path = "notebooks/new_exploration/explorating_DNN_TOP_Classification.ipynb"
with open(path, "r") as f:
    notebook = json.load(f)

# Modify Cell 22
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and "dnn_models = train_dnn_ensemble(" in "".join(cell.get('source', [])):
        source = "".join(cell['source'])
        
        old_dnn_train = """            # Entraînement DNN
            print("  -> Training DNN Ensemble...")
            dnn_models = train_dnn_ensemble(X_train, y_train, X_val, y_val, scaler, num_models=DNN_MODELS, epochs=DNN_EPOCHS, patience=3)
            dnn_preds = predict_dnn_ensemble(dnn_models, scaler, X_test)
            
            # --- SUIVI ANNÉE PAR ANNÉE ---
            r2_dnn_year = float(calculate_spearman_ic(y_test.values, dnn_preds))
            
            # On force le cast en int() pour le step !
            mlflow.log_metric("Yearly_DNN_R2_OOS", r2_dnn_year, step=int(test_year))
            
            print(f"  -> DNN R2 {test_year}: {r2_dnn_year:.4f}")
            
            # Stockage des prédictions pour cette année
            test_df_year = df.loc[test_idx].copy()
            test_df_year['dnn_pred'] = dnn_preds"""
            
        new_rf_train = old_dnn_train + """
            
            # Entraînement Random Forest
            print("  -> Training Random Forest...")
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf_model.fit(X_train, y_train)
            rf_preds = rf_model.predict(X_test)
            r2_rf_year = float(calculate_spearman_ic(y_test.values, rf_preds))
            mlflow.log_metric("Yearly_RF_R2_OOS", r2_rf_year, step=int(test_year))
            print(f"  -> RF R2 {test_year}: {r2_rf_year:.4f}")
            test_df_year['rf_pred'] = rf_preds"""
        
        if old_dnn_train in source:
            source = source.replace(old_dnn_train, new_rf_train)
            print("Successfully replaced RF train part")
        else:
            print("Warning: old_dnn_train not found")

        old_global_r2 = """        # Évaluation R^2 globale sur toute la période OOS
        r2_dnn_global = calculate_spearman_ic(final_test_df['target_y'].values, final_test_df['dnn_pred'].values)
        
        mlflow.log_metric("Global_DNN_R2_OOS", r2_dnn_global)
        
        # Passage au simulateur de portefeuille
        print("Running Backtest for DNN Strategy...")
        bt_dnn, metrics_dnn = backtest_portfolio(final_test_df, 'dnn_pred', transaction_cost=TRANS_COST, top_n=TOP)
        
        # 4. AFFICHAGE ET SAUVEGARDE FINALE
        if not bt_dnn.empty:
            
            # Sauvegarde des métriques financières dans MLflow en forçant le type float()
            mlflow.log_metrics({f"DNN_{k}": float(v) for k, v in metrics_dnn.items()})
            
            # Sauvegarde aussi le R2 global en forçant le float() (juste au cas où)
            mlflow.log_metric("Global_DNN_R2_OOS", float(r2_dnn_global))
            
            # Formatage du joli tableau pour la console
            metrics_table = pd.DataFrame({
                "Métrique": ["Total Return", "CAGR (Annualisé)", "Sharpe Ratio", "Max Drawdown"],
                "DNN Ensemble": [f"{metrics_dnn['Total_Return']*100:.2f}%", f"{metrics_dnn['CAGR']*100:.2f}%", f"{metrics_dnn['Sharpe_Ratio']:.2f}", f"{metrics_dnn['Max_Drawdown']*100:.2f}%"]
            })"""

        new_global_r2 = old_global_r2 + """
        
        r2_rf_global = calculate_spearman_ic(final_test_df['target_y'].values, final_test_df['rf_pred'].values)
        mlflow.log_metric("Global_RF_R2_OOS", r2_rf_global)
        print("Running Backtest for RF Strategy...")
        bt_rf, metrics_rf = backtest_portfolio(final_test_df, 'rf_pred', transaction_cost=TRANS_COST, top_n=TOP)
        
        if not bt_rf.empty:
            mlflow.log_metrics({f"RF_{k}": float(v) for k, v in metrics_rf.items()})
            mlflow.log_metric("Global_RF_R2_OOS", float(r2_rf_global))
            metrics_table["Random Forest"] = [f"{metrics_rf['Total_Return']*100:.2f}%", f"{metrics_rf['CAGR']*100:.2f}%", f"{metrics_rf['Sharpe_Ratio']:.2f}", f"{metrics_rf['Max_Drawdown']*100:.2f}%"]"""
        
        if old_global_r2 in source:
            source = source.replace(old_global_r2, new_global_r2)
            print("Successfully replaced RF global part")
        else:
            print("Warning: old_global_r2 not found")

        # Split back to list of lines retaining \n
        lines = source.splitlines()
        cell['source'] = [line + '\\n' for line in lines[:-1]] + [lines[-1]] if lines else []
        cell['source'] = [s.replace('\\n', '\n') for s in cell['source']]

# Modify Cell 24
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and "bt_dnn, metrics_dnn = backtest_portfolio(final_test_df, 'dnn_pred', transaction_cost=0.001, top_n=5)" in "".join(cell.get('source', [])):
        new_source = '''import pandas as pd
import matplotlib.pyplot as plt

print("Running Backtest for DNN and RF Strategy...")
bt_dnn, metrics_dnn = backtest_portfolio(final_test_df, 'dnn_pred', transaction_cost=0.001, top_n=5)
bt_rf, metrics_rf = backtest_portfolio(final_test_df, 'rf_pred', transaction_cost=0.001, top_n=5)

if not bt_dnn.empty and not bt_rf.empty:
    metrics_table = pd.DataFrame({
        "Métrique": ["Total Return", "CAGR (Annualisé)", "Sharpe Ratio", "Max Drawdown"],
        "DNN Ensemble": [f"{metrics_dnn['Total_Return']*100:.2f}%", f"{metrics_dnn['CAGR']*100:.2f}%", f"{metrics_dnn['Sharpe_Ratio']:.2f}", f"{metrics_dnn['Max_Drawdown']*100:.2f}%"],
        "Random Forest": [f"{metrics_rf['Total_Return']*100:.2f}%", f"{metrics_rf['CAGR']*100:.2f}%", f"{metrics_rf['Sharpe_Ratio']:.2f}", f"{metrics_rf['Max_Drawdown']*100:.2f}%"]
    })
    
    print("\\n" + "="*75)
    print("📊 TABLEAU DES PERFORMANCES FINALES (HEBDOMADAIRE - AVEC FRAIS 0.1%)")
    print("="*75)
    print(metrics_table.to_string(index=False))
    print("="*75 + "\\n")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(bt_dnn['date'], bt_dnn['cum_return'], label=f"DNN (Sharpe: {metrics_dnn['Sharpe_Ratio']:.2f})", color='green', linewidth=2)
    ax1.plot(bt_rf['date'], bt_rf['cum_return'], label=f"Random Forest (Sharpe: {metrics_rf['Sharpe_Ratio']:.2f})", color='blue', linewidth=2)
    
    ax1.plot(bt_dnn['date'], bt_dnn['cum_benchmark'], label=f"Benchmark S&P 500 (Sharpe: {metrics_dnn['Bench_Sharpe']:.2f})", color='black', linestyle='--', linewidth=2)
    
    ax1.axhline(y=1.0, color='red', linestyle=':', alpha=0.5)
    ax1.set_yscale('log')
    ax1.set_title('Walk-Forward Cumulative Returns vs Benchmark S&P 500 (WEEKLY - LOG SCALE)')
    ax1.set_ylabel('Cumulative Return (Log Scale, 1.0 = Initial Capital)')
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    
    ax2.fill_between(bt_dnn['date'], bt_dnn['outperformance'], 0, where=(bt_dnn['outperformance'] >= 0), color='green', alpha=0.2, label='DNN Outperformance')
    ax2.fill_between(bt_dnn['date'], bt_dnn['outperformance'], 0, where=(bt_dnn['outperformance'] < 0), color='red', alpha=0.2, label='DNN Underperformance')
    ax2.fill_between(bt_rf['date'], bt_rf['outperformance'], 0, where=(bt_rf['outperformance'] >= 0), color='blue', alpha=0.2, label='RF Outperformance')
    ax2.fill_between(bt_rf['date'], bt_rf['outperformance'], 0, where=(bt_rf['outperformance'] < 0), color='orange', alpha=0.2, label='RF Underperformance')
    
    ax2.axhline(y=0.0, color='black', linewidth=1)
    ax2.set_title('Outperformance Spread (Strategy - Benchmark)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Spread (Linear)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    plot_path = "global_cumulative_returns_vs_benchmark.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    
    plt.show()
else:
    print("Le backtest du DNN ou RF est vide.")
'''
        lines = new_source.splitlines()
        cell['source'] = [line + '\\n' for line in lines[:-1]] + [lines[-1]] if lines else []
        cell['source'] = [s.replace('\\n', '\n') for s in cell['source']]
        print("Successfully replaced Cell 24")

with open(path, "w") as f:
    json.dump(notebook, f, indent=1)
    
print("Notebook updated!")
