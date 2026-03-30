import re

with open('scripts/generate_exploration_nb.py', 'r') as f:
    content = f.read()

# we'll slice out from "    # 2. Expanding Window Time-Series Split"
# to "    nb['cells'].append(nbf.v4.new_code_cell(code_run))"

start_str = "    # 2. Expanding Window Time-Series Split"
end_str = "    nb['cells'].append(nbf.v4.new_code_cell(code_run))"

start_idx = content.find(start_str)
end_idx = content.find(end_str) + len(end_str)

if start_idx == -1 or end_idx < len(end_str):
    print("Could not find boundaries")
    exit(1)

new_block = """    # 2. Algorithm Model Architectures
    markdown_models = "## 2. Algorithm Model Architectures"
    nb['cells'].append(nbf.v4.new_markdown_cell(markdown_models))
    
    # Feature engineering
    code_features = \"\"\"# Define features. Exclude identifiers and target.
features = ['retvol', 'maxret', 'ill', 'mom1m', 'beta', 'mom6m', 'mom12m']
# Ensure no NaNs in features
df = df.dropna(subset=features)

# Reset index so that train/val/test splits (which use df.index) match correctly
df = df.reset_index(drop=True)
\"\"\"
    nb['cells'].append(nbf.v4.new_code_cell(code_features))

    # RF
    markdown_rf = "### Random Forest (GridSearchCV + MLflow Autolog)"
    nb['cells'].append(nbf.v4.new_markdown_cell(markdown_rf))

    code_rf = \"\"\"def train_rf_cv(X_train, y_train):
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RF_GridSearch", nested=True):
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 6],
            'max_features': ['sqrt', 'log2']
        }
        
        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best RF Params: {grid_search.best_params_}")
        return grid_search.best_estimator_
\"\"\"
    nb['cells'].append(nbf.v4.new_code_cell(code_rf))

    # DNN
    markdown_dnn = "### Deep Neural Network (KerasTuner + MLflow Autolog + Ensembling)"
    nb['cells'].append(nbf.v4.new_markdown_cell(markdown_dnn))

    code_dnn = \"\"\"def build_model(hp, input_dim):
    \"\"\"KerasTuner model building function\"\"\"
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    # Tune the number of units in the first Dense layer
    hp_units1 = hp.Int('units_1', min_value=16, max_value=64, step=16)
    model.add(layers.Dense(units=hp_units1, kernel_regularizer=regularizers.l1(1e-5)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Tune the number of units in the second Dense layer
    hp_units2 = hp.Int('units_2', min_value=8, max_value=32, step=8)
    model.add(layers.Dense(units=hp_units2, kernel_regularizer=regularizers.l1(1e-5)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Third hidden layer (fixed small)
    model.add(layers.Dense(8, kernel_regularizer=regularizers.l1(1e-5)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Output layer
    model.add(layers.Dense(1))
    
    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mse',
                  metrics=['mae'])
    return model

def train_dnn_ensemble(X_train, y_train, X_val, y_val, scaler, num_models=5, epochs=50, patience=5):
    mlflow.tensorflow.autolog(disable=True) # Avoid logging every single tuner step if it's too noisy, but we will log final models
    
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    input_dim = X_train_scaled.shape[1]
    
    # 1. Hyperparameter Search
    print("Starting KerasTuner Search...")
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, input_dim),
        objective='val_loss',
        max_epochs=20, # Limit epochs for demonstration
        directory='keras_tuner_dir',
        project_name='momentum_dnn',
        overwrite=True
    )
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    tuner.search(X_train_scaled, y_train, epochs=20, validation_data=(X_val_scaled, y_val), callbacks=[early_stopping], verbose=0)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best HPs found: Layer1: {best_hps.get('units_1')}, Layer2: {best_hps.get('units_2')}, LR: {best_hps.get('learning_rate')}")
    
    # 2. Train Ensemble with Best HPs
    models = []
    
    mlflow.tensorflow.autolog() # Re-enable for the final ensemble training
    
    for i in range(num_models):
        with mlflow.start_run(run_name=f"DNN_Ensemble_Member_{i+1}", nested=True):
            # Log best HPs
            mlflow.log_params(best_hps.values)
            
            # Set unique seed per member
            tf.random.set_seed(42 + i)
            
            model = tuner.hypermodel.build(best_hps)
            history = model.fit(
                X_train_scaled, y_train,
                epochs=epochs,
                validation_data=(X_val_scaled, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            val_loss = min(history.history['val_loss'])
            print(f"Trained DNN component {i+1}/{num_models} - Best Val Loss: {val_loss:.6f}")
            models.append(model)
        
    return models

def predict_dnn_ensemble(models, scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    preds = []
    for model in models:
        pred = model.predict(X_test_scaled, verbose=0).flatten()
        preds.append(pred)
        
    # Average predictions
    return np.mean(preds, axis=0)
\"\"\"
    nb['cells'].append(nbf.v4.new_code_cell(code_dnn))

    # 3. Expanding Window Time-Series Split
    markdown_split = "## 3. Expanding Window Time-Series Split"
    nb['cells'].append(nbf.v4.new_markdown_cell(markdown_split))

    code_split = \"\"\"def get_train_val_test_splits(df, initial_train_years=3, val_years=2, test_years=1):
    \"\"\"
    Yields train, val, test indices for an expanding window split.
    \"\"\"
    years = sorted(df['year'].unique())
    start_year = years[0]
    
    current_test_year = start_year + initial_train_years + val_years
    
    splits = []
    
    while current_test_year <= years[-1]:
        train_end = current_test_year - val_years - 1
        val_end = current_test_year - 1
        
        train_idx = df[df['year'] <= train_end].index
        val_idx = df[(df['year'] > train_end) & (df['year'] <= val_end)].index
        test_idx = df[df['year'] == current_test_year].index
        
        splits.append((train_idx, val_idx, test_idx, current_test_year))
        current_test_year += 1
        
    return splits

splits = get_train_val_test_splits(df)
print(f"Total expanding window splits: {len(splits)}")
for i, (tr, val, ts, yr) in enumerate(splits):
    print(f"Split {i+1} | Test Year: {yr} | Train: {len(tr)} | Val: {len(val)} | Test: {len(ts)}")
\"\"\"
    nb['cells'].append(nbf.v4.new_code_cell(code_split))

    # 4. Evaluation Metrics
    markdown_metrics = "## 4. Evaluation Metrics ($R^2_{OOS}$)"
    nb['cells'].append(nbf.v4.new_markdown_cell(markdown_metrics))

    code_metrics = \"\"\"def calculate_r2_oos(y_true, y_pred):
    \"\"\"
    Out-of-Sample R^2.
    Denominator uses 0 as the prediction benchmark (predicting exactly zero excess return).
    \"\"\"
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - 0)**2)
    return 1 - (numerator / denominator)
\"\"\"
    nb['cells'].append(nbf.v4.new_code_cell(code_metrics))

    # 5. Backtesting Portfolio Construction
    markdown_bt = "## 5. Backtesting Portfolio Construction (Decile Sorting)"
    nb['cells'].append(nbf.v4.new_markdown_cell(markdown_bt))
    
    code_bt = \"\"\"def backtest_portfolio(test_df, predictions_col):
    \"\"\"
    Simulates a Long/Short Value-Weighted portfolio and calculates performance metrics.
    \"\"\"
    results = []
    
    for (year, month), month_data in test_df.groupby(['year', 'month']):
        if len(month_data) < 10: 
            continue
            
        try:
            month_data = month_data.copy()
            month_data['decile'] = pd.qcut(month_data[predictions_col], 10, labels=False) + 1
            
            if 'dollar_volume' not in month_data.columns:
                # Fallback to equal weighting if volume/adjClose are missing
                if 'volume' in month_data.columns and 'adjClose' in month_data.columns:
                    month_data['dollar_volume'] = month_data['volume'] * month_data['adjClose']
                else:
                    month_data['dollar_volume'] = 1.0
            
            long_portfolio = month_data[month_data['decile'] == 10]
            short_portfolio = month_data[month_data['decile'] == 1]
            
            if long_portfolio['dollar_volume'].sum() == 0 or short_portfolio['dollar_volume'].sum() == 0:
                continue
                
            long_weights = long_portfolio['dollar_volume'] / long_portfolio['dollar_volume'].sum()
            short_weights = short_portfolio['dollar_volume'] / short_portfolio['dollar_volume'].sum()
            
            ret_long = np.sum(long_weights * long_portfolio['target_y'])
            ret_short = np.sum(short_weights * short_portfolio['target_y'])
            
            portfolio_return = ret_long - ret_short
            
            results.append({
                'date': pd.Timestamp(year=year, month=month, day=1),
                'ret_long': ret_long,
                'ret_short': ret_short,
                'portfolio_return': portfolio_return,
                'num_long': len(long_portfolio),
                'num_short': len(short_portfolio)
            })
        except Exception as e:
            pass
            
    bt_df = pd.DataFrame(results)
    if bt_df.empty:
        return bt_df, {}
        
    bt_df['cum_return'] = (1 + bt_df['portfolio_return']).cumprod()
    
    # Calculate Metrics
    n_years = len(bt_df) / 12.0
    cagr = (bt_df['cum_return'].iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    mean_ret = bt_df['portfolio_return'].mean()
    std_ret = bt_df['portfolio_return'].std()
    sharpe = (mean_ret / std_ret) * np.sqrt(12) if std_ret > 0 else 0
    
    rolling_max = bt_df['cum_return'].cummax()
    drawdown = (bt_df['cum_return'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    metrics = {
        'CAGR': cagr,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown': max_dd
    }
    
    print(f"--- Metrics ({predictions_col}) ---")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print("-" * 30)
    
    return bt_df, metrics
\"\"\"
    nb['cells'].append(nbf.v4.new_code_cell(code_bt))

    # 6. Run the Pipeline
    markdown_run = "## 6. Run the Pipeline over the Expanding Window"
    nb['cells'].append(nbf.v4.new_markdown_cell(markdown_run))
    
    code_run = \"\"\"# For demonstration, we'll just run on the FIRST valid split. 

if len(splits) > 0:
    train_idx, val_idx, test_idx, test_year = splits[0]
    
    print(f"Running pipeline for Test Year {test_year}...")
    
    X_train, y_train = df.loc[train_idx, features], df.loc[train_idx, 'target_y']
    X_val, y_val = df.loc[val_idx, features], df.loc[val_idx, 'target_y']
    X_test, y_test = df.loc[test_idx, features], df.loc[test_idx, 'target_y']
    
    # Setup scaler for DNN
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    with mlflow.start_run(run_name=f"Expanding_Window_{test_year}") as parent_run:
        # 1. Train RF
        print("Training RF...")
        rf_model = train_rf_cv(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        
        # 2. Train DNN Ensemble
        print("Training DNN Ensemble...")
        dnn_models = train_dnn_ensemble(X_train, y_train, X_val, y_val, scaler, num_models=3, epochs=20, patience=3)
        dnn_preds = predict_dnn_ensemble(dnn_models, scaler, X_test)
        
        # 3. Evaluate R^2 OOS
        r2_rf = calculate_r2_oos(y_test.values, rf_preds)
        r2_dnn = calculate_r2_oos(y_test.values, dnn_preds)
        
        print(f"RF R^2 OOS: {r2_rf:.5f}")
        print(f"DNN R^2 OOS: {r2_dnn:.5f}")
        
        mlflow.log_metric("RF_R2_OOS", r2_rf)
        mlflow.log_metric("DNN_R2_OOS", r2_dnn)
        
        # 4. Backtest
        test_df = df.loc[test_idx].copy()
        test_df['rf_pred'] = rf_preds
        test_df['dnn_pred'] = dnn_preds
        
        print("Running Backtest for RF Strategy...")
        bt_rf, metrics_rf = backtest_portfolio(test_df, 'rf_pred')
        
        print("Running Backtest for DNN Strategy...")
        bt_dnn, metrics_dnn = backtest_portfolio(test_df, 'dnn_pred')
        
        # Compute Cumulative Returns
        if not bt_rf.empty and not bt_dnn.empty:
            # Log total returns and new metrics
            total_ret_rf = bt_rf['cum_return'].iloc[-1] - 1
            total_ret_dnn = bt_dnn['cum_return'].iloc[-1] - 1
            mlflow.log_metric("RF_Total_Return", total_ret_rf)
            mlflow.log_metric("DNN_Total_Return", total_ret_dnn)
            
            mlflow.log_metric("RF_CAGR", metrics_rf.get('CAGR', 0))
            mlflow.log_metric("RF_Sharpe", metrics_rf.get('Sharpe_Ratio', 0))
            mlflow.log_metric("RF_MaxDD", metrics_rf.get('Max_Drawdown', 0))
            
            mlflow.log_metric("DNN_CAGR", metrics_dnn.get('CAGR', 0))
            mlflow.log_metric("DNN_Sharpe", metrics_dnn.get('Sharpe_Ratio', 0))
            mlflow.log_metric("DNN_MaxDD", metrics_dnn.get('Max_Drawdown', 0))
            
            plt.figure(figsize=(10,6))
            plt.plot(bt_rf['date'], bt_rf['cum_return'], label='RF Long/Short')
            plt.plot(bt_dnn['date'], bt_dnn['cum_return'], label='DNN Long/Short')
            plt.title('Cumulative Returns of AI Spread Strategies')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            
            # Save plot to MLflow
            plot_path = "cumulative_returns.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            
            plt.show()
        else:
            print("Backtest yielded empty results (perhaps not enough data to form deciles).")
          
else:
    print("Not enough data to form a split (require initial_train_years + val_years + 1). Check your timeframe.")
\"\"\"
    nb['cells'].append(nbf.v4.new_code_cell(code_run))"""

new_content = content[:start_idx] + new_block + content[end_idx:]

with open('scripts/generate_exploration_nb.py', 'w') as f:
    f.write(new_content)

print("Replacement successful")
