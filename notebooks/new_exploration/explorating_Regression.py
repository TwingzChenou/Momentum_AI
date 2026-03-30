#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Backtesting Pipeline (TensorFlow & MLflow)
# This notebook implements an Expanding Window Time-Series Split, Random Forest (with GridSearchCV), Deep Neural Network (TensorFlow/Keras + KerasTuner), and Decile Sorting Backtesting.
# All model training and hyperparameter searches are tracked using MLflow.
# 

# In[1]:


import IPython.core.pylabtools
import IPython.core.pylabtools
import os
import sys
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import mlflow
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from scipy.stats import spearmanr

# Ask TensorFlow to list all available physical GPUs
gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print(f"✅ M3 Pro GPU ACTIVATED! Found: {gpu_devices}")
    # Optional: Set memory growth to prevent TF from hoarding all unified memory
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("❌ GPU not found. TensorFlow is falling back to the CPU.")


# ## 1. Setup & Data Loading (with MLflow)

# In[2]:


# Fix random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Spark setup
from dotenv import load_dotenv
os.chdir(os.path.abspath(os.path.join(os.getcwd(), '../../')))
sys.path.append(os.getcwd())

from src.common.setup_spark import create_spark_session
from config.config_spark import Paths

# MLflow Setup
mlflow.set_tracking_uri("sqlite:///mlflow.db") # Local SQLite database for tracking
experiment_name = "SP500_Momentum_Backtest"
mlflow.set_experiment(experiment_name)
print(f"MLflow Experiment set to: {experiment_name}")

spark = create_spark_session()
print("Spark Session created.")

# Load Data
df_gold = spark.read.format("delta").load(Paths.SP500_MOMENTUM_CRASH_WEEKLY_GOLD)
df_gold.createOrReplaceTempView("gold_prices")

df = df_gold.toPandas()

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.weekday

#df = df[df['bull_market']==1]

print(f"Data loaded: {df.shape}")
print(f"Years: {df['year'].unique().min()}")


# In[3]:


df.describe()


# ### Calculate Target Variable: 1-Month Ahead Expected Excess Return

# In[4]:


import pandas as pd
import numpy as np

def create_advanced_target(df, forward_weeks=4):
    """
    Calculates a rolling forward return on weekly data, then converts it 
    into a Cross-Sectional Z-Score to isolate 'Alpha' from 'Beta'.
    """
    df = df.copy()
    
    # 0. S'assurer des formats et du tri initial (CRITIQUE pour le shift)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['symbol', 'date'])
    
    # 1. Calculer le rendement futur sur N semaines (ex: 4 semaines = 1 mois)
    # On divise le prix dans 4 semaines par le prix d'aujourd'hui
    df['return_forward'] = df.groupby('symbol')['adjClose'].shift(-forward_weeks) / df['adjClose'] - 1

    df['return_forward'] = df['return_forward'].clip(lower=-0.50, upper=1.00)
    
    # 2. Nettoyer les extrêmes absolus (les penny stocks qui font +1000%)
    df['target_y'] = df.groupby('date')['return_forward'].transform(lambda x: (x - x.mean()) / x.std(ddof=1))
    
    # 5. Nettoyer le dataset final
    df = df.drop(columns=['return_forward'])
    df = df.dropna(subset=['target_y'])
    
    return df

df = create_advanced_target(df, forward_weeks=4)


# ## 4. Algorithm Model Architectures

# In[5]:


colonne_list = df.columns.tolist()
colonnes_texte = df.select_dtypes(exclude=['number']).columns.tolist()
colonnes_biais = ['date', 'volume', 'adjClose','symbol', 'workingCapital', 'investedCapital', 'grahamNumber', 'target_y', 'adjClose_GSPC', 'year']
features = set(colonne_list) - set(colonnes_texte) - set(colonnes_biais)
features = list(features)
print("features :", features)
print(len(features))


# In[6]:


# Define features. Exclude identifiers and target.
print(df.isna().sum())

# Ensure no NaNs in features

df = df.dropna(subset=features)
df = df.reset_index(drop=True)

print(f"Data after target creation: {df.shape}")
print(f"Years: {df['year'].unique().min()}")


# ## 2. Expanding Window Time-Series Split

# In[7]:


def get_classic_train_val_test_splits(df, val_years, test_years):
    """    
    Yields a single, classic chronological split (Train -> Val -> Test).
    It splits based on the last N years of your dataset.
    """
    years = sorted(df['year'].unique())
    
    print(f"Total years in dataset: {years[0]} to {years[-1]}")
    
    # Calculate the cutoff years from the end of the dataset
    test_start_year = years[-test_years]
    val_start_year = years[-(test_years + val_years)]
    
    print(f"Train ends before: {val_start_year}")
    print(f"Validation: {val_start_year} to {test_start_year - 1}")
    print(f"Test: {test_start_year} to {years[-1]}")
    
    # 1. Train: Everything before the validation period
    train_idx = df[df['year'] < val_start_year].index
    
    # 2. Validation: The specific validation years
    val_idx = df[(df['year'] >= val_start_year) & (df['year'] < test_start_year)].index
    
    # 3. Test: The final years
    test_idx = df[df['year'] >= test_start_year].index
    
    # Create a label for the test period so your MLflow charts name it correctly
    test_label = test_start_year
    
    # Return as a single-element list so your existing 'for' loop still works perfectly!
    return [(train_idx, val_idx, test_idx, test_label)]

# --- Testing the function ---
splits = get_classic_train_val_test_splits(df, val_years=5, test_years=10)

print(f"\nTotal static splits: {len(splits)}")
for i, (tr, val, ts, yr) in enumerate(splits):
    print(f"Split {i+1} | Test Period: {yr} | Train: {len(tr)} rows | Val: {len(val)} rows | Test: {len(ts)} rows")


# In[8]:


"""
def get_train_val_test_splits(df, initial_train_years=20, val_years=3, test_years=1):
    
    #Yields train, val, test indices for an expanding window split.
    
    years = sorted(df['year'].unique())

    start_year = years[0]
    
    current_test_year = start_year + initial_train_years + val_years
    print(current_test_year)
    
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
"""


# ## Reduction Features

# In[9]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# On s'assure qu'on a bien des splits
if len(splits) > 0:
    # 1. Récupération de X_train (en format DataFrame)
    train_idx, val_idx, test_idx, test_year = splits[0]
    X_train_df = df.loc[train_idx, features] 
    
    # ---------------------------------------------------------
    # 2. FILTRE DES CORRÉLATIONS
    # ---------------------------------------------------------
    threshold = 0.8
    col_corr = set()  # Ensemble des colonnes à supprimer
    corr_matrix = X_train_df.corr()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)

    print(f"Features à supprimer (corrélation > {threshold}) :")
    print(col_corr)

    # DataFrame réduit
    #df_reduced = X_train_df.drop(columns=col_corr)
    #print(f"Dimensions avant PCA : {df_reduced.shape}")
    
    # ---------------------------------------------------------
    # 3. SCALING & PCA (Sur le DataFrame réduit !)
    # ---------------------------------------------------------
    # Centrage et Réduction
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train_df) 
    
    # Entraînement de la PCA
    pca = PCA()
    pca.fit(X_train_scale)

    # Calcul de la variance expliquée
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # 4. Affichage du graphique
    plt.figure(figsize=(10, 6))
    # range(1, ...) permet d'avoir l'axe X qui commence à 1 composante (et pas 0)
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o')
    plt.axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label='Seuil de 90%') # Ligne visuelle
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquée cumulée")
    plt.title("Variance expliquée par PCA (Après filtre de corrélation)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.show()
    
else:
    print("Pas assez de données pour le split.")


# ## 3. Evaluation Metrics ($R^2_{OOS}$)

# In[10]:


def calculate_spearman_ic(y_true, y_pred):
    """
    Calculates the Spearman Rank Information Coefficient (IC).
    Measures how well the predicted rankings match the actual return rankings.
    Returns the correlation coefficient (IC).
    """
    # Flatten just in case they are passed as column vectors or pandas Series
    y_true_array = np.asarray(y_true).flatten()
    y_pred_array = np.asarray(y_pred).flatten()
    
    # Calculate Spearman Rank Correlation
    # nan_policy='omit' ensures that any missing data won't crash the calculation
    ic, p_value = spearmanr(y_pred_array, y_true_array, nan_policy='omit')
    
    # If the array is perfectly constant (e.g., model predicts the same value for all stocks), 
    # scipy returns NaN. We catch this and return 0.
    if np.isnan(ic):
        return 0.0
        
    return ic


# ### Deep Neural Network (KerasTuner + MLflow Autolog + Ensembling)

# In[11]:


def build_model(hp, input_dim):
    """KerasTuner model building function"""
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
    hp_units3 = hp.Int('units_3', min_value=4, max_value=8, step=2)
    model.add(layers.Dense(units=hp_units3, kernel_regularizer=regularizers.l1(1e-5)))
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

def train_dnn_ensemble(X_train, y_train, X_val, y_val, scaler, num_models=5, epochs=100, patience=5):
    mlflow.tensorflow.autolog(disable=True) 
    
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    input_dim = X_train_scaled.shape[1]
    
    # ==========================================
    # 🚀 THE MLOPS FIX: tf.data PIPELINE
    # ==========================================
    batch_size = 32768 # You can increase this to 512 or 1024 if VRAM allows
    
    # 1. Train Dataset: Shuffle, Batch, Cache, and Prefetch
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train.values))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train_scaled)) \
                                 .batch(batch_size) \
                                 .cache() \
                                 .prefetch(tf.data.AUTOTUNE)
                                 
    # 2. Validation Dataset: Batch, Cache, and Prefetch (No need to shuffle val data)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val.values))
    val_dataset = val_dataset.batch(batch_size) \
                             .cache() \
                             .prefetch(tf.data.AUTOTUNE)
    # ==========================================

    print("Starting KerasTuner RandomSearch...")
    with tf.device('/CPU:0'):
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, input_dim),
            objective='val_loss',
            max_trials=5,     
            directory='keras_tuner_dir',
            project_name='momentum_dnn',
            overwrite=True
        )
    
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        
        # Pass the datasets instead of raw arrays. Drop the 'batch_size' arg since the dataset handles it.
        tuner.search(train_dataset, epochs=30, validation_data=val_dataset, callbacks=[early_stopping], verbose=0)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best HPs found: Layer1: {best_hps.get('units_1')}, Layer2: {best_hps.get('units_2')}, LR: {best_hps.get('learning_rate')}")
    
    models = []
    mlflow.tensorflow.autolog() 
    
    for i in range(num_models):
        with mlflow.start_run(run_name=f"DNN_Ensemble_Member_{i+1}", nested=True):
            mlflow.log_params(best_hps.values)
            tf.random.set_seed(42 + i)
            
            model = tuner.hypermodel.build(best_hps)
            
            # Use the pre-built tf.data pipeline for blazing fast training
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=[early_stopping],
                verbose=0
            )
            val_loss = min(history.history['val_loss'])
            print(f"Trained DNN component {i+1}/{num_models} - Best Val Loss: {val_loss:.6f}")
            models.append(model)
        
    return models

def predict_dnn_ensemble(models, scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    
    # Create a fast prediction pipeline
    test_dataset = tf.data.Dataset.from_tensor_slices(X_test_scaled)
    test_dataset = test_dataset.batch(32768).cache().prefetch(tf.data.AUTOTUNE)
    
    preds = []
    for model in models:
        # Pass the dataset to predict
        pred = model.predict(test_dataset, verbose=0).flatten()
        preds.append(pred)
        
    return np.mean(preds, axis=0)


# ## 5. Backtesting Portfolio Construction (Decile Sorting)

# In[12]:


import pandas as pd
import numpy as np

def backtest_portfolio(test_df, predictions_col, transaction_cost=0.001, top_n=10):
    """
    Simulates a Long-Only portfolio with dynamic transaction costs.
    Invests strictly in the Top N performing stocks based on predictions_col.
    """
    results = []
    
    # CRITICAL FIX 1: Sort dataframe before shift(-1)
    test_df = test_df.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    test_df = test_df.sort_values(by=['symbol', 'date'])
    
    # Calculate future return (what we will earn if we hold for the next period)
    test_df['return+1'] = test_df.groupby('symbol')['adjClose'].shift(-1) / test_df['adjClose'] - 1
    
    # Initialize empty set to track what we held the previous week
    prev_long_symbols = set()

    for date, week_data in test_df.groupby('date'):
        # Skip weeks with too few stocks to form a proper Top N portfolio
        if len(week_data) < top_n: 
            continue
            
        week_data = week_data.copy()
        
        # --- 1. BENCHMARK CALCULATION ---
        if 'dollar_volume' not in week_data.columns:
            week_data['dollar_volume'] = week_data['volume'] * week_data['adjClose']
            
        universe_weights = week_data['dollar_volume'] / week_data['dollar_volume'].sum()
        benchmark_return = np.sum(universe_weights * week_data['return+1'])
        
        # --- 2. PORTFOLIO TOP N (LONG ONLY) ---
        try:
            # 🚀 NOUVEAU : On trie les actions par la colonne de prédiction/performance
            # ascending=False permet d'avoir les plus grandes valeurs en haut
            week_data = week_data.sort_values(by=predictions_col, ascending=False)
            
            # On prend strictement le Top N
            long_portfolio = week_data.head(top_n)
            
            curr_long_symbols = set(long_portfolio['symbol'])
            num_l = len(curr_long_symbols)
            
            if num_l > 0:
                # Gross Return (Equal-Weighted)
                ret_long = long_portfolio['return+1'].mean()
                gross_portfolio_return = ret_long 
                
                # --- 3. DYNAMIC TRANSACTION COSTS (TURNOVER) ---
                new_longs = curr_long_symbols - prev_long_symbols
                
                # Turnover: Fraction of the portfolio that is new
                turnover_long = len(new_longs) / num_l if prev_long_symbols else 1.0 
                
                # Fee to BUY the new stock AND SELL the old stock
                cost_long = turnover_long * transaction_cost * 2
                
                # Net Return
                net_portfolio_return = gross_portfolio_return - cost_long
                
                # Update memory for the next loop
                prev_long_symbols = curr_long_symbols
            else:
                ret_long, gross_portfolio_return, net_portfolio_return, turnover_long, num_l = 0, 0, 0, 0, 0
                
        except Exception as e:
            ret_long, gross_portfolio_return, net_portfolio_return, turnover_long, num_l = 0, 0, 0, 0, 0
            
        # Append results for this week
        results.append({
            'date': date,
            'gross_return': gross_portfolio_return,
            'portfolio_return': net_portfolio_return,
            'benchmark_return': benchmark_return,
            'turnover_pct': turnover_long,
            'num_long': num_l
        })
            
    bt_df = pd.DataFrame(results)

    # --- METRICS CALCULATION ---
    if bt_df.empty:
        return bt_df, {}
        
    bt_df['date'] = pd.to_datetime(bt_df['date'])
    bt_df = bt_df.dropna(subset=['portfolio_return'])
    
    # Cumulative returns
    bt_df['cum_return'] = (1 + bt_df['portfolio_return']).cumprod()
    bt_df['cum_benchmark'] = (1 + bt_df['benchmark_return']).cumprod()
    bt_df['outperformance'] = bt_df['cum_return'] - bt_df['cum_benchmark']

    n_years = bt_df['date'].dt.year.nunique()
    
    # Portfolio Metrics
    total_ret = bt_df['cum_return'].iloc[-1] - 1
    cagr = (bt_df['cum_return'].iloc[-1]) ** (1 / max(1, n_years)) - 1
    mean_ret = bt_df['portfolio_return'].mean()
    std_ret = bt_df['portfolio_return'].std()
    sharpe = ((mean_ret / std_ret) * np.sqrt(52)) if std_ret > 0 else 0
    max_dd = ((bt_df['cum_return'] - bt_df['cum_return'].cummax()) / bt_df['cum_return'].cummax()).min()

    # Benchmark Metrics
    bench_total_ret = bt_df['cum_benchmark'].iloc[-1] - 1
    bench_cagr = (bt_df['cum_benchmark'].iloc[-1]) ** (1 / max(1, n_years)) - 1 if n_years > 0 and bt_df['cum_benchmark'].iloc[-1] > 0 else 0
    bench_mean = bt_df['benchmark_return'].mean()
    bench_std = bt_df['benchmark_return'].std()
    bench_sharpe = (bench_mean / bench_std) * np.sqrt(52) if bench_std > 0 else 0
    bench_max_dd = ((bt_df['cum_benchmark'] - bt_df['cum_benchmark'].cummax()) / bt_df['cum_benchmark'].cummax()).min()
        
    metrics = {
        'Total_Return': total_ret,
        'CAGR': cagr,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown': max_dd,
        'Bench_Total_Return': bench_total_ret,
        'Bench_CAGR': bench_cagr,
        'Bench_Sharpe': bench_sharpe,
        'Bench_Max_Drawdown': bench_max_dd
    }
    
    return bt_df, metrics


# ## 6. Run the Pipeline over the Expanding Window

# In[13]:


if len(splits) > 0:
    print(f"🚀 Lancement du Walk-Forward Backtest sur {len(splits)} années...")
    
    all_oos_results = []

    STRATEGY_FREQ = "weekly"
    TRANS_COST = 0.001
    DNN_EPOCHS = 100
    DNN_MODELS = 5
    TARGET_FREQ = "1 months"
    NUM_SPLITS = 1
    TOP = 10
    NOTES = "Test avec une target de 1 mois et une fréquence hebdomadaire avec 1 split. Momentum et Crash data"

    mlflow.set_experiment("Maximize_CAGR_SP500")
    
    run_name = f"Test_{STRATEGY_FREQ}_Cost{TRANS_COST}_DnnEp{DNN_EPOCHS}_DnnModels{DNN_MODELS}_Target{TARGET_FREQ}_Top{TOP}_Splits{NUM_SPLITS}_Notes{NOTES}"

    # On ouvre un seul grand Run MLflow pour tout le backtest
    with mlflow.start_run(run_name="CAGR_Forward_Backtest") as parent_run:

        mlflow.set_tag("Frequency", "Weekly")
        
        # 1. ENREGISTREMENT DES PARAMÈTRES GLOBAUX
        mlflow.log_params({
            "frequency": STRATEGY_FREQ,
            "features_count": len(features),
            "features_list": ", ".join(features),
            "transaction_cost": TRANS_COST,
            "dnn_epochs": DNN_EPOCHS,
            "dnn_num_models": DNN_MODELS,
            "target_freq": TARGET_FREQ,
            "top": TOP,
            "num_splits": NUM_SPLITS,
            "notes": NOTES
        })
        
        # 2. LA BOUCLE SUR TOUTES LES ANNÉES
        for i, (train_idx, val_idx, test_idx, test_year) in enumerate(splits):
            print(f"\n--- Traitement de l'Année de Test {test_year} ({i+1}/{len(splits)}) ---")
            
            X_train, y_train = df.loc[train_idx, features], df.loc[train_idx, 'target_y']
            X_val, y_val = df.loc[val_idx, features], df.loc[val_idx, 'target_y']
            X_test, y_test = df.loc[test_idx, features], df.loc[test_idx, 'target_y']
            
            # Setup scaler for DNN (réajusté chaque année)
            scaler = StandardScaler()
            scaler.fit(X_train)
            
            # Entraînement DNN
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
            test_df_year['dnn_pred'] = dnn_preds
            
            all_oos_results.append(test_df_year)
            
        # =========================================================
        # 3. L'ASSEMBLAGE (Le vrai Backtest Global)
        # =========================================================
        print("\n" + "="*50)
        print("🔗 Assemblage des prédictions et calcul du Backtest Global...")
        
        final_test_df = pd.concat(all_oos_results).sort_values(['symbol', 'date'])
        
        # --- SAUVEGARDE DES PRÉDICTIONS BRUTES EN ARTEFACT ---
        print("💾 Sauvegarde des prédictions brutes dans MLflow...")
        csv_path = "final_predictions.csv"
        final_test_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        # Évaluation R^2 globale sur toute la période OOS
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
            })
            
            print("\n" + "="*55)
            print("📊 TABLEAU DES PERFORMANCES FINALES (AVEC FRAIS)")
            print("="*55)
            print(metrics_table.to_string(index=False))
            print("="*55 + "\n")
            
            # Sauvegarde du tableau en format texte brut dans MLflow
            with open("performance_metrics.txt", "w") as f:
                f.write(metrics_table.to_string(index=False))
            mlflow.log_artifact("performance_metrics.txt")
            
        else:
            print("Backtest yielded empty results (perhaps not enough data to form deciles).")
          
else:
    print("Not enough data to form a split. Check your timeframe.")


# ## 7. Visualisation results

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt

print("Running Backtest for DNN Strategy...")
# On garde ton paramétrage (transaction_cost=0.001, decile=55)
bt_dnn, metrics_dnn = backtest_portfolio(final_test_df, 'dnn_pred', transaction_cost=0.001, top_n=5)

if not bt_dnn.empty:
    # 1. TABLEAU DES PERFORMANCES (DNN UNIQUEMENT)
    metrics_table = pd.DataFrame({
        "Métrique": ["Total Return", "CAGR (Annualisé)", "Sharpe Ratio", "Max Drawdown"],
        "DNN Ensemble": [f"{metrics_dnn['Total_Return']*100:.2f}%", f"{metrics_dnn['CAGR']*100:.2f}%", f"{metrics_dnn['Sharpe_Ratio']:.2f}", f"{metrics_dnn['Max_Drawdown']*100:.2f}%"]
    })
    
    print("\n" + "="*55)
    print("📊 TABLEAU DES PERFORMANCES FINALES (HEBDOMADAIRE - AVEC FRAIS 0.1%)")
    print("="*55)
    print(metrics_table.to_string(index=False))
    print("="*55 + "\n")

    # 2. GRAPHIQUES (DNN vs BENCHMARK)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Axe 1 : Rendements cumulés (Échelle Logarithmique) ---
    ax1.plot(bt_dnn['date'], bt_dnn['cum_return'], label=f"DNN (Sharpe: {metrics_dnn['Sharpe_Ratio']:.2f})", color='green', linewidth=2)
    
    # On utilise maintenant metrics_dnn pour récupérer le Sharpe du Benchmark
    ax1.plot(bt_dnn['date'], bt_dnn['cum_benchmark'], label=f"Benchmark S&P 500 (Sharpe: {metrics_dnn['Bench_Sharpe']:.2f})", color='black', linestyle='--', linewidth=2)
    
    ax1.axhline(y=1.0, color='red', linestyle=':', alpha=0.5)
    ax1.set_yscale('log')
    ax1.set_title('Walk-Forward Cumulative Returns vs Benchmark S&P 500 (WEEKLY - LOG SCALE)')
    ax1.set_ylabel('Cumulative Return (Log Scale, 1.0 = Initial Capital)')
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    
    # --- Axe 2 : Spread d'Outperformance (DNN uniquement) ---
    ax2.fill_between(bt_dnn['date'], bt_dnn['outperformance'], 0, where=(bt_dnn['outperformance'] >= 0), color='green', alpha=0.3, label='DNN Outperformance')
    ax2.fill_between(bt_dnn['date'], bt_dnn['outperformance'], 0, where=(bt_dnn['outperformance'] < 0), color='red', alpha=0.3, label='DNN Underperformance')
    
    ax2.axhline(y=0.0, color='black', linewidth=1)
    ax2.set_title('Outperformance Spread (Strategy - Benchmark)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Spread (Linear)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # 3. SAUVEGARDE MLFLOW
    plot_path = "global_cumulative_returns_vs_benchmark.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    
    plt.show()
    
else:
    print("Le backtest du DNN est vide (peut-être pas assez de données pour former les déciles).")


# In[ ]:


final_test_df.sort_values(by='date', ascending=False).head(10)


# In[17]:


final_test_df[final_test_df['date'] == '2023-01-06'].sort_values(by='dnn_pred', ascending=False).head(10)


# In[18]:


final_test_df[final_test_df['date'] == '2023-01-06'].sort_values(by='target_y', ascending=False).head(10)


# In[19]:


df_test_decile = final_test_df.copy()


# In[20]:


df_test_decile['decile'] = pd.qcut(df_test_decile['dnn_pred'], 65, labels=False, duplicates='drop') + 1
df_test_decile.describe()


# In[21]:


max_decile = df_test_decile['decile'].max()
decile_long = df_test_decile[df_test_decile['decile'] == max_decile].sort_values(by='date', ascending=True)
decile_long_error = decile_long[decile_long['target_y'] < 0]
percent_false = len(decile_long_error)/len(decile_long)
percent_false


# In[22]:


min_decile = df_test_decile['decile'].min()
decile_short = df_test_decile[df_test_decile['decile'] == min_decile].sort_values(by='date', ascending=True)
decile_short_error = decile_short[decile_short['target_y'] > 0]
percent_false = len(decile_short_error)/len(decile_short)
percent_false


# In[23]:


decile_long.sort_values(by='date', ascending=False).head(20)


# In[ ]:




