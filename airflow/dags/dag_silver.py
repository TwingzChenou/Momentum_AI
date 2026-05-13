import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from utils.config import (
    AIRFLOW_HOME, DBT_DIR, DBT_BIN, BUCKET_NAME,
    PREFIX_SPARK, PREFIX_DBT, DEFAULT_ARGS
)

BUCKET = BUCKET_NAME

with DAG(
    '02_prod_silver_processing',
    default_args=DEFAULT_ARGS,
    description='Pipeline de nettoyage et rééchantillonnage Silver (DuckDB + Spark)',
    schedule_interval=None,  # Exécution manuelle
    max_active_runs=1,  # Limite le nombre de runs simultanés à 1
    catchup=False,  # Ne pas exécuter les runs manquées
    tags=['prod', 'silver', 'duckdb'],  # Tags pour filtrer et organiser les DAGs
) as dag:

    # Configuration des types de données
    data_types = {
        '2b': ('stg_2b_prices_daily', 'data_raw_2b'),
        'etf': ('stg_etf_prices_daily', 'data_raw_etf'),
        'sp500': ('stg_sp500_index_daily', 'data_raw_sp500'),
        'sp500_stocks': ('stg_sp500_stocks_daily', 'sp500_stock_prices')
    }

    BUCKET = "finance-data-lake-unique-id"
    
    for key, (model, folder) in data_types.items():
        daily_path = f"gs://{BUCKET}/silver/{folder}.parquet"
        weekly_path = f"gs://{BUCKET}/silver/{folder}_weekly"
        monthly_path = f"gs://{BUCKET}/silver/{folder}_monthly"

        # 1. Nettoyage Daily via dbt (DuckDB)
        task_daily = BashOperator(
            task_id=f'daily_clean_{key}',
            bash_command=f'{PREFIX_DBT} && {DBT_BIN} run --select {model} --profiles-dir {DBT_DIR} --project-dir {DBT_DIR} && {DBT_BIN} test --select {model} --profiles-dir {DBT_DIR} --project-dir {DBT_DIR}',
            # Exécute des commandes dbt pour nettoyer les données quotidiennes
        )

        # 2. Rééchantillonnage Weekly (Spark Python)
        task_weekly = BashOperator(
            task_id=f'resample_weekly_{key}',
            bash_command=f'{PREFIX_SPARK} && python3 {AIRFLOW_HOME}/src/data_engineering/prod/silver/resample_data.py --source {daily_path} --target {weekly_path} --freq W-FRI --name {key}_weekly',
            # Exécute un script Python pour rééchantillonner les données hebdomadaires
        )

        # 3. Rééchantillonnage Monthly (Spark Python)
        task_monthly = BashOperator(
            task_id=f'resample_monthly_{key}',
            bash_command=f'{PREFIX_SPARK} && python3 {AIRFLOW_HOME}/src/data_engineering/prod/silver/resample_data.py --source {daily_path} --target {monthly_path} --freq M --name {key}_monthly',
            # Exécute un script Python pour rééchantillonner les données mensuelles
        )

        # Dépendances
        task_daily >> task_weekly >> task_monthly

    # NOUVEAU : Trigger vers l'optimisation hebdomadaire au lieu du Gold direct
    trigger_gold_layer = TriggerDagRunOperator(
        task_id='trigger_gold_layer',
        trigger_dag_id='03_prod_gold_features',
        wait_for_completion=False,
    )

    all_monthly_tasks = [dag.get_task(f'resample_monthly_{key}') for key in data_types.keys()]
    all_monthly_tasks >> trigger_gold_layer