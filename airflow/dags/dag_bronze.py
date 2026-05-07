import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from utils.config import PREFIX_SPARK, DEFAULT_ARGS

# Alias pour la compatibilité avec le code existant
PREFIX_CMD = PREFIX_SPARK

with DAG(
    '01_prod_bronze_ingestion',
    default_args=DEFAULT_ARGS,
    description='Pipeline d\'ingestion Bronze Momentum AI (Correction Spark Config)',
    schedule_interval='15 22 * * 5',
    max_active_runs=1,
    catchup=False,
    tags=['prod', 'bronze'],
) as dag:

    # 1. Récupération de la liste des tickers
    task_fetch_tickers_2b = BashOperator(
        task_id='fetch_tickers_2b',
        bash_command=f'{PREFIX_CMD} && python3 /opt/airflow/src/data_engineering/prod/bronze/List_ticker_YF.py',
    )

    task_fetch_sp500_list = BashOperator(
        task_id='fetch_sp500_list_fmp',
        bash_command=f'{PREFIX_CMD} && python3 /opt/airflow/src/data_engineering/prod/bronze/sp500_list_ingestion.py',
    )

    # 2. Consolidation de l'historique S&P 500
    task_consolidate_history = BashOperator(
        task_id='consolidate_sp500_history',
        bash_command=f'{PREFIX_CMD} && python3 /opt/airflow/src/data_engineering/prod/bronze/sp500_consolidated_history.py',
    )

    # 3. Ingestion Parallèle des prix
    task_ingest_stocks_2b = BashOperator(
        task_id='ingest_stocks_2b',
        bash_command=f'{PREFIX_CMD} && python3 /opt/airflow/src/data_engineering/prod/bronze/data_raw_2b.py',
    )

    task_ingest_etfs = BashOperator(
        task_id='ingest_raw_etfs',
        bash_command=f'{PREFIX_CMD} && python3 /opt/airflow/src/data_engineering/prod/bronze/data_raw_etfs.py',
    )

    task_ingest_sp500_index = BashOperator(
        task_id='ingest_sp500_index',
        bash_command=f'{PREFIX_CMD} && python3 /opt/airflow/src/data_engineering/prod/bronze/data_raw_sp500.py',
    )

    task_ingest_sp500_stocks = BashOperator(
        task_id='ingest_sp500_stocks_daily',
        bash_command=f'{PREFIX_CMD} && python3 /opt/airflow/src/data_engineering/prod/bronze/sp500_prices_daily.py',
    )

    # 4. Trigger de la couche Silver
    trigger_silver = TriggerDagRunOperator(
        task_id='trigger_silver_layer',
        trigger_dag_id='02_prod_silver_processing',
        wait_for_completion=False,
    )

    # Dépendances logic
    # Branche 2B
    task_fetch_tickers_2b >> task_ingest_stocks_2b
    
    # Branche SP500
    task_fetch_sp500_list >> task_consolidate_history >> task_ingest_sp500_stocks
    
    # Ingestions indépendantes (ETF et Index se lancent au début)
    # [task_ingest_etfs, task_ingest_sp500_index] # Pas de prérequis nécessaire

    # On s'assure que TOUT est fini avant de déclencher Silver
    [task_ingest_stocks_2b, task_ingest_etfs, task_ingest_sp500_index, task_ingest_sp500_stocks] >> trigger_silver
