from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'momentum_ai',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Planification : Chaque Vendredi à 22h15
# Cron: 15 22 * * 5
with DAG(
    '01_prod_bronze_ingestion',
    default_args=default_args,
    description='Pipeline d\'ingestion Bronze Momentum AI',
    schedule_interval='15 22 * * 5',
    catchup=False,
    tags=['prod', 'bronze'],
) as dag:

    # 1. Récupération de la liste des tickers (Must run first)
    task_fetch_tickers = BashOperator(
        task_id='fetch_tickers_2b',
        bash_command='python3 /opt/airflow/src/data_enginnering/prod/bronze/List_ticker_YF.py',
    )

    # 2. Ingestion Parallèle des prix
    task_ingest_stocks_2b = BashOperator(
        task_id='ingest_stocks_2b',
        bash_command='python3 /opt/airflow/src/data_enginnering/prod/bronze/data_raw_2b.py',
    )

    task_ingest_etfs = BashOperator(
        task_id='ingest_raw_etfs',
        bash_command='python3 /opt/airflow/src/data_enginnering/prod/bronze/data_raw_etfs.py',
    )

    task_ingest_sp500 = BashOperator(
        task_id='ingest_raw_sp500',
        bash_command='python3 /opt/airflow/src/data_enginnering/prod/bronze/data_raw_sp500.py',
    )

    # 3. Trigger de la couche Silver
    trigger_silver = TriggerDagRunOperator(
        task_id='trigger_silver_layer',
        trigger_dag_id='02_prod_silver_processing',
        wait_for_completion=False,
    )

    # Dépendances
    task_fetch_tickers >> [task_ingest_stocks_2b, task_ingest_etfs, task_ingest_sp500] >> trigger_silver
