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

# Déclenché par la couche Bronze (pas besoin de scheduler)
with DAG(
    '02_prod_silver_processing',
    default_args=default_args,
    description='Pipeline de nettoyage Silver Momentum AI',
    schedule_interval=None,
    catchup=False,
    tags=['prod', 'silver'],
) as dag:

    # 1. Vérification par blocs
    task_check_2b = BashOperator(
        task_id='checking_stocks_2b',
        bash_command='python3 /opt/airflow/src/data_enginnering/prod/silver/data_checking_2b.py',
    )

    task_check_etfs = BashOperator(
        task_id='checking_etfs',
        bash_command='python3 /opt/airflow/src/data_enginnering/prod/silver/data_checking_etfs.py',
    )

    task_check_sp500 = BashOperator(
        task_id='checking_sp500',
        bash_command='python3 /opt/airflow/src/data_enginnering/prod/silver/data_checking_sp500.py',
    )

    # 2. Trigger de la couche Gold
    trigger_gold = TriggerDagRunOperator(
        task_id='trigger_gold_layer',
        trigger_dag_id='03_prod_gold_features',
        wait_for_completion=False,
    )

    # Dépendances (Tout en parallèle avant de trigger la Gold)
    [task_check_2b, task_check_etfs, task_check_sp500] >> trigger_gold
