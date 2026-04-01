from airflow import DAG
from airflow.operators.bash import BashOperator
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

# Déclenché par la couche Silver
with DAG(
    '03_prod_gold_features',
    default_args=default_args,
    description='Pipeline de calcul des indicateurs Gold Momentum AI',
    schedule_interval=None,
    catchup=False,
    tags=['prod', 'gold'],
) as dag:

    # 1. Calcul des Features (Une seule tâche consolidée car le script gère tout)
    task_generate_features = BashOperator(
        task_id='generate_indicators_gold',
        bash_command='python3 /opt/airflow/src/data_enginnering/prod/gold/features_2b_etf.py',
    )

    # Dépendance
    task_generate_features
