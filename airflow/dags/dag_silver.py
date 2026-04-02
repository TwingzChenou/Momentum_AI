from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.email import send_email
from datetime import datetime, timedelta

def on_failure_callback(context):
    subject = f"🚨 Airflow Alert: Failure in {context['task_instance'].dag_id}"
    html_content = f"""
    <h3>Pipeline Failure Detected</h3>
    <p><b>DAG:</b> {context['task_instance'].dag_id}</p>
    <p><b>Task:</b> {context['task_instance'].task_id}</p>
    <p><b>Execution Date:</b> {context['execution_date']}</p>
    <p><b>Log URL:</b> <a href="{context['task_instance'].log_url}">Click here for logs</a></p>
    <hr>
    <p>Si l'échec provient de la tâche <b>validate_silver_data</b>, veuillez consulter les 
    <a href="https://storage.googleapis.com/finance-data-lake-unique-id/data_quality_reports/index.html">Data Docs Great Expectations</a> sur GCS.</p>
    """
    send_email(to='quentin-forget@hotmail.fr', subject=subject, html_content=html_content)

default_args = {
    'owner': 'momentum_ai',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['quentin-forget@hotmail.fr'],
    'on_failure_callback': on_failure_callback,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Déclenché par la couche Bronze (pas besoin de scheduler)
with DAG(
    '02_prod_silver_processing',
    default_args=default_args,
    description='Pipeline de nettoyage Silver Momentum AI with GX validation',
    schedule_interval=None,
    catchup=False,
    tags=['prod', 'silver', 'gx'],
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

    # 2. VALIDATION GREAT EXPECTATIONS
    task_validate_silver = BashOperator(
        task_id='validate_silver_data',
        bash_command='python3 /opt/airflow/scripts/validate_silver.py',
    )

    # 3. Trigger de la couche Gold
    trigger_gold = TriggerDagRunOperator(
        task_id='trigger_gold_layer',
        trigger_dag_id='03_prod_gold_features',
        wait_for_completion=False,
    )

    # Dépendances (Tout en parallèle avant de trigger la Gold)
    [task_check_2b, task_check_etfs, task_check_sp500] >> task_validate_silver >> trigger_gold
