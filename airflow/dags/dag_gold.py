from airflow import DAG
from airflow.operators.bash import BashOperator
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
    <p>Si l'échec provient de la tâche <b>validate_gold_data</b>, veuillez consulter les 
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

# Déclenché par la couche Silver
with DAG(
    '03_prod_gold_features',
    default_args=default_args,
    description='Pipeline de calcul des indicateurs Gold Momentum AI with GX validation',
    schedule_interval=None,
    catchup=False,
    tags=['prod', 'gold', 'gx'],
) as dag:

    # 1. Calcul des Features
    task_generate_features = BashOperator(
        task_id='generate_indicators_gold',
        bash_command='python3 /opt/airflow/src/data_enginnering/prod/gold/features_2b_etf.py',
    )

    # 2. VALIDATION GREAT EXPECTATIONS
    task_validate_gold = BashOperator(
        task_id='validate_gold_data',
        bash_command='python3 /opt/airflow/scripts/validate_gold.py',
    )

    # Dépendance
    task_generate_features >> task_validate_gold
