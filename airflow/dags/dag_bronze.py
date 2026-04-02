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
    <p>Si l'échec provient de la tâche <b>validate_bronze_data</b>, veuillez consulter les 
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

# Planification : Chaque Vendredi à 22h15
with DAG(
    '01_prod_bronze_ingestion',
    default_args=default_args,
    description='Pipeline d\'ingestion Bronze Momentum AI with GX validation',
    schedule_interval='15 22 * * 5',
    catchup=False,
    tags=['prod', 'bronze', 'gx'],
) as dag:

    # 1. Récupération de la liste des tickers
    task_fetch_tickers = BashOperator(
        task_id='fetch_tickers_2b',
        bash_command='python3 /opt/airflow/src/data_enginnering/prod/bronze/List_ticker_YF.py',
    )

    # 2. Ingestion Parallèle
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

    # 3. VALIDATION GREAT EXPECTATIONS
    task_validate_bronze = BashOperator(
        task_id='validate_bronze_data',
        bash_command='python3 /opt/airflow/scripts/validate_bronze.py',
    )

    # 4. Trigger de la couche Silver
    trigger_silver = TriggerDagRunOperator(
        task_id='trigger_silver_layer',
        trigger_dag_id='02_prod_silver_processing',
        wait_for_completion=False,
    )

    # Dépendances
    task_fetch_tickers >> [task_ingest_stocks_2b, task_ingest_etfs, task_ingest_sp500] >> task_validate_bronze >> trigger_silver
