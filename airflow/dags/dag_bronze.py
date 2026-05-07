import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.email import send_email
from datetime import datetime, timedelta

# Configuration des chemins et Spark (Identique aux autres DAGs pour la cohérence)
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME', '/opt/airflow')
GCP_KEY = "/opt/airflow/config/keys_gcp/finance-ml-project-486410-f5aa9a641051.json"
SPARK_PACKAGES = "io.delta:delta-spark_2.12:3.2.1,com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.34.0"
GCS_JAR = "/opt/airflow/jars/gcs-connector-hadoop3-latest.jar"
HADOOP_CONFS = (
    "--conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension "
    "--conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog "
    "--conf spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem "
    "--conf spark.hadoop.fs.AbstractFileSystem.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS "
    "--conf spark.hadoop.google.cloud.auth.service.account.enable=true "
    f"--conf spark.hadoop.google.cloud.auth.service.account.json.keyfile={GCP_KEY}"
)

PREFIX_CMD = (
    f'export BUCKET_NAME="finance-data-lake-unique-id" && '
    f'export PYSPARK_SUBMIT_ARGS="--conf spark.jars.ivy=/tmp/ivy_cache_$RANDOM --packages {SPARK_PACKAGES} --jars {GCS_JAR} {HADOOP_CONFS} pyspark-shell" && '
    f'export GCP_KEY_PATH={GCP_KEY}'
)

def on_failure_callback(context):
    subject = f"🚨 Airflow Alert: Failure in {context['task_instance'].dag_id}"
    html_content = f"""
    <h3>Pipeline Failure Detected</h3>
    <p><b>DAG:</b> {context['task_instance'].dag_id}</p>
    <p><b>Task:</b> {context['task_instance'].task_id}</p>
    <p><b>Execution Date:</b> {context['execution_date']}</p>
    <p><b>Log URL:</b> <a href="{context['task_instance'].log_url}">Click here for logs</a></p>
    <hr>
    <p>Veuillez consulter les logs Airflow pour plus de détails sur l'échec de la validation du pipeline.</p>
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

with DAG(
    '01_prod_bronze_ingestion',
    default_args=default_args,
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
