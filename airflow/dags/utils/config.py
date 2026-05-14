import os
from datetime import datetime, timedelta
from airflow.utils.email import send_email

# --- PATHS & ENVIRONMENT ---
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME', '/opt/airflow')
DBT_DIR = f"{AIRFLOW_HOME}/dbt"
DBT_BIN = "/home/airflow/.local/bin/dbt"
GCP_KEY = "/opt/airflow/config/keys_gcp/finance-ml-project-486410-f5aa9a641051.json"
BUCKET_NAME = "finance-data-lake-unique-id"

# --- SPARK CONFIGURATION ---
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

# --- BASH COMMAND PREFIXES ---
PREFIX_SPARK = (
    f'export BUCKET_NAME="{BUCKET_NAME}" && '
    f'export DOCKER_ENV="true" && '
    f'export MLFLOW_TRACKING_URI="http://momentum-mlflow-server:5000" && '
    f'export PYSPARK_SUBMIT_ARGS="--conf spark.jars.ivy=/tmp/ivy_cache_$RANDOM --packages {SPARK_PACKAGES} --jars {GCS_JAR} {HADOOP_CONFS} pyspark-shell" && '
    f'export GCP_KEY_PATH={GCP_KEY}'
)

PREFIX_DBT = (
    f"export GCS_ACCESS_KEY_ID=$(grep GCS_ACCESS_KEY_ID {AIRFLOW_HOME}/.env | cut -d'=' -f2 | tr -d '\" ') && "
    f"export GCS_SECRET_ACCESS_KEY=$(grep GCS_SECRET_ACCESS_KEY {AIRFLOW_HOME}/.env | cut -d'=' -f2 | tr -d '\" ') && "
    f"export BUCKET_NAME=\"{BUCKET_NAME}\""
)

# --- AIRFLOW CALLBACKS ---
def on_failure_callback(context):
    """Callback en cas d'échec d'une tâche."""
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

# --- DEFAULT DAG ARGS ---
DEFAULT_ARGS = {
    'owner': 'momentum_ai',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['quentin-forget@hotmail.fr'],
    'on_failure_callback': on_failure_callback,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
