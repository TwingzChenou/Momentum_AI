from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.email import send_email
from datetime import datetime, timedelta
from config.config_spark import Paths

def on_failure_callback(context):
    subject = f"🚨 Airflow Alert: Failure in {context['task_instance'].dag_id}"
    html_content = f"""
    <h3>Pipeline Failure Detected</h3>
    <p><b>DAG:</b> {context['task_instance'].dag_id}</p>
    <p><b>Task:</b> {context['task_instance'].task_id}</p>
    <p><b>Execution Date:</b> {context['execution_date']}</p>
    <p><b>Log URL:</b> <a href="{context['task_instance'].log_url}">Click here for logs</a></p>
    <hr>
    <p>Veuillez consulter les logs Airflow pour plus de détails sur l'échec de la validation de la couche Gold.</p>
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

# --- CONFIGURATION SPARK & DBT ---
DBT_DIR = "/opt/airflow/dbt"
DBT_BIN = "/home/airflow/.local/bin/dbt"
GCP_KEY = "/opt/airflow/config/keys_gcp/finance-ml-project-486410-f5aa9a641051.json"
GCS_JAR = "/opt/airflow/jars/gcs-connector-hadoop3-latest.jar"

# On centralise les packages ici
SPARK_PACKAGES = (
    "io.delta:delta-spark_2.12:3.2.1,"
    "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.34.0"
)

# Configuration Hadoop pour GCS
HADOOP_CONFS = (
    "--conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension "
    "--conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog "
    "--conf spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem "
    "--conf spark.hadoop.fs.AbstractFileSystem.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS "
    "--conf spark.hadoop.google.cloud.auth.service.account.enable=true "
    f"--conf spark.hadoop.google.cloud.auth.service.account.json.keyfile={GCP_KEY}"
)

# Commande préfixe pour isoler le cache Ivy et configurer Spark
PREFIX_CMD = (
    f'export BUCKET_NAME="finance-data-lake-unique-id" && '
    f'export PYSPARK_SUBMIT_ARGS="--conf spark.jars.ivy=/tmp/ivy_cache_$RANDOM --packages {SPARK_PACKAGES} --jars {GCS_JAR} {HADOOP_CONFS} pyspark-shell" && '
    f'export GCP_KEY_PATH={GCP_KEY}'
)

# Déclenché par la couche Silver
with DAG(
    '03_prod_gold_features',
    default_args=default_args,
    description='Pipeline de calcul des indicateurs Gold Momentum AI avec validation DBT',
    schedule_interval=None,
    max_active_runs=1,
    catchup=False,
    tags=['prod', 'gold'],
) as dag:

    # 1. Calcul des Features via dbt (Python Models)
    task_generate_features = BashOperator(
        task_id='generate_indicators_gold_dbt',
        bash_command=(
            f'{PREFIX_CMD} && {DBT_BIN} run-operation drop_old_gold_tables --profiles-dir {DBT_DIR} --project-dir {DBT_DIR} '
            f'&& {DBT_BIN} run --select models/gold --profiles-dir {DBT_DIR} --project-dir {DBT_DIR} '
            f'--vars \'{{"register_silver": true, '
            f'"stock_features_path": "{Paths.STOCK_FEATURES_GOLD}", '
            f'"etf_features_path": "{Paths.ETF_FEATURES_GOLD}", '
            f'"index_features_path": "{Paths.INDEX_FEATURES_GOLD}"}}\' '
            f'&& {DBT_BIN} test --select models/gold --profiles-dir {DBT_DIR} --project-dir {DBT_DIR} '
            f'--vars \'{{"register_silver": true}}\''
        )
    )

    # 2. Export des données Gold vers BigQuery
    task_export_bq = BashOperator(
        task_id='export_gold_to_bigquery',
        bash_command=f'{PREFIX_CMD} && python3 /opt/airflow/src/data_enginnering/prod/gold/export_gold_to_bq.py'
    )

    # Dépendances
    task_generate_features >> task_export_bq
