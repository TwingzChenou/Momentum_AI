from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.email import send_email
from datetime import datetime, timedelta
import os
import json

# Configuration des chemins
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME', '/opt/airflow')
DBT_DIR = f"{AIRFLOW_HOME}/dbt"
DBT_BIN = "/home/airflow/.local/bin/dbt"
GCP_KEY = "/opt/airflow/config/keys_gcp/finance-ml-project-486410-f5aa9a641051.json"
CONFIG_FILE = f"{AIRFLOW_HOME}/config/best_strategy_params.json"

def get_dbt_vars(**context):
    """Lit le fichier JSON des meilleurs paramètres et retourne une chaîne formatée pour dbt --vars."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                params = json.load(f)
            # On retourne le JSON en string pour l'injecter dans la commande bash
            return json.dumps(params)
        else:
            print(f"WARNING: {CONFIG_FILE} not found. Using default dbt vars.")
            return "{}"
    except Exception as e:
        print(f"ERROR reading config: {e}")
        return "{}"

default_args = {
    'owner': 'momentum_ai',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['quentin-forget@hotmail.fr'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Configuration Spark pour l'export BQ
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

PREFIX_SPARK = (
    f'export BUCKET_NAME="finance-data-lake-unique-id" && '
    f'export PYSPARK_SUBMIT_ARGS="--conf spark.jars.ivy=/tmp/ivy_cache_$RANDOM --packages {SPARK_PACKAGES} --jars {GCS_JAR} {HADOOP_CONFS} pyspark-shell" && '
    f'export GCP_KEY_PATH={GCP_KEY}'
)

PREFIX_DBT = (
    f"export GCS_ACCESS_KEY_ID=$(grep GCS_ACCESS_KEY_ID {AIRFLOW_HOME}/.env | cut -d'=' -f2 | tr -d '\" ') && "
    f"export GCS_SECRET_ACCESS_KEY=$(grep GCS_SECRET_ACCESS_KEY {AIRFLOW_HOME}/.env | cut -d'=' -f2 | tr -d '\" ') && "
    f"export BUCKET_NAME=\"finance-data-lake-unique-id\""
)

with DAG(
    '03_prod_gold_features',
    default_args=default_args,
    description='Pipeline Gold Momentum AI dynamique (Paramètres optimisés par Optuna)',
    schedule_interval=None,
    catchup=False,
    tags=['prod', 'gold', 'duckdb', 'dynamic'],
) as dag:

    # 1. Récupération des paramètres
    fetch_params = PythonOperator(
        task_id='fetch_optimization_params',
        python_callable=get_dbt_vars,
    )

    # 2. Calcul des Features via dbt (DuckDB) avec injection des variables
    # On utilise {{ task_instance.xcom_pull(task_ids='fetch_optimization_params') }}
    task_generate_features = BashOperator(
        task_id='generate_indicators_gold_dbt',
        bash_command=(
            f'{PREFIX_DBT} && '
            f'{DBT_BIN} run-operation drop_old_gold_tables --profiles-dir {DBT_DIR} --project-dir {DBT_DIR} && '
            f'{DBT_BIN} run --select models/gold --profiles-dir {DBT_DIR} --project-dir {DBT_DIR} '
            f"--vars '{{{{ task_instance.xcom_pull(task_ids=\"fetch_optimization_params\") }}}}' && "
            f'{DBT_BIN} test --select models/gold --profiles-dir {DBT_DIR} --project-dir {DBT_DIR} '
            f"--vars '{{{{ task_instance.xcom_pull(task_ids=\"fetch_optimization_params\") }}}}'"
        )
    )

    # 3. Export des données Gold vers BigQuery
    task_export_bq = BashOperator(
        task_id='export_gold_to_bigquery',
        bash_command=f'{PREFIX_SPARK} && python3 {AIRFLOW_HOME}/src/data_engineering/prod/gold/export_gold_to_bq.py'
    )

    # Dépendances
    fetch_params >> task_generate_features >> task_export_bq
