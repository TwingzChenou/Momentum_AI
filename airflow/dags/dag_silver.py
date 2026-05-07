import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# Configuration des chemins
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME', '/opt/airflow')
DBT_DIR = f"{AIRFLOW_HOME}/dbt"
DBT_BIN = "/home/airflow/.local/bin/dbt"  # Chemin dans le container Airflow
GCP_KEY = "/opt/airflow/config/keys_gcp/finance-ml-project-486410-f5aa9a641051.json"

# --- CONFIGURATION SPARK (Conservée uniquement pour le rééchantillonnage) ---
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

# Préfixe pour les tâches Spark (Python)
PREFIX_SPARK = (
    f'export BUCKET_NAME="finance-data-lake-unique-id" && '
    f'export PYSPARK_SUBMIT_ARGS="--conf spark.jars.ivy=/tmp/ivy_cache_$RANDOM --packages {SPARK_PACKAGES} --jars {GCS_JAR} {HADOOP_CONFS} pyspark-shell" && '
    f'export GCP_KEY_PATH={GCP_KEY}'
)

# Préfixe pour les tâches dbt (DuckDB)
PREFIX_DBT = (
    f"export GCS_ACCESS_KEY_ID=$(grep GCS_ACCESS_KEY_ID {AIRFLOW_HOME}/.env | cut -d'=' -f2 | tr -d '\" ') && "
    f"export GCS_SECRET_ACCESS_KEY=$(grep GCS_SECRET_ACCESS_KEY {AIRFLOW_HOME}/.env | cut -d'=' -f2 | tr -d '\" ') && "
    f"export BUCKET_NAME=\"finance-data-lake-unique-id\""
)

default_args = {
    'owner': 'momentum_ai',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    '02_prod_silver_processing',
    default_args=default_args,
    description='Pipeline de nettoyage et rééchantillonnage Silver (DuckDB + Spark)',
    schedule_interval=None,
    max_active_runs=1,
    catchup=False,
    tags=['prod', 'silver', 'duckdb'],
) as dag:

    # Configuration des types de données
    data_types = {
        '2b': ('stg_2b_prices_daily', 'data_raw_2b'),
        'etf': ('stg_etf_prices_daily', 'data_raw_etf'),
        'sp500': ('stg_sp500_index_daily', 'data_raw_sp500'),
        'sp500_stocks': ('stg_sp500_stocks_daily', 'sp500_stock_prices')
    }

    BUCKET = "finance-data-lake-unique-id"
    
    for key, (model, folder) in data_types.items():
        daily_path = f"gs://{BUCKET}/silver/{folder}.parquet"
        weekly_path = f"gs://{BUCKET}/silver/{folder}_weekly"
        monthly_path = f"gs://{BUCKET}/silver/{folder}_monthly"

        # 1. Nettoyage Daily via dbt (DuckDB)
        task_daily = BashOperator(
            task_id=f'daily_clean_{key}',
            bash_command=f'{PREFIX_DBT} && {DBT_BIN} run --select {model} --profiles-dir {DBT_DIR} --project-dir {DBT_DIR} && {DBT_BIN} test --select {model} --profiles-dir {DBT_DIR} --project-dir {DBT_DIR}'
        )

        # 2. Rééchantillonnage Weekly (Spark Python)
        task_weekly = BashOperator(
            task_id=f'resample_weekly_{key}',
            bash_command=f'{PREFIX_SPARK} && python3 {AIRFLOW_HOME}/src/data_engineering/prod/silver/resample_data.py --source {daily_path} --target {weekly_path} --freq W-FRI --name {key}_weekly'
        )

        # 3. Rééchantillonnage Monthly (Spark Python)
        task_monthly = BashOperator(
            task_id=f'resample_monthly_{key}',
            bash_command=f'{PREFIX_SPARK} && python3 {AIRFLOW_HOME}/src/data_engineering/prod/silver/resample_data.py --source {daily_path} --target {monthly_path} --freq M --name {key}_monthly'
        )

        # Dépendances
        task_daily >> task_weekly >> task_monthly

    # NOUVEAU : Trigger vers l'optimisation hebdomadaire au lieu du Gold direct
    trigger_optimization = TriggerDagRunOperator(
        task_id='trigger_strategy_optimization',
        trigger_dag_id='strategy_optimization_weekly',
        wait_for_completion=False,
    )

    all_monthly_tasks = [dag.get_task(f'resample_monthly_{key}') for key in data_types.keys()]
    all_monthly_tasks >> trigger_optimization