from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import os
import json
from utils.config import (
    AIRFLOW_HOME, DBT_DIR, DBT_BIN,
    PREFIX_SPARK, PREFIX_DBT, DEFAULT_ARGS
)

CONFIG_FILE = f"{AIRFLOW_HOME}/config/best_strategy_params.json"

def get_dbt_vars(**context):
    """
    Lit le fichier JSON des meilleurs paramètres et retourne une chaîne formatée pour dbt --vars.
    """
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

with DAG(
    '03_prod_gold_features',
    default_args=DEFAULT_ARGS,
    description='Pipeline Gold Momentum AI dynamique (Paramètres optimisés par Optuna)',
    schedule_interval=None,  # Exécution manuelle
    catchup=False,  # Ne pas exécuter les runs manquées
    tags=['prod', 'gold', 'duckdb', 'dynamic'],  # Tags pour filtrer et organiser les DAGs
) as dag:

    # 1. Récupération des paramètres
    fetch_params = PythonOperator(
        task_id='fetch_optimization_params',
        python_callable=get_dbt_vars,
        # Lit le fichier JSON des meilleurs paramètres et retourne une chaîne formatée pour dbt --vars
    )

    # 2. Calcul des Features via dbt (DuckDB) avec injection des variables
    task_generate_features = BashOperator(
        task_id='generate_indicators_gold_dbt',
        bash_command=(
            f'{PREFIX_DBT} && '
            f'{DBT_BIN} run-operation drop_old_gold_tables --profiles-dir {DBT_DIR} --project-dir {DBT_DIR} && '
            f'{DBT_BIN} run --select models/gold --profiles-dir {DBT_DIR} --project-dir {DBT_DIR} '
            f"--vars '{{{{ task_instance.xcom_pull(task_ids='fetch_optimization_params') }}}}' && "
            f'{DBT_BIN} test --select models/gold --profiles-dir {DBT_DIR} --project-dir {DBT_DIR} '
            f"--vars '{{{{ task_instance.xcom_pull(task_ids='fetch_optimization_params') }}}}'"
        ),
        # Exécute des commandes dbt pour générer les indicateurs dans la couche Gold
    )

    # 3. Export des données Gold vers BigQuery
    task_export_bq = BashOperator(
        task_id='export_gold_to_bigquery',
        bash_command=f'{PREFIX_SPARK} && python3 {AIRFLOW_HOME}/src/data_engineering/prod/gold/export_gold_to_bq.py',
        # Exécute un script Python pour exporter les données Gold vers BigQuery
    )

    # Dépendances
    fetch_params >> task_generate_features >> task_export_bq
