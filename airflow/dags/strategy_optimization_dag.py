from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from utils.config import DEFAULT_ARGS
import os
import sys

# Ajout du chemin projet pour les imports (Airflow utilise /opt/airflow)
PROJECT_DIR = "/opt/airflow"
sys.path.append(PROJECT_DIR)

from airflow.operators.bash import BashOperator

def run_optuna_optimization():
    """
    Lance le script d'optimisation via le conteneur
    """
    from src.optimization.strategy_optimizer import run_optimization
    run_optimization(n_trials=200)

with DAG(
    'strategy_optimization_weekly',
    default_args=DEFAULT_ARGS,
    description='Optimisation hebdomadaire de la stratégie Momentum via Optuna et MLFlow',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=['momentum', 'optimization', 'mlflow'],
) as dag:

    optimize_task = PythonOperator(
        task_id='run_strategy_optimization',
        python_callable=run_optuna_optimization,
    )

    trigger_gold = TriggerDagRunOperator(
        task_id='trigger_gold_layer',
        trigger_dag_id='03_prod_gold_features',
        wait_for_completion=False,
    )

    optimize_task >> trigger_gold
