from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Ajout du chemin projet pour les imports
PROJECT_DIR = "/app"
sys.path.append(PROJECT_DIR)

default_args = {
    'owner': 'momentum_ai',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_optuna_optimization():
    """
    Lance le script d'optimisation via le conteneur
    """
    from src.optimization.strategy_optimizer import run_optimization
    run_optimization(n_trials=50)

with DAG(
    'strategy_optimization_weekly',
    default_args=default_args,
    description='Optimisation hebdomadaire de la stratégie Momentum via Optuna et MLFlow',
    schedule_interval='@weekly', # Tourne une fois par semaine
    catchup=False,
    tags=['momentum', 'optimization', 'mlflow'],
) as dag:

    optimize_task = PythonOperator(
        task_id='run_strategy_optimization',
        python_callable=run_optuna_optimization,
    )

    optimize_task
