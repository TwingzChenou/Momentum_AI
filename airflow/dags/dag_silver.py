import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from utils.config import (
    AIRFLOW_HOME, PREFIX_SPARK, DEFAULT_ARGS
)

with DAG(
    '02_prod_silver_processing',
    default_args=DEFAULT_ARGS,
    description='Pipeline de nettoyage et rééchantillonnage Silver (Full Spark SQL)',
    schedule_interval=None,
    max_active_runs=1,
    catchup=False,
    tags=['prod', 'silver', 'spark'],
) as dag:

    # Tâche unifiée en Pure Spark-SQL
    # Remplace les 12 tâches précédentes (dbt daily, spark weekly, spark monthly pour chaque type)
    unified_resampling = BashOperator(
        task_id='unified_silver_resampling',
        bash_command=f'{PREFIX_SPARK} && python3 {AIRFLOW_HOME}/src/data_engineering/prod/silver/resample_all_spark.py',
    )

    # Trigger vers la couche Gold
    trigger_gold_layer = TriggerDagRunOperator(
        task_id='trigger_gold_layer',
        trigger_dag_id='03_prod_gold_features',
        wait_for_completion=False,
    )

    unified_resampling >> trigger_gold_layer