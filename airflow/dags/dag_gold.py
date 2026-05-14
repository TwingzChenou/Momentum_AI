from airflow import DAG
from airflow.operators.bash import BashOperator
from utils.config import (
    AIRFLOW_HOME, PREFIX_SPARK, DEFAULT_ARGS
)

with DAG(
    '03_prod_gold_features',
    default_args=DEFAULT_ARGS,
    description='Pipeline Gold Momentum AI unifié (Full Spark-SQL)',
    schedule_interval=None,
    max_active_runs=1,
    catchup=False,
    tags=['prod', 'gold', 'spark'],
) as dag:

    # Tâche unifiée en Pure Spark-SQL
    # Remplace dbt run, dbt test et l'export manuel vers BigQuery
    generate_features = BashOperator(
        task_id='generate_indicators_gold_spark',
        bash_command=f'{PREFIX_SPARK} && python3 {AIRFLOW_HOME}/src/data_engineering/prod/gold/compute_gold_spark.py',
    )

    generate_features
